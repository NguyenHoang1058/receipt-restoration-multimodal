import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set
import cv2
from jsonschema import Draft202012Validator
from augraphy import (
    AugraphyPipeline,
    Brightness,
    DirtyRollers,
    Folding,
    InkBleed,
    LightingGradient,
    LowInkRandomLines
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_augraphy_pipeline() -> AugraphyPipeline:
    """Create a realistic physical receipt degradation pipeline."""
    ink_phase = [
        InkBleed(intensity_range=(0.1, 0.3), p=0.4),
        LowInkRandomLines(count_range=(2, 5), p=0.3)
    ]
    paper_phase = [
        Folding(fold_count=2, fold_noise=0.05, gradient_width=(0.01, 0.05), p=0.4),
        DirtyRollers(line_width_range=(2, 6), p=0.2)
    ]
    post_phase = [
        LightingGradient(light_position=None, direction=None, max_brightness=240, min_brightness=120, p=0.5),
        Brightness(brightness_range=(0.85, 1.15), p=0.4)
    ]
    return AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)


def augment_and_validate(
    input_dir: Path,
    paired_img_dir: Path,
    final_output_dir: Path,
    schema_path: Path,
    report_path: Path
) -> None:
    if not schema_path.is_file():
        raise FileNotFoundError(f"Master JSON Schema not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as f:
        schema_obj = json.load(f)
    validator = Draft202012Validator(schema_obj)

    pipeline = build_augraphy_pipeline()
    final_output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    split_groups: Dict[str, Set[str]] = {"train": set(), "validation": set(), "test": set()}
    stats = {"total": 0, "train": 0, "validation": 0, "test": 0, "easy": 0, "medium": 0, "hard": 0}

    for filepath in files:
        with filepath.open("r", encoding="utf-8") as f:
            record = json.load(f)

        sample_id = record["sample_id"]
        split_name = record["project_split"]
        group_id = record["source"]["group_id"]

        if split_name not in split_groups:
            raise ValueError(f"[{sample_id}] Invalid project_split: {split_name}")

        orig_info = record["document"]["images"]["original"]
        orig_path = Path(orig_info["path"])
        img = cv2.imread(str(orig_path))
        if img is None:
            raise FileNotFoundError(f"[{sample_id}] Cannot read original image: {orig_path}")

        # Generate degraded image inside split-specific folder
        degraded_img = pipeline.augment(img)["output"]
        deg_dir = paired_img_dir / split_name / "degraded"
        deg_dir.mkdir(parents=True, exist_ok=True)
        deg_path = deg_dir / f"{sample_id}_deg.jpg"
        cv2.imwrite(str(deg_path), degraded_img)

        h, w = degraded_img.shape[:2]
        record["document"]["images"]["degraded"] = {
            "path": str(deg_path.resolve()),
            "width": int(w),
            "height": int(h)
        }

        if record["annotation_meta"]["annotation_status"] != "reviewed":
            record["annotation_meta"]["annotation_status"] = "final"
        record["annotation_meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Validate strictly against Draft 2020-12 Schema
        validator.validate(record)

        split_groups[split_name].add(group_id)
        stats["total"] += 1
        stats[split_name] += 1
        stats[record["quality"]["difficulty"]] += 1

        out_json = final_output_dir / filepath.name
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4, allow_nan=False)

    # Zero Data Leakage Assertion
    leak_tv = split_groups["train"] & split_groups["validation"]
    leak_tt = split_groups["train"] & split_groups["test"]
    leak_vt = split_groups["validation"] & split_groups["test"]
    if leak_tv or leak_tt or leak_vt:
        raise RuntimeError(f"Data leakage detected across splits! Overlapping groups: {leak_tv | leak_tt | leak_vt}")

    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset_version": "1.0.0", "leakage_check": "PASSED", "statistics": stats}, f, indent=4)

    logger.info("Dataset v1.0 validated and exported successfully. Stats: %s", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Augraphy Augmentation and Final Validation.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/interim/mc_ocr_2021/bio_annotations"))
    parser.add_argument("--paired_img_dir", type=Path, default=Path("data/paired/mc_ocr_2021"))
    parser.add_argument("--final_output_dir", type=Path, default=Path("data/processed/mc_ocr_2021/annotations"))
    parser.add_argument("--schema_path", type=Path, default=Path("schemas/receipt.schema.json"))
    parser.add_argument("--report_path", type=Path, default=Path("reports/dataset_statistics/mc_ocr_v1_report.json"))
    args = parser.parse_args()

    augment_and_validate(
        args.input_dir,
        args.paired_img_dir,
        args.final_output_dir,
        args.schema_path,
        args.report_path
    )