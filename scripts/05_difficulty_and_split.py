import json
import argparse
import hashlib
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_difficulty(image_path: Path, source_score: Any) -> str:
    """
    Combine Laplacian blur variance and MC-OCR anno_image_quality to classify difficulty.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Corrupted or unreadable image: {image_path}")

    blur_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    qual_score = float(source_score) if source_score is not None else 0.75

    if qual_score >= 0.80 and blur_var >= 150.0:
        return "easy"
    if qual_score < 0.65 or blur_var < 60.0:
        return "hard"
    return "medium"


def derive_group_id(store_name: Any, sample_id: str) -> str:
    """
    Generate a deterministic group_id based on normalized store_name to prevent template leakage.
    """
    if not store_name or not str(store_name).strip():
        return f"grp_singleton_{sample_id}"
    cleaned = re.sub(r"[^a-z0-9]", "", str(store_name).lower())
    if len(cleaned) < 3:
        return f"grp_singleton_{sample_id}"
    digest = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"grp_{digest}"


def assess_and_split(
    bio_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(bio_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No files found in {bio_dir}")

    groups: Dict[str, List[Path]] = defaultdict(list)
    records_cache: Dict[Path, Dict[str, Any]] = {}

    for filepath in files:
        with filepath.open("r", encoding="utf-8") as f:
            record = json.load(f)

        img_path = Path(record["document"]["images"]["original"]["path"])
        src_qual = record.get("quality", {}).get("source_quality_score")

        difficulty = compute_difficulty(img_path, src_qual)
        record["quality"] = {"difficulty": difficulty}

        store_val = record["annotations"]["store_name"].get("value")
        group_id = derive_group_id(store_val, record["sample_id"])
        record["source"]["group_id"] = group_id

        groups[group_id].append(filepath)
        records_cache[filepath] = record

    unique_groups = list(groups.keys())
    random.seed(42)
    random.shuffle(unique_groups)

    n_total = len(unique_groups)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])

    manifests: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}

    for gid, file_list in groups.items():
        split_name = "train" if gid in train_groups else ("validation" if gid in val_groups else "test")
        for fp in file_list:
            rec = records_cache[fp]
            rec["project_split"] = split_name
            manifests[split_name].append(rec["sample_id"])
            with fp.open("w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=4, allow_nan=False)

    for split_name, sample_ids in manifests.items():
        manifest_path = splits_dir / f"mc_ocr_{split_name}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump({"split": split_name, "count": len(sample_ids), "sample_ids": sorted(sample_ids)}, f, indent=4)

    logger.info(
        "Completed Step 7. Train: %d | Val: %d | Test: %d",
        len(manifests["train"]), len(manifests["validation"]), len(manifests["test"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Difficulty Assessment & Group Splitting.")
    parser.add_argument("--bio_dir", type=Path, default=Path("data/interim/mc_ocr_2021/bio_annotations"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    assess_and_split(args.bio_dir, args.splits_dir)