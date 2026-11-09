import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_to_label_studio(normalized_dir: Path, output_file: Path, limit: int = 350) -> None:
    """
    Export normalized JSON annotations into a single Label Studio import payload.
    """
    files = sorted(normalized_dir.glob("*.json"))[:limit]
    tasks: List[Dict[str, Any]] = []

    for filepath in files:
        with filepath.open("r", encoding="utf-8") as f:
            record = json.load(f)

        img_info = record["document"]["images"]["original"]
        ann = record["annotations"]

        tasks.append({
            "data": {
                "sample_id": record["sample_id"],
                "image": img_info["path"],
                "store_name": ann["store_name"]["raw_text"] or "",
                "store_address": ann["store_address"]["raw_text"] or "",
                "transaction_datetime": ann["transaction_datetime"]["raw_text"] or "",
                "total_amount": ann["total_amount"]["raw_text"] or ""
            }
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info("Exported %d tasks for Label Studio to %s", len(tasks), output_file)


def import_from_label_studio(
    ls_export_path: Path,
    normalized_dir: Path,
    detailed_dir: Path,
    reviewer_id: str
) -> None:
    """
    Merge Label Studio reviewed item rows back into the Master Schema JSONs.
    Expects Label Studio JSON export containing 'sample_id' in task['data']
    and structured item rows in task['annotations'][0]['result'].
    """
    detailed_dir.mkdir(parents=True, exist_ok=True)

    with ls_export_path.open("r", encoding="utf-8") as f:
        ls_tasks = json.load(f)

    reviewed_map: Dict[str, List[Dict[str, Any]]] = {}
    for task in ls_tasks:
        sample_id = task.get("data", {}).get("sample_id")
        annotations = task.get("annotations", [])
        if not sample_id or not annotations:
            continue

        # Extract custom JSON item rows entered or bounding boxes labeled in Label Studio
        rows: List[Dict[str, Any]] = []
        for res in annotations[0].get("result", []):
            if res.get("from_name") == "items_json" and "text" in res.get("value", {}):
                raw_json_str = res["value"]["text"][0]
                parsed_rows = json.loads(raw_json_str)
                rows.extend(parsed_rows)
        reviewed_map[sample_id] = rows

    for src_file in sorted(normalized_dir.glob("*.json")):
        with src_file.open("r", encoding="utf-8") as f:
            record = json.load(f)

        sample_id = record["sample_id"]
        if sample_id in reviewed_map and reviewed_map[sample_id]:
            record["annotations"]["items"] = {
                "status": "annotated",
                "rows": reviewed_map[sample_id]
            }
            record["annotation_meta"]["annotation_status"] = "reviewed"
            record["annotation_meta"]["reviewer_id"] = reviewer_id
            record["annotation_meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()

        out_path = detailed_dir / src_file.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4, allow_nan=False)

    logger.info("Merged Label Studio annotations into %s", detailed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Label Studio Bridge for MC-OCR Expansion.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    exp_parser = subparsers.add_parser("export")
    exp_parser.add_argument("--normalized_dir", type=Path, required=True)
    exp_parser.add_argument("--output_file", type=Path, default=Path("data/interim/mc_ocr_2021/ls_tasks.json"))
    exp_parser.add_argument("--limit", type=int, default=350)

    imp_parser = subparsers.add_parser("import")
    imp_parser.add_argument("--ls_export_path", type=Path, required=True)
    imp_parser.add_argument("--normalized_dir", type=Path, required=True)
    imp_parser.add_argument("--detailed_dir", type=Path, default=Path("data/interim/mc_ocr_2021/mc_ocr_detailed"))
    imp_parser.add_argument("--reviewer_id", type=str, default="hoang_si_nguyen")

    args = parser.parse_args()
    if args.mode == "export":
        export_to_label_studio(args.normalized_dir, args.output_file, args.limit)
    else:
        import_from_label_studio(args.ls_export_path, args.normalized_dir, args.detailed_dir, args.reviewer_id)