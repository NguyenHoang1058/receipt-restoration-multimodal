"""Prepare MC-OCR Step 5 review drafts from Step 4 sample_issues.jsonl.

Copies unmodified normalized JSONs to corrections/pending and writes a
review_queue.jsonl manifest. Nothing is marked as corrected or reviewed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, default=Path("reports/data_quality/mc_ocr_2021/sample_issues.jsonl"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/interim/mc_ocr_2021/normalized_annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/mc_ocr_2021/corrections/pending"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pending drafts; never use after manual editing")
    args = parser.parse_args()

    if not args.issues.is_file():
        parser.error(f"Issue report not found: {args.issues}")
    if not args.normalized_dir.is_dir():
        parser.error(f"Normalized directory not found: {args.normalized_dir}")

    issues_by_id: dict[str, dict] = {}
    with args.issues.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                parser.error(f"Invalid JSONL on line {line_num}: {exc}")
            if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
                parser.error(f"Missing sample_id on JSONL line {line_num}")
            sid = item["sample_id"]
            if not SAFE_ID.fullmatch(sid):
                parser.error(f"Unsafe sample_id on line {line_num}: {sid!r}")
            rec = issues_by_id.setdefault(sid, {"issue_types": [], "warnings": [], "source_issue_records": 0})
            rec["source_issue_records"] += 1
            if item.get("issue_type"):
                rec["issue_types"].append(item["issue_type"])
            rec["warnings"].extend(item.get("warnings") or [])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = Counter()
    manifest = []
    for sid, issue in sorted(issues_by_id.items()):
        source = args.normalized_dir / f"{sid}.json"
        target = args.output_dir / f"{sid}.json"
        row = {
            "sample_id": sid,
            "review_status": "pending",
            "issue_types": sorted(set(issue["issue_types"])),
            "warnings": sorted(set(issue["warnings"])),
            "normalized_source_path": source.as_posix(),
            "review_draft_path": None,
            "queue_status": None,
        }
        if not source.is_file():
            row["queue_status"] = "normalized_json_missing"
            status["normalized_json_missing"] += 1
        else:
            try:
                record = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or record.get("sample_id") != sid:
                    raise ValueError("sample_id mismatch or non-object JSON")
            except (json.JSONDecodeError, ValueError) as exc:
                row["queue_status"] = "invalid_normalized_json"
                row["detail"] = str(exc)
                status["invalid_normalized_json"] += 1
            else:
                row["review_draft_path"] = target.as_posix()
                if target.exists() and not args.overwrite:
                    row["queue_status"] = "already_exists_untouched"
                    status["already_exists_untouched"] += 1
                else:
                    shutil.copy2(source, target)
                    row["queue_status"] = "copied"
                    status["copied"] += 1
        manifest.append(row)

    manifest_path = args.output_dir.parent / "review_queue.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

    print(f"Unique review sample IDs: {len(issues_by_id)}")
    print(f"Copied: {status['copied']}")
    print(f"Already exists (preserved): {status['already_exists_untouched']}")
    print(f"Missing normalized JSON: {status['normalized_json_missing']}")
    print(f"Invalid normalized JSON: {status['invalid_normalized_json']}")
    print(f"Review manifest: {manifest_path}")
    print("NOTE: Copied files are review drafts, NOT verified corrections.")


if __name__ == "__main__":
    main()
