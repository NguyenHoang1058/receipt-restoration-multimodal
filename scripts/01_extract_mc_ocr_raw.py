"""Export MC-OCR CSV records to one JSON per image, with extraction audit.

Step 1 only: no parsing of anno_polygons, no label alignment or normalization.
Requires Python 3.9+; uses only the standard library.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("mcocr.extract")

REQUIRED_COLUMNS = (
    "img_id", "anno_polygons", "anno_texts", "anno_labels",
    "anno_num", "anno_image_quality",
)
ANNOTATION_COLUMNS = ("anno_polygons", "anno_texts", "anno_labels")


def atomic_json_dump(path: Path, value: Any) -> None:
    """Write JSON atomically, replacing only the exact same output filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_jsonl_dump(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def extract_sample_id(img_id: str) -> str:
    """Accept a filename (not a directory path); reject unsafe/ambiguous IDs."""
    if not img_id or img_id != img_id.strip():
        raise ValueError("img_id is empty or has surrounding whitespace")
    if "/" in img_id or "\\" in img_id:
        raise ValueError("img_id must be an image filename, not a path")
    name = Path(img_id)
    if name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("img_id must end with .jpg, .jpeg or .png")
    sample_id = name.stem
    if not sample_id or sample_id in {".", ".."} or not re.fullmatch(r"[\w.\-]+", sample_id):
        raise ValueError("img_id contains an unsafe sample_id")
    return sample_id


def parse_optional_integer(value: str, field: str, warnings: list[str]) -> int | None:
    if value.strip() == "":
        warnings.append(f"missing_{field}")
        return None
    try:
        result = int(value.strip())
    except ValueError:
        warnings.append(f"invalid_{field}")
        return None
    if result < 0:
        warnings.append(f"invalid_{field}")
        return None
    return result


def parse_optional_float(value: str, field: str, warnings: list[str]) -> float | None:
    if value.strip() == "":
        warnings.append(f"missing_{field}")
        return None
    try:
        result = float(value.strip())
    except ValueError:
        warnings.append(f"invalid_{field}")
        return None
    if not math.isfinite(result):
        warnings.append(f"invalid_{field}")
        return None
    return result


def extract_csv_to_raw_json(
    csv_path: Path,
    output_dir: Path,
    report_dir: Path,
    *,
    allow_existing_outputs: bool = True,
) -> dict[str, Any]:
    """Export CSV rows without changing annotation text/polygon syntax.

    Blank annotation cells become JSON null (not fabricated empty annotations).
    Nonblank text and polygons are preserved *as strings* for Step 2 to parse.
    Numeric metadata is converted to int/float or null, with audit warnings.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    issue_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    written: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV is empty or missing a header")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for csv_row_number, row in enumerate(reader, start=2):
            counts["total_rows"] += 1
            warnings: list[str] = []
            sample_id: str | None = None
            img_id = row.get("img_id")
            try:
                if None in row:  # csv.DictReader uses None for excess columns
                    raise ValueError("Row has more columns than CSV header")
                if any(row.get(k) is None for k in REQUIRED_COLUMNS):
                    raise ValueError("Malformed row: missing one or more CSV cells")
                sample_id = extract_sample_id(str(img_id))
                if sample_id in seen:
                    counts["duplicate_sample_id"] += 1
                    issue_records.append({
                        "csv_row": csv_row_number,
                        "sample_id": sample_id,
                        "img_id": img_id,
                        "status": "skipped",
                        "issues": ["duplicate_sample_id"],
                    })
                    continue
                seen.add(sample_id)

                record: dict[str, Any] = dict(row)
                for field in ANNOTATION_COLUMNS:
                    value = record[field]
                    # Preserve every nonblank character exactly as read from CSV.
                    if value is None or value.strip() == "":
                        record[field] = None
                        warnings.append(f"missing_{field}")

                record["anno_num"] = parse_optional_integer(
                    record["anno_num"], "anno_num", warnings
                )
                record["anno_image_quality"] = parse_optional_float(
                    record["anno_image_quality"], "anno_image_quality", warnings
                )

                output_file = output_dir / f"{sample_id}.json"
                if not allow_existing_outputs and output_file.exists():
                    raise FileExistsError(f"Output already exists: {output_file}")
                atomic_json_dump(output_file, record)
                written.append(sample_id)
                counts["exported"] += 1
                if warnings:
                    counts["exported_with_issues"] += 1
                    issue_records.append({
                        "csv_row": csv_row_number,
                        "sample_id": sample_id,
                        "img_id": img_id,
                        "status": "exported_with_issues",
                        "issues": warnings,
                    })
                    for warning in set(warnings):
                        counts[warning] += 1
            except (ValueError, TypeError, OSError) as exc:
                counts["failed_rows"] += 1
                LOGGER.error("CSV row %d failed: %s", csv_row_number, exc)
                issue_records.append({
                    "csv_row": csv_row_number,
                    "sample_id": sample_id,
                    "img_id": img_id,
                    "status": "failed",
                    "issues": ["extraction_error"],
                    "error": str(exc),
                })

    summary = {
        "dataset": "mc_ocr_2021",
        "stage": "01_csv_to_raw_json",
        "source_csv": csv_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "counts": dict(sorted(counts.items())),
        "note": "Issue counts overlap; don't add them to obtain unique affected samples.",
    }
    atomic_json_dump(report_dir / "extraction_summary.json", summary)
    atomic_jsonl_dump(report_dir / "extraction_issues.jsonl", issue_records)
    LOGGER.info("Completed: %s", summary["counts"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-OCR CSV -> raw JSON (Step 1)")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("data/interim/mc_ocr_2021/raw_json"),
    )
    parser.add_argument(
        "--report_dir", type=Path,
        default=Path("reports/data_quality/mc_ocr_2021"),
    )
    parser.add_argument(
        "--no_overwrite", action="store_true",
        help="Fail a row instead of overwriting an existing JSON file",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    extract_csv_to_raw_json(
        args.csv_path, args.output_dir, args.report_dir,
        allow_existing_outputs=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
