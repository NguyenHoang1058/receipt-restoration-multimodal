"""MC-OCR 2021 -> parsed -> index-aligned -> Master Schema 1.0.0.

Ground-truth rules:
* Never infer 'not_present' from missing source annotation.
* Never silently repair schema output or a suspicious source annotation.
* Keep the complete original segment evidence in a separate audit sidecar.
* Unknown/ambiguous monetary and date values are NOT guessed.

Run: python mcocr_steps_2_4_revised.py --image-dir data/raw/mc_ocr_2021/images
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

LOG = logging.getLogger("mcocr")
FIELDS = {"SELLER": "store_name", "ADDRESS": "store_address", "TIMESTAMP": "transaction_datetime", "TOTAL_COST": "total_amount"}
RAW_KEYS = {"img_id", "anno_polygons", "anno_texts", "anno_labels", "anno_num", "anno_image_quality"}
# MC-OCR category IDs are NOT assumed: supply a verified mapping with --category-map.


class MissingSourceAnnotationError(ValueError):
    """The original dataset has no text annotation for a sample."""


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace a JSONL report atomically to avoid duplicates on rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_issue(sample_id: str, issue_type: str, image_id: str | None,
               source_json_path: Path, image_dir: Path,
               reason: str, *, usable_for_ie_training: bool,
               warnings: list[str] | None = None) -> dict[str, Any]:
    """Create a machine-readable issue record without copying raw images."""
    name = Path(image_id.replace("\\", "/")).name if image_id else None
    return {
        "sample_id": sample_id,
        "dataset": "mc_ocr_2021",
        "issue_type": issue_type,
        "source_json_path": source_json_path.as_posix(),
        "image_id": image_id,
        "image_path": (image_dir / name).as_posix() if name else None,
        "reason": reason,
        "review_status": "pending",
        "usable_for_ie_training": usable_for_ie_training,
        "warnings": warnings or [],
    }


def parse_polygons(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = ast.literal_eval(value) if value.strip() else []
    if not isinstance(value, list) or any(not isinstance(p, dict) for p in value):
        raise ValueError("anno_polygons must be a list of dictionaries (or its Python literal)")
    return value


def split_text(value: Any, field: str) -> list[str]:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string (empty string allowed), got {type(value).__name__}")
    return value.split("|||") if value else []


def parse_record(raw: dict[str, Any], sample_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("source record must be a JSON object")
    missing = RAW_KEYS - raw.keys()
    if missing:
        raise ValueError(f"missing source keys: {sorted(missing)}")
    if not isinstance(raw["img_id"], str) or not raw["img_id"].strip():
        raise ValueError("img_id must be a nonempty string")
    if raw["anno_texts"] is None:
        raise MissingSourceAnnotationError("source CSV has no anno_texts (null)")
    polys = parse_polygons(raw["anno_polygons"])
    texts = split_text(raw["anno_texts"], "anno_texts")
    labels = split_text(raw["anno_labels"], "anno_labels")
    warnings = []
    # Some source exports store anno_num as a number-like scalar; validate without guessing its semantics.
    num = raw["anno_num"]
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        warnings.append("anno_num_unexpected_type_or_range")
    elif num != len(labels):
        warnings.append("anno_num_differs_from_label_count_check_source_semantics")
    quality = raw["anno_image_quality"]
    if isinstance(quality, bool) or not isinstance(quality, (float, int)) or not math.isfinite(quality):
        warnings.append("source_image_quality_invalid")
    if len(texts) != len(labels):
        warnings.append("text_label_count_mismatch")
    if len(polys) != len(labels):
        warnings.append("polygon_label_count_mismatch")
    if any(label.strip() not in FIELDS for label in labels):
        warnings.append("unknown_source_labels")
    if any("Ã" in text or "á»" in text or "Â" in text for text in texts):
        warnings.append("possible_mojibake_review_original_encoding")
    return {"sample_id": sample_id, "img_id": raw["img_id"], "anno_num": num,
            "anno_image_quality": quality, "polygons": polys, "texts": texts,
            "labels": labels, "warnings": sorted(set(warnings))}


def polygon_bbox(poly: dict[str, Any], width: int, height: int) -> tuple[list[int] | None, str]:
    """Do not clamp: raw geometries outside image go to human review."""
    seg = poly.get("segmentation")
    contours: list[list[Any]] = []
    if isinstance(seg, list) and seg:
        contours = [seg] if all(isinstance(v, (int, float)) for v in seg) else seg
    coords: list[tuple[float, float]] = []
    if contours:
        for contour in contours:
            if not isinstance(contour, list) or len(contour) < 6 or len(contour) % 2:
                return None, "invalid_segmentation"
            try:
                pairs = [(float(contour[i]), float(contour[i + 1])) for i in range(0, len(contour), 2)]
            except (TypeError, ValueError, OverflowError):
                return None, "invalid_segmentation"
            if not all(math.isfinite(x) and math.isfinite(y) for x, y in pairs):
                return None, "nonfinite_segmentation"
            coords.extend(pairs)
    if not coords:
        bbox = poly.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None, "missing_geometry"
        try:
            x, y, w, h = [float(v) for v in bbox]
        except (ValueError, TypeError, OverflowError):
            return None, "invalid_source_bbox"
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return None, "invalid_source_bbox"
        coords = [(x, y), (x + w, y + h)]
    x1 = math.floor(min(x for x, _ in coords))
    y1 = math.floor(min(y for _, y in coords))
    x2 = math.ceil(max(x for x, _ in coords))
    y2 = math.ceil(max(y for _, y in coords))
    if x2 <= x1 or y2 <= y1:
        return None, "degenerate_bbox"
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None, "out_of_image_bounds"
    return [x1, y1, x2, y2], "valid"

def detect_repeated_phrase(text: str | None) -> dict | None:
    """
    Detect consecutive repeated phrases.
    Only suggests corrections; never modifies source text.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    words = text.split()
    max_phrase_len = min(5, len(words) // 2)

    for length in range(max_phrase_len, 0, -1):
        for start in range(len(words) - 2 * length + 1):
            first = words[start:start + length]
            second = words[start + length:start + 2 * length]

            if [w.casefold() for w in first] == [
                w.casefold() for w in second
            ]:
                corrected = (
                    words[:start + length]
                    + words[start + 2 * length:]
                )

                return {
                    "issue_type": "suspected_repeated_phrase",
                    "original_text": text,
                    "repeated_phrase": " ".join(second),
                    "suggested_text": " ".join(corrected),
                    "review_status": "pending"
                }

    return None

def align_record(parsed: dict[str, Any], image_path: Path, category_map: dict[int, str]) -> dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
        orientation = img.getexif().get(274)
        img.verify()
    warnings = list(parsed["warnings"])
    if orientation not in (None, 1):
        warnings.append("nonstandard_exif_orientation")
    polys, texts, labels = parsed["polygons"], parsed["texts"], parsed["labels"]
    # Positional linkage is used ONLY when all list lengths agree.
    counts_match = len(polys) == len(texts) == len(labels)
    text_label_match = len(texts) == len(labels)
    segments = []
    for index in range(max(len(polys), len(texts), len(labels))):
        poly = polys[index] if index < len(polys) else None
        label = labels[index].strip() if index < len(labels) else None
        original_text = texts[index] if index < len(texts) else None
        # Detect possible repeated phrases in original annotations
        repeat_issue = detect_repeated_phrase(original_text)
        if repeat_issue is not None:
            warnings.append(
                f"segment_{index}:suspected_repeated_phrase"
            )
        bbox, status = None, "unverified_count_mismatch"
        if counts_match and poly is not None:
            cat_id = poly.get("category_id")
            if category_map and cat_id in category_map and category_map[cat_id] != label:
                status = "category_label_mismatch"
            else:
                bbox, status = polygon_bbox(poly, width, height)
        if status != "valid":
            warnings.append(f"segment_{index}:{status}")
        segments.append({"index": index, "label": label, "raw_text": original_text,
                         "text": original_text.strip() if original_text is not None else None,
                         "polygon": poly, "bbox": bbox, "geometry_status": status, "text_quality_issue": repeat_issue})
    return {"sample_id": parsed["sample_id"], "img_id": parsed["img_id"],
            "image_width": width, "image_height": height, "exif_orientation": orientation,
            "source_quality_score": parsed["anno_image_quality"], "source_anno_num": parsed["anno_num"],
            "text_label_match": text_label_match, "counts_match": counts_match,
            "segments": segments, "warnings": sorted(set(warnings))}


# All alternatives must match the whole candidate string. No inference from the *last* colon.
_AMOUNT = re.compile(r"[+-]?(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d{2})?")
_CURRENCY = re.compile(r"(?i)(?:vnd|vnđ|đồng|dong|₫|\bđ\b)")


def parse_vnd(text: str) -> int | None:
    """Conservative VND parser; a unique monetary candidate is required."""
    stripped = _CURRENCY.sub(" ", text)
    candidates = _AMOUNT.findall(stripped)
    if len(candidates) != 1:
        return None
    token = candidates[0]
    # Reject attached or unparsed digits, so malformed/compound numeric data is not accepted.
    remainder = stripped.replace(token, "", 1)
    if re.search(r"\d", remainder) or token.startswith("-"):
        return None
    token = token.lstrip("+")
    # Mixed whitespace and punctuation grouping is not safely interpretable.
    if " " in token and ("," in token or "." in token):
        return None
    # A trailing fractional '.00' or ',00' after a grouped number may denote decimals.
    if re.search(r"[.,]\d{2}$", token):
        whole, frac = token[:-3], token[-2:]
        if frac != "00" or not re.search(r"[., ]\d{3}", whole):
            return None  # Do not guess whether 1,00 means 1 or 100.
        token = whole
    # A lone '1,00' is ambiguous, not auto-decoded as 1 VND.
    if not re.fullmatch(r"(?:\d{1,3}(?:[., ]\d{3})+|\d+)", token):
        return None
    return int(re.sub(r"[., ]", "", token))


def parse_datetime(text: str, date_order: str) -> tuple[str | None, str | None]:
    """Return (ISO date/time or None, reason). Explicit DMY/YMD policy prevents guessing."""
    if date_order not in {"DMY", "YMD"}:
        raise ValueError("date_order must be DMY or YMD")
    date_match = re.search(r"(?<!\d)(\d{1,4})[/.-](\d{1,2})[/.-](\d{2,4})(?!\d)", text)
    verbal = re.search(r"(?i)ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", text)
    if verbal:
        day, month, year = map(int, verbal.groups())
    elif date_match:
        a, b, c = map(int, date_match.groups())
        if len(date_match.group(1)) == 4:
            year, month, day = a, b, c
        elif len(date_match.group(3)) == 4 and date_order == "DMY":
            day, month, year = a, b, c
        else:
            return None, "date_format_unresolved"
    else:
        return None, "date_not_found"
    # Do not select the date's separators as times; seconds and AM/PM optional.
    clock = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", text)
    try:
        if clock:
            hour = int(clock.group(1)); minute = int(clock.group(2)); second = int(clock.group(3) or 0)
            dt = datetime(year, month, day, hour, minute, second)
            return dt.strftime("%Y-%m-%dT%H:%M:%S"), None
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d"), None
    except ValueError:
        return None, "invalid_datetime"


def empty_field() -> dict[str, Any]:
    return {"status": "not_evaluable", "value": None, "raw_text": None, "bbox": None}


def merged_bbox(segments: list[dict[str, Any]]) -> list[int] | None:
    if not segments or any(s["bbox"] is None for s in segments):
        return None
    b = [s["bbox"] for s in segments]
    return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]


def normalize_record(aligned: dict[str, Any], image_path: Path, root: Path,
                     source_split: str | None, date_order: str,
                     validator: Draft202012Validator) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        relative_image = image_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("image path outside --project-root; use matching root") from exc
    warnings = list(aligned["warnings"])
    grouped = {target: [] for target in FIELDS.values()}
    # If source text-label pairing is broken, do not build false semantic GT.
    if aligned["text_label_match"]:
        for segment in aligned["segments"]:
            if segment["label"] in FIELDS:
                grouped[FIELDS[segment["label"]]].append(segment)
    else:
        warnings.append("semantic_mapping_blocked_by_text_label_mismatch")

    provenance = {}
    mapped = {}
    for field in FIELDS.values():
        original = grouped[field]
        nonempty = [s for s in original if s["text"]]
        selected = nonempty
        value = None
        error = None
        if field == "total_amount":
            # Individual source segments are candidates. Never join two amounts.
            amounts = [(s, parse_vnd(s["text"])) for s in nonempty]
            amounts = [(s, n) for s, n in amounts if n is not None]
            if len(amounts) == 1:
                selected, value = [amounts[0][0]], amounts[0][1]
            else:
                error = "total_amount_missing_or_ambiguous_candidates"
        elif field == "transaction_datetime":
            if len(nonempty) == 1:
                value, error = parse_datetime(nonempty[0]["text"], date_order)
            else:
                error = "datetime_missing_or_multiple_segments"
        elif nonempty:
            # Preserve original indexing order, without spatial heuristic reordering.
            value = " ".join(s["text"] for s in nonempty)
        else:
            error = "missing_or_empty_source_annotation"
        if value is None:
            mapped[field] = empty_field()
            if error:
                warnings.append(f"{field}:{error}")
        else:
            mapped[field] = {"status": "annotated", "value": value,
                             "raw_text": " ".join(s["text"] for s in selected),
                             "bbox": merged_bbox(selected)}
        provenance[field] = {"original_segments": original,
                             "selected_indices": [s["index"] for s in selected],
                             "normalization_issue": error}

    payload = {
        "schema_version": "1.0.0", "sample_id": aligned["sample_id"],
        "source": {"dataset": "mc_ocr_2021", "source_id": aligned["img_id"],
                   "source_split": source_split, "group_id": None},
        "document": {"language": "vi", "currency": "VND", "images": {
            "original": {"path": relative_image, "width": aligned["image_width"],
                         "height": aligned["image_height"]}, "degraded": None, "restored": None}},
        "annotations": {"store_name": mapped["store_name"], "store_address": mapped["store_address"],
                        "transaction_datetime": mapped["transaction_datetime"],
                        "receipt_number": empty_field(), "items": {"status": "not_evaluable", "rows": None},
                        "discount_or_tax": empty_field(), "total_amount": mapped["total_amount"]},
        "ocr": {"engine": None, "engine_version": None, "tokens": []},
        "quality": {"difficulty": None}, "project_split": None,
        "annotation_meta": {"annotation_status": "draft", "annotator_id": "mc_ocr_2021_source",
                            "reviewer_id": None, "updated_at": None}
    }
    validator.validate(payload)  # No mutation, no missing schema fallback.
    audit = {"sample_id": aligned["sample_id"], "source_id": aligned["img_id"],
             "needs_review": bool(warnings), "warnings": sorted(set(warnings)),
             "source_quality_score": aligned["source_quality_score"],
             "source_anno_num": aligned["source_anno_num"],
             "exif_orientation": aligned["exif_orientation"], "fields": provenance,
             "all_segments": aligned["segments"]}
    return payload, audit


def write_json(path: Path, data: Any) -> None:
    """Atomic writes, so interrupted runs do not leave truncated JSON files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(args: argparse.Namespace) -> dict[str, int]:
    if not args.schema_path.is_file():
        raise FileNotFoundError(f"required schema file missing: {args.schema_path}")
    if not args.raw_json_dir.is_dir() or not args.image_dir.is_dir():
        raise FileNotFoundError("raw JSON or image directory missing")
    with args.schema_path.open(encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    category_map: dict[int, str] = {}
    if args.category_map is not None:
        with args.category_map.open(encoding="utf-8") as f:
            mapping = json.load(f)
        category_map = {int(k): v for k, v in mapping.items()}
        if any(v not in FIELDS for v in category_map.values()):
            raise ValueError("category map values must be known MC-OCR label names")
    files = sorted(args.raw_json_dir.glob("*.json"))
    if not files:
        raise ValueError("no source JSON files")
    stats = Counter(total=len(files))
    audit_samples: list[dict[str, Any]] = []
    issue_records: list[dict[str, Any]] = []
    missing_source_ids: list[str] = []
    run_time = datetime.now(timezone.utc).isoformat()

    for file in files:
        sid = file.stem
        parsed_file = args.parsed_dir / f"{sid}.json"
        aligned_file = args.aligned_dir / f"{sid}.json"
        normalized_file = args.normalized_dir / f"{sid}.json"
        audit_file = args.audit_dir / f"{sid}.json"
        raw: dict[str, Any] | None = None
        try:
            with file.open(encoding="utf-8") as f:
                raw = json.load(f)
            parsed = parse_record(raw, sid)
            write_json(parsed_file, parsed)
            image_name = Path(parsed["img_id"].replace("\\", "/")).name
            image_path = args.image_dir / image_name
            aligned = align_record(parsed, image_path, category_map)
            write_json(aligned_file, aligned)
            normalized, audit = normalize_record(
                aligned, image_path, args.project_root,
                args.source_split, args.date_order, validator
            )
            write_json(normalized_file, normalized)
            write_json(audit_file, audit)
            audit_samples.append({"sample_id": sid, "needs_review": audit["needs_review"],
                                  "warnings": audit["warnings"]})
            if audit["needs_review"]:
                stats["review"] += 1

                issue_record = make_issue(
                    sid,
                    "annotation_needs_review",
                    raw["img_id"],
                    file,
                    args.image_dir,
                    "Annotation quality warnings; inspect audit sidecar",
                    usable_for_ie_training=False,
                    warnings=audit["warnings"]
                )

                issue_record["text_quality_issues"] = [
                    {
                        "segment_index": seg["index"],
                        "label": seg["label"],
                        **seg["text_quality_issue"]
                    }
                    for seg in aligned["segments"]
                    if seg.get("text_quality_issue") is not None
                ]

                issue_records.append(issue_record)
            else:
                stats["clean"] += 1
        except MissingSourceAnnotationError as exc:
            stats["source_annotation_missing"] += 1
            missing_source_ids.append(sid)
            for stale in (parsed_file, aligned_file, normalized_file, audit_file):
                stale.unlink(missing_ok=True)
            issue_records.append(make_issue(
                sid, "source_annotation_missing",
                raw.get("img_id") if isinstance(raw, dict) else None,
                file, args.image_dir, str(exc), usable_for_ie_training=False,
                warnings=["missing_anno_texts"]
            ))
            audit_samples.append({"sample_id": sid, "failed": True,
                                  "error_type": "source_annotation_missing", "error": str(exc)})
            LOG.warning("%s: missing source annotation: %s", sid, exc)
        except Exception as exc:
            stats["pipeline_errors"] += 1
            for stale in (parsed_file, aligned_file, normalized_file, audit_file):
                stale.unlink(missing_ok=True)
            issue_records.append(make_issue(
                sid, "pipeline_error",
                raw.get("img_id") if isinstance(raw, dict) else None,
                file, args.image_dir, f"{type(exc).__name__}: {exc}",
                usable_for_ie_training=False
            ))
            audit_samples.append({"sample_id": sid, "failed": True,
                                  "error_type": "pipeline_error",
                                  "error": f"{type(exc).__name__}: {exc}"})
            LOG.error("%s: %s", sid, exc)

    # Guarantee a stable report schema, including zero counts.
    counts = {
        "total": len(files),
        "clean": stats["clean"],
        "review": stats["review"],
        "source_annotation_missing": stats["source_annotation_missing"],
        "pipeline_errors": stats["pipeline_errors"],
        "failed": stats["source_annotation_missing"] + stats["pipeline_errors"],
    }
    assert sum(counts[k] for k in ("clean", "review", "source_annotation_missing", "pipeline_errors")) == len(files)
    summary = {
        "dataset": "mc_ocr_2021", "pipeline": "steps_2_4",
        "run_at_utc": run_time,
        "counts": counts,
        "percentages": {
            key: round(100 * counts[key] / len(files), 2)
            for key in ("clean", "review", "source_annotation_missing", "pipeline_errors")
        },
        "samples": audit_samples,
    }
    write_json(args.audit_dir / "summary.json", summary)
    write_jsonl(args.audit_dir / "sample_issues.jsonl", issue_records)
    write_json(args.audit_dir / "excluded_samples.json", {
        "dataset": "mc_ocr_2021",
        "purpose": "supervised_information_extraction_before_manual_review",
        "excluded_sample_ids": sorted({issue["sample_id"] for issue in issue_records}),
        "source_annotation_missing_ids": sorted(missing_source_ids),
        "note": "No raw images are deleted. Exclusion is task-specific, pending human review."
    })
    LOG.info("Completed: %s", counts)
    LOG.info("Issue reports written to %s", args.audit_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json-dir", type=Path, default=Path("data/interim/mc_ocr_2021/raw_json"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/interim/mc_ocr_2021/parsed_annotations"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/interim/mc_ocr_2021/aligned_annotations"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/interim/mc_ocr_2021/normalized_annotations"))
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/data_quality/mc_ocr_2021"))
    parser.add_argument("--schema-path", type=Path, default=Path("schemas/receipt.schema.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-split", default=None, help="Actual source split if verified; otherwise null")
    parser.add_argument("--date-order", choices=["DMY", "YMD"], default="DMY",
                        help="Use DMY only when confirmed for this Vietnamese source; unknown cases go to review")
    parser.add_argument("--category-map", type=Path, default=None, help="Verified category_id-to-label JSON map")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
