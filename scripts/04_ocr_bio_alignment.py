import json
import argparse
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENTITY_BIO_KEYS = {
    "store_name": "STORE_NAME",
    "store_address": "STORE_ADDRESS",
    "transaction_datetime": "DATETIME",
    "receipt_number": "RECEIPT_NUM",
    "total_amount": "TOTAL"
}


def similarity(a: str, b: str) -> float:
    """Compute normalized character similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def merge_boxes(boxes: List[List[int]]) -> Optional[List[int]]:
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes)
    ]


def align_and_tag_bio(
    tokens: List[Dict[str, Any]],
    annotations: Dict[str, Any],
    threshold: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Match OCR tokens against ground truth fields using sliding window fuzzy matching.
    Injects BIO tags into tokens and backfills missing ground truth bounding boxes.
    """
    for token in tokens:
        token["bio_tag"] = "O"

    for field_key, bio_label in ENTITY_BIO_KEYS.items():
        field_obj = annotations.get(field_key, {})
        if field_obj.get("status") != "annotated" or not field_obj.get("raw_text"):
            continue

        target_text = str(field_obj["raw_text"]).strip()
        target_words = target_text.split()
        max_win = min(len(tokens), max(1, len(target_words) + 3))

        best_score = 0.0
        best_span: Optional[tuple[int, int]] = None

        for win_size in range(1, max_win + 1):
            for start_idx in range(len(tokens) - win_size + 1):
                candidate_tokens = tokens[start_idx:start_idx + win_size]
                if any(t["bio_tag"] != "O" for t in candidate_tokens):
                    continue
                candidate_text = " ".join(t["text"] for t in candidate_tokens)
                score = similarity(target_text, candidate_text)
                if score > best_score:
                    best_score = score
                    best_span = (start_idx, start_idx + win_size)

        if best_span and best_score >= threshold:
            start, end = best_span
            matched_boxes = []
            for idx in range(start, end):
                prefix = "B" if idx == start else "I"
                tokens[idx]["bio_tag"] = f"{prefix}-{bio_label}"
                matched_boxes.append(tokens[idx]["bbox"])

            # Backfill missing bbox in ground truth
            if field_obj.get("bbox") is None:
                field_obj["bbox"] = merge_boxes(matched_boxes)

    return tokens


def process_ocr_and_bio(input_dir: Path, output_dir: Path, lang: str = "vi") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    files = sorted(input_dir.glob("*.json"))
    for filepath in files:
        with filepath.open("r", encoding="utf-8") as f:
            record = json.load(f)

        img_path = Path(record["document"]["images"]["original"]["path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for OCR: {img_path}")

        ocr_res = ocr_engine.ocr(str(img_path), cls=True)
        tokens: List[Dict[str, Any]] = []

        if ocr_res and ocr_res[0]:
            for idx, line in enumerate(ocr_res[0]):
                poly = line[0]
                text_val = str(line[1][0]).strip()
                conf = float(line[1][1])
                xs = [int(pt[0]) for pt in poly]
                ys = [int(pt[1]) for pt in poly]
                tokens.append({
                    "token_id": idx,
                    "text": text_val,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": round(conf, 4)
                })

        # Sort tokens top-to-bottom, left-to-right
        tokens.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
        tokens = align_and_tag_bio(tokens, record["annotations"])

        record["ocr"] = {
            "engine": "PaddleOCR_PP-OCRv3",
            "engine_version": "2.6.0",
            "tokens": tokens
        }

        out_path = output_dir / filepath.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4, allow_nan=False)

    logger.info("Completed Step 6: OCR & BIO alignment for %d files.", len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Run OCR, Fuzzy Alignment, and BIO Tagging.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/interim/mc_ocr_2021/mc_ocr_detailed"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/interim/mc_ocr_2021/bio_annotations"))
    parser.add_argument("--lang", type=str, default="vi")
    args = parser.parse_args()

    process_ocr_and_bio(args.input_dir, args.output_dir, args.lang)