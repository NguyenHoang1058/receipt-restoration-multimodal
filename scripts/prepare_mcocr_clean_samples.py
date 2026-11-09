
from pathlib import Path
from collections import Counter
import argparse
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT / "data/interim/mc_ocr_2021/normalized_annotations"
)

ANNOTATION_DIR = (
    ROOT / "data/annotations/mc_ocr_2021"
)

CORRECTIONS_DIR = ANNOTATION_DIR / "corrections"

PENDING_DIR = CORRECTIONS_DIR / "pending"
CANDIDATES_DIR = CORRECTIONS_DIR / "clean_candidates"
EXCLUDED_DIR = CORRECTIONS_DIR / "excluded"

REVIEWED_DIR = ANNOTATION_DIR / "reviewed"

REVIEW_QUEUE = ANNOTATION_DIR / "review_queue.jsonl"
REVIEW_MANIFEST = ANNOTATION_DIR / "review_manifest.jsonl"

BLOCKED_REVIEW_STATUSES = {
    "pending",
    "in_progress",
    "corrected",
    "confirmed",
    "reviewed",
    "excluded",
    "verified_no_change",
    "unresolved",
}

BLOCKED_DOCUMENT_VALIDITIES = {
    "invalid_document",
    "uncertain_document_type",
}


def read_jsonl(path):
    """Read JSONL without modifying the source."""
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at {path}:{line_number}"
                )

            records.append(record)

    return records


def folder_sample_ids(folder):
    if not folder.exists():
        return set()

    return {
        path.stem
        for path in folder.glob("*.json")
    }


def get_blocked_samples():
    blocked = set()

    # Samples already assigned to a review workflow.
    for folder in (
        PENDING_DIR,
        CANDIDATES_DIR,
        EXCLUDED_DIR,
        REVIEWED_DIR,
    ):
        blocked.update(folder_sample_ids(folder))

    # Samples queued for manual review.
    for record in read_jsonl(REVIEW_QUEUE):
        sample_id = record.get("sample_id")
        if sample_id:
            blocked.add(str(sample_id))

    # Explicit states recorded during review.
    for record in read_jsonl(REVIEW_MANIFEST):
        sample_id = record.get("sample_id")
        if not sample_id:
            continue

        status = str(
            record.get("review_status", "")
        ).lower()

        validity = str(
            record.get("document_validity", "")
        ).lower()

        if (
            status in BLOCKED_REVIEW_STATUSES
            or validity in BLOCKED_DOCUMENT_VALIDITIES
        ):
            blocked.add(str(sample_id))

    return blocked


def prepare_clean_samples(execute=False):
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)

    blocked = get_blocked_samples()
    stats = Counter()

    source_files = sorted(SOURCE_DIR.glob("*.json"))

    for source in source_files:
        sample_id = source.stem

        if sample_id in blocked:
            stats["blocked_by_review"] += 1
            continue

        target = CANDIDATES_DIR / source.name

        if target.exists():
            stats["already_exists"] += 1
            continue

        stats["candidates"] += 1

        if execute:
            CANDIDATES_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(source, target)
            stats["copied"] += 1

    print(f"Mode: {'EXECUTE' if execute else 'DRY RUN'}")
    print(f"Normalized samples: {len(source_files)}")
    print(f"Blocked by review: {stats['blocked_by_review']}")
    print(f"Already in candidates: {stats['already_exists']}")
    print(f"New candidates: {stats['candidates']}")

    if execute:
        print(f"Copied: {stats['copied']}")
        print(f"Destination: {CANDIDATES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy eligible samples to clean_candidates",
    )

    args = parser.parse_args()
    prepare_clean_samples(execute=args.execute)
