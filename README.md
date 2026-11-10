# Receipt Restoration & Multimodal Information Extraction

**Research title:** Document Image Restoration and Multimodal Learning for Information Extraction from In-the-Wild Vietnamese and English Receipts  
**Vietnamese title:** Nghiên cứu kết hợp phục hồi ảnh tài liệu và mô hình học đa phương thức nhằm tự động hóa bóc tách thông tin chứng từ, hóa đơn tiếng Việt và tiếng Anh chụp thực tế.  
**Authors:** Hoang Si Nguyen & Cao Thi Thuy An  
**Program:** Second bachelor's degree, University of Information Technology (UIT)

> **Project status — 2026-11-10:** MC-OCR 2021 raw-data extraction and automated normalization (**Steps 1–4**) are implemented. **Step 5 (Manual Review / Ground Truth Correction) is in progress.** The English SROIE dataset adapter and review pipeline have **not yet been implemented or verified**. The datasets are **not yet finalized for model training**.

## 1. Project objectives and datasets

The target research workflow has two model-oriented directions:

- **Document image restoration:** OpenCV baseline, UVDoc and NAFNet (planned).
- **Information extraction:** OCR-dependent PP-OCRv3 / VietOCR + LiLT, compared with an OCR-free vision-language approach such as Qwen2-VL using QLoRA (planned; model configuration remains subject to experimental validation).

| Dataset | Language | Intended role | Status |
|---|---|---|---|
| **MC-OCR 2021** | Vietnamese | Main Vietnamese benchmark source | Steps 1–4 implemented; Step 5 ongoing |
| **SROIE** | English | Main English benchmark source | Adapter, normalization and QA to be implemented by another team member |
| **CORD** | Indonesian | Auxiliary dataset, evaluated separately | Not merged into the Vietnamese–English benchmark; no completed adapter verified |

Avoid interpreting the presence of a directory or a README plan as proof that a model, adapter or experiment has been executed.

## 2. Repository layout

The following shows the scripts verified in the current GitHub repository and the **local / generated directories** used by the MC-OCR pipeline. Since `data/` and `reports/` are gitignored, they may not appear in GitHub.

```text
receipt-restoration-multimodal/
├── README.md
├── requirements.txt
├── .gitignore
├── schemas/
│   └── receipt.schema.json
├── scripts/
│   ├── 01_extract_mc_ocr_raw.py
│   ├── 02_normalize_mc_ocr.py
│   ├── manual_bbox_selector.py
│   └── prepare_mcocr_clean_samples.py
├── data/                                     # Local only; ignored by Git
│   ├── raw/
│   │   ├── mc_ocr_2021/
│   │   │   └── train/
│   │   │       ├── images/
│   │   │       └── <your_annotations>.csv
│   │   └── sroie/                            # Proposed location
│   ├── interim/
│   │   ├── mc_ocr_2021/
│   │   │   ├── raw_json/
│   │   │   ├── parsed_annotations/
│   │   │   ├── aligned_annotations/
│   │   │   └── normalized_annotations/
│   │   └── sroie/                            # To be implemented
│   └── annotations/
│       ├── mc_ocr_2021/
│       │   ├── corrections/
│       │   │   ├── pending/
│       │   │   ├── clean_candidates/
│       │   │   └── excluded/
│       │   ├── reviewed/                     # Single final reviewed folder
│       │   ├── review_queue.jsonl           # If generated locally
│       │   └── review_manifest.jsonl        # If generated locally
│       └── sroie/                            # Proposed parallel structure
└── reports/                                  # Local only; ignored by Git
    └── data_quality/
        └── mc_ocr_2021/
            ├── extraction_summary.json
            ├── extraction_issues.jsonl
            ├── summary.json
            ├── sample_issues.jsonl
            ├── excluded_samples.json
            └── <sample_id>.json              # Original-segment audit sidecar
```

**Directory rule:** Keep `interim/` output reproducible. Only `data/annotations/mc_ocr_2021/reviewed/` is intended to hold human-reviewed receipt annotations. `corrections/clean_candidates/` contains automatically clean *candidates*, **not verified Ground Truth**.

## 3. Installation

Run commands **from the repository root**. The current local development environment uses **Python 3.10**.

### Windows PowerShell

```powershell
py -3.10 -m venv receipt_env
.\receipt_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, cv2, PIL, jsonschema; print('Dependencies OK')"
```

### Linux / macOS

```bash
python3.10 -m venv receipt_env
source receipt_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, cv2, PIL, jsonschema; print('Dependencies OK')"
```

**Dependency note:** The current `requirements.txt` includes `numpy==1.26.4`, `opencv-python==4.10.0.84`, `scikit-image`, `augraphy`, `datasets[vision]` and `jsonschema`. `02_normalize_mc_ocr.py` imports Pillow (`PIL`), which may arrive as a transitive dependency. If `PIL` is missing, run `python -m pip install Pillow`; add and pin Pillow explicitly in `requirements.txt` after testing the environment. Installing these dependencies is **not** equivalent to setting up GPU/CUDA or installing the full future training stack.

## 4. Current scripts and responsibilities

| Script | Stage | What it does | Main input | Main output |
|---|---|---|---|---|
| `scripts/01_extract_mc_ocr_raw.py` | Step 1 | Reads MC-OCR annotation CSV; exports one raw JSON per image; preserves annotation strings; logs malformed/missing fields | MC-OCR CSV | `interim/mc_ocr_2021/raw_json/`; extraction reports |
| `scripts/02_normalize_mc_ocr.py` | Steps 2–4 | Parses `anno_polygons`, `anno_texts`, `anno_labels`; aligns segments and bboxes; maps labels; validates Master Schema; creates audit/QA | Raw JSON, original images, `receipt.schema.json` | `parsed_annotations/`, `aligned_annotations/`, `normalized_annotations/`, QA reports |
| `scripts/manual_bbox_selector.py` | Step 5 helper | Opens a **hard-coded** image in OpenCV, allows ROI selection, converts resized-display coordinates to original-image XYXY bbox | Original image chosen in script | Prints bbox; **does not modify JSON** |
| `scripts/prepare_mcocr_clean_samples.py` | Step 5 helper | Copies eligible normalized JSON into `corrections/clean_candidates/`; excludes IDs in review directories and JSONL queue/manifest | Step 4 normalized JSON, local review metadata | Candidate JSON copies; **dry-run unless `--execute`** |

**Not verified in current repository:** `scripts/finalize_mcocr_review.py` (previously discussed as a possible new helper). Do not assume automatic finalization, correction history or manifest synchronization is already implemented.

### Step 1 — Export MC-OCR CSV to raw JSON

Input CSV must have these columns: `img_id`, `anno_polygons`, `anno_texts`, `anno_labels`, `anno_num`, `anno_image_quality`.

```powershell
python scripts/01_extract_mc_ocr_raw.py `
  --csv_path "data/raw/mc_ocr_2021/train/<your_annotations>.csv"
```

Replace the placeholder with the **actual** CSV filename. Optional arguments: `--output_dir`, `--report_dir`, `--no_overwrite`.

Step 1 **does not parse** polygons or normalize text labels. It preserves nonempty source annotation strings and converts empty annotation cells to `null`.

### Steps 2–4 — Parse, align and normalize MC-OCR

```powershell
python scripts/02_normalize_mc_ocr.py `
  --image-dir "data/raw/mc_ocr_2021/train/images" `
  --source-split train `
  --date-order DMY
```

The above assumes source images are in the indicated directory and the CSV was exported using Step 1 defaults. Supported optional arguments include `--raw-json-dir`, `--parsed-dir`, `--aligned-dir`, `--normalized-dir`, `--audit-dir`, `--schema-path`, `--project-root`, `--source-split`, `--date-order` and `--category-map`. Use `--category-map` only with a **verified** MC-OCR `category_id` → source label mapping. Use `train` only if this is the actual source split.

Default inputs/outputs:

```text
Input:
  data/interim/mc_ocr_2021/raw_json/*.json
  data/raw/mc_ocr_2021/train/images/*
  schemas/receipt.schema.json
Outputs:
  data/interim/mc_ocr_2021/parsed_annotations/*.json
  data/interim/mc_ocr_2021/aligned_annotations/*.json
  data/interim/mc_ocr_2021/normalized_annotations/*.json
  reports/data_quality/mc_ocr_2021/summary.json
  reports/data_quality/mc_ocr_2021/sample_issues.jsonl
  reports/data_quality/mc_ocr_2021/excluded_samples.json
  reports/data_quality/mc_ocr_2021/<sample_id>.json
```

`excluded_samples.json` from Step 4 records pre-review exclusion candidates for supervised IE (including warnings), **not confirmed non-receipts**. `annotation_meta.annotation_status` is initially `draft`.

### Step 5 helper — Copy clean candidates

```powershell
# Preview only: no file operations
python scripts/prepare_mcocr_clean_samples.py

# Copy eligible files (without changing Step 4 source)
python scripts/prepare_mcocr_clean_samples.py --execute
```

**Known limitation in committed source:** `prepare_mcocr_clean_samples.py` blocks **all** sample IDs found in `review_queue.jsonl`, regardless of individual review status. This is correct only if the queue contains exclusively samples needing review. If the queue contains the entire dataset, the resulting candidate count may be zero. Inspect queue membership and preview counts first; the script does not independently check the `needs_review` flag in the Step 4 audit.

### Step 5 helper — Manually select a bounding box

```powershell
python scripts/manual_bbox_selector.py
```

Currently edit the script's `image_path` to select the required source image. Drag a rectangle and confirm with Enter/Space. The printed coordinates use `[xmin, ymin, xmax, ymax]` in **original image pixels**, not `[x, y, width, height]`. Copy the result into a Step 5 working JSON after checking it against the original image. This helper **does not save** the JSON or manage review statuses.

## 5. Canonical Master Schema

`schemas/receipt.schema.json` is JSON Schema **Draft 2020-12**, schema version `1.0.0`. Ten **conceptual** labels are represented using seven top-level annotation keys:

- Header: `store_name`, `store_address`, `transaction_datetime`, `receipt_number`.
- Repeated line-item labels inside `items.rows`: `item_name`, `quantity`, `unit_price`, `line_total`.
- Summary: `discount_or_tax`, `total_amount`.

MC-OCR source labels currently mapped by `02_normalize_mc_ocr.py`:

| MC-OCR source label | Master Schema field |
|---|---|
| `SELLER` | `store_name` |
| `ADDRESS` | `store_address` |
| `TIMESTAMP` | `transaction_datetime` |
| `TOTAL_COST` | `total_amount` |

Fields **not annotated by the MC-OCR source** (e.g. `receipt_number`, `items`, `discount_or_tax` in the current mapping) default to `not_evaluable`. They must not be interpreted as `not_present` simply because source labels are missing.

Field status values currently allowed by schema: `annotated`, `not_present`, `not_evaluable`, `pending`. Record review status in `annotation_meta.annotation_status`: `draft`, `reviewed`, `final`.

| Field status | Interpretation |
|---|---|
| `annotated` | A value is available; automatically generated values may still require manual correction |
| `not_present` | A reviewer verified the field is truly absent from a **valid receipt** |
| `not_evaluable` | Insufficient evidence, ambiguity, unreadable value or absent source supervision |
| `pending` | Field awaiting review; do not treat as final supervision |

For non-annotated scalar fields, `value`, `raw_text` and `bbox` must be `null` as specified by the schema. `items.rows` has distinct null/empty-array rules. `transaction_datetime.value` should use ISO-style date/time; `raw_text` should preserve source-visible wording. A monetary `value` is numeric. Bboxes are **XYXY** on the referenced original image.

## 6. End-to-end dataset pipeline: raw JSON → train-ready

This is the **target workflow**, clearly separating implementation from plans. The MC-OCR source begins with CSV; SROIE may use a different original label format but should emit an equivalent dataset-specific raw JSON layer.

| Stage | Operation | Output | Implementation status |
|---|---|---|---|
| **1. Raw extraction** | Convert original source annotations to one raw JSON per image without losing provenance | `interim/<dataset>/raw_json/` | **MC-OCR implemented; SROIE not implemented** |
| **2. Annotation parsing** | Parse source label structure, text, polygon/bbox fields | `parsed_annotations/` | **MC-OCR implemented** |
| **3. Alignment + QA** | Align source text/labels/geometry and detect warnings | `aligned_annotations/`, QA issues | **MC-OCR implemented** |
| **4. Schema normalization** | Map dataset-specific labels to Master Schema; preserve original evidence | `normalized_annotations/`, audit sidecars | **MC-OCR implemented** |
| **5A. Manual review** | Check original images, correct text/value/bbox, identify invalid documents | `reviewed/`, `excluded/`, review manifest | **MC-OCR ongoing; workflow partially manual** |
| **5B. Annotation expansion** | Add missing receipt fields and line-item rows where annotation is feasible | More complete reviewed labels | **Not completed** |
| **6. Final validation + inventory** | Validate JSON Schema, image bounds, exclusions, unique IDs and review decisions | Accepted/versioned sample manifest | **Planned** |
| **7. OCR + layout alignment** | Run selected OCR; align tokens/bboxes to IE labels; separate OCR predictions from source GT | OCR inputs for LiLT and evaluation | **Planned; GT-vs-predicted OCR policy not locked** |
| **8. Split/freeze benchmark** | Prevent duplicate/merchant/template leakage; create train/validation/test and label masks | Reproducible split manifests | **Planned** |
| **9. Model-specific preprocessing** | Create restoration clean/degraded pairs and data formats for LiLT/VLM | Train-ready datasets | **Planned** |
| **10. Training/evaluation** | Train models, compare restoration and extraction performance | Metrics, ablations, bilingual evaluation | **Planned** |

**Data acceptance policy:** exclude confirmed non-receipt images from supervised Receipt IE; do not treat unknown fields as negative labels; apply target masks for `not_evaluable`/`pending`. A receipt may be usable for some fields but not others. Report partial/full 10-label coverage, item-row counts, source languages, excluded samples, difficulty strata when assigned, and reviewed-versus-auto-clean provenance. CORD remains a separate auxiliary dataset, not an English benchmark replacement.

### Step 5: human review decisions

```text
Step 4: normalized_annotations/  (reproducible, not manually edited)
       |
       +-- warning samples ---> corrections/pending/ ----------+
       |                                                       |
       +-- auto-clean -------> corrections/clean_candidates/ --+--> Human QA
                                                                  |
                                            +---------------------+------------------+
                                            |                                        |
                                       Valid receipt                           Not a receipt
                                            |                                        |
                               annotations/mc_ocr_2021/reviewed/       corrections/excluded/
```

For a corrected field, update `value`, `raw_text` and `bbox` consistently; preserve the **before/after correction** and a review decision separately. An automatically `annotated` field may be wrong. If date and time appear in multiple boxes, retain per-segment boxes/indices in audit and correction records even if the canonical field stores a merged bbox. If the image is not a receipt, mark sample-level `document_validity=invalid_document` in the review metadata and exclude it; **do not** mark all fields `not_present`.

**Important:** At the time of README preparation, the committed scripts do **not** demonstrate a complete automated finalization process or automated review-manifest synchronization. The local `review_manifest.jsonl` / `review_queue.jsonl` structure should be checked before creating automation. Never claim a sample was manually reviewed just because it had no automatic warnings.

## 7. Teammate handoff: English SROIE

The English dataset work should use the **same Master Schema**, reporting and review policies, but **not reuse MC-OCR-specific source assumptions** (CSV columns, `|||` delimiter, polygon layout, `TIMESTAMP` label names or the Vietnamese `DMY` policy).

**Suggested work package:**

1. Inventory SROIE release/version, license, image count, annotation format, available labels and original splits.
2. Keep the original SROIE data unchanged under `data/raw/sroie/`. Build a **new source adapter** that exports one raw JSON per image to `data/interim/sroie/raw_json/` with source identifiers and provenance.
3. Implement SROIE-specific parse/alignment logic from **actual source annotation structure**, and map only supported source labels to `schemas/receipt.schema.json`.
4. Save dataset-specific parsed/aligned/normalized JSON and audit/QA issues under `interim/sroie/` and `reports/data_quality/sroie/`.
5. Use `not_evaluable` for unannotated schema fields (notably unavailable line items); never fabricate absent labels or infer `not_present` from the source's limited supervision.
6. Carry out the same image-grounded Manual Review: correct values/bboxes; record before/after; classify valid/invalid documents; keep reviewed and excluded outputs separate.
7. Validate reviewed JSON against the **shared schema**, produce per-field coverage and issue statistics, and verify that scripts are re-runnable without overwriting manual corrections.
8. Submit implementation, usage commands, tests and a small sanitized example through a pull request. **Do not commit raw data, full derived annotations or sensitive receipts**.

**Suggested responsibility split:** MC-OCR Vietnamese processing and review — current owner; SROIE English adapter and review — second team member; shared schema, dataset release policy, train/validation/test integration and model evaluation — joint review.

## 8. Progress checklist

- [x] MC-OCR CSV → one raw JSON per sample, with extraction reports.
- [x] MC-OCR text/label/polygon parsing and index alignment.
- [x] MC-OCR normalization to Master Schema 1.0.0.
- [x] Automated geometry, datetime, amount and text-quality warnings with audit reports.
- [x] Manual bbox selector helper and clean-candidate copy helper.
- [ ] **Finish MC-OCR Step 5 Manual Review and Ground Truth Correction.**
- [ ] Implement/validate sample-level review manifest and safe finalization workflow.
- [ ] Expand missing MC-OCR target annotations, including line items where needed and feasible.
- [ ] Implement SROIE adapter, quality reports and Manual Review.
- [ ] Validate/freeze bilingual dataset inventory and train/val/test splits.
- [ ] Implement OCR input generation/alignment and model-specific datasets.
- [ ] Implement image restoration, training, evaluation and reporting.

## 9. Reproducibility and security

- Preserve `data/raw/` and reproducible Step 1–4 outputs; do not run normalization over manually reviewed JSON.
- `data/` and `reports/` are excluded by `.gitignore`. Ensure files are not already Git-tracked before committing.
- Record exact dataset release, command arguments, review decisions, exclusion reasons and splits.
- Use `python scripts/<script_name>.py --help` to inspect actual supported CLI arguments.
- **This README was authored from the code committed as of 2026-11-10; commands have not been executed against the local datasets as part of this documentation update.**
