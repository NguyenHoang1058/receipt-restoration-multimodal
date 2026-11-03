# receipt-restoration-multimodal
Repository for my second bachelor's degree at the University of Information Technology (UIT).

Below is detail description for this project

# Document Image Restoration and Multimodal Learning for Information Extraction from In-the-Wild Vietnamese and English Receipts

**Vietnamese Title:** Nghiên cứu kết hợp phục hồi ảnh tài liệu và mô hình học đa phương thức nhằm tự động hóa bóc tách thông tin chứng từ, hóa đơn tiếng Việt và tiếng Anh chụp thực tế.

**Authors:** Hoang Si Nguyen & Cao Thi Thuy An

## 📖 Project Overview
This project aims to automate key information extraction (KIE) from heavily degraded, mobile-captured receipts in both Vietnamese and English. It introduces a two-stage pipeline:
1. **Document Image Restoration:** Utilizing UVDoc and NAFNet (along with an OpenCV baseline) to unwarp geometric distortions and remove illumination artifacts (shadows).
2. **Multimodal Information Extraction:** Comparing two state-of-the-art approaches:
   - *OCR-dependent pipeline:* PP-OCRv3 / VietOCR combined with LiLT (Language-Independent Layout Transformer).
   - *OCR-free pipeline:* Qwen2-VL-2B-Instruct fine-tuned via QLoRA.

## 📂 Project Structure
The repository is strictly structured following MLOps standards to separate data engineering, model training, and evaluation workflows.

```text
receipt-restoration-multimodal/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/                   # YAML configuration files (dataset, splitting, augmentation)
├── schemas/
│   └── receipt.schema.json    # Master JSON Schema (Draft 2020-12) for all annotations
├── data/                      # Data directory (ignored in version control)
│   ├── raw/                   # Original MC-OCR, CORD, SROIE data
│   ├── processed/             # Unified JSON annotations & cleaned images
│   └── paired/                # Augmented paired data (clean vs. degraded) via Augraphy
├── src/                       # Core source code
│   ├── receipt_dataset/       # Data loaders and schema validation adapters
│   ├── restoration/           # OpenCV baseline, UVDoc, NAFNet inference modules
│   └── metrics/               # Evaluation metrics (PSNR, SSIM, CER, WER, F1, TEDS)
├── scripts/                   # Executable pipelines
│   ├── 01_prepare_raw_data.py
│   ├── 02_normalize_datasets.py
│   ├── 03_generate_annotations.py
│   └── 04_validate_annotations.py
└── notebooks/                 # EDA and quality analysis
```

## 📊 Dataset & Master Schema
We unify three different public datasets (MC-OCR 2021, CORD, SROIE) into a single canonical JSON structure. 
The **Master Label Set** consists of 10 unified fields:
- **Header:** `store_name`, `store_address`, `transaction_datetime`, `receipt_number`
- **Line Items:** `item_name`, `quantity`, `unit_price`, `line_total`
- **Summary:** `discount_or_tax`, `total_amount`

*Note: SROIE dataset lacks line-item annotations. To prevent negative label conflicts during training, missing fields in SROIE are flagged as `status: "not_evaluable"` in the JSON schema, which will dynamically map to `ignore_index = -100` in PyTorch's loss function.*

## 🚀 Current Progress (Phase 1: Weeks 1 - 3)
- [x] Defined the project structure and MLOps pipeline.
- [x] Drafted and finalized the `receipt.schema.json` (Draft 2020-12) supporting `bbox`, OCR tokens, and dynamic `status` flags.
- [x] Implemented raw data downloading and extraction scripts for SROIE (Kaggle/HF), CORD (HF .arrow), and MC-OCR (CSV).
- [ ] Implement OpenCV baseline for morphological transformations.
- [ ] Run VLM to generate pseudo-labels for line items and validate via Label Studio.
- [ ] Generate degraded paired images using the `Augraphy` library.

## ⚙️ Setup & Installation
```bash
# Create a virtual environment
python -m venv receipt_env
source receipt_env/bin/activate  # On Windows: receipt_env\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```