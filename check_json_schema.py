import json
from pathlib import Path
from jsonschema import Draft202012Validator

schema_path = Path("schemas/receipt.schema.json")
normalized_dir = Path(
    "data/interim/mc_ocr_2021/normalized_annotations"
)

schema = json.loads(schema_path.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)

files = sorted(normalized_dir.glob("*.json"))
invalid = []

for file in files:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        validator.validate(data)
    except Exception as exc:
        invalid.append((file.name, str(exc)))

print("Total:", len(files))
print("Valid:", len(files) - len(invalid))
print("Invalid:", len(invalid))

for filename, error in invalid[:10]:
    print(filename, error)