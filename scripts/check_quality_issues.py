import json
from pathlib import Path
from collections import Counter

path = Path(
    "reports/data_quality/mc_ocr_2021/sample_issues.jsonl"
)

counter = Counter()
affected = Counter()

with path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        issues = record.get("warnings", [])

        if not issues and record.get("issue_type"):
            issues = [record["issue_type"]]

        for issue in issues:
            counter[str(issue)] += 1

        for issue in set(map(str, issues)):
            affected[issue] += 1

print("=== Data Quality Issues ===")

for issue, count in counter.most_common():
    print(
        f"{issue}: {count} warnings, "
        f"{affected[issue]} samples"
    )