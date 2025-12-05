import json
import re

INPUT_JSONL = "/Users/edward/Downloads/data_clean/questions/US/test.jsonl"
OUTPUT_JSONL = "test_redacted.jsonl"

# Age patterns to redact
age_patterns = [
    r"\b\d{1,3}-year-old\b",
    r"\b\d{1,3} year old\b",
    r"\b\d{1,3}-year old\b",
    r"\b\d{1,3} yo\b",

    r"\b\d{1,3}-month-old\b",
    r"\b\d{1,3} month old\b",
    r"\b\d{1,3}-month old\b",

    r"\b\d{1,3}-week-old\b",
    r"\b\d{1,3} week old\b",
    r"\b\d{1,3}-week old\b"
]

def redact_age(text):
    for p in age_patterns:
        text = re.sub(p, "[AGE]", text, flags=re.IGNORECASE)
    return text

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        if "question" in item:
            item["question"] = redact_age(item["question"])
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Saved:", OUTPUT_JSONL)
