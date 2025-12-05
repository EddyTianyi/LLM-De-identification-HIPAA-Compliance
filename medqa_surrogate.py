import json
import random

INPUT_JSONL = "test_redacted.jsonl"   
OUTPUT_JSONL = "test_surrogate.jsonl" 

def sample_age() -> int:
    return random.randint(18, 80)

def replace_age_placeholders(text: str) -> str:
    if "[AGE]" not in text:
        return text
    age = sample_age()
    replacement = f"{age}-year-old"
    return text.replace("[AGE]", replacement)

def main():
    random.seed(42)  

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            q = obj.get("question", "")
            if isinstance(q, str):
                obj["question"] = replace_age_placeholders(q)  

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done! Saved to:", OUTPUT_JSONL)

if __name__ == "__main__":
    main()
