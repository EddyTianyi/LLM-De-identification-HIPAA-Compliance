import json
import re
import time
from typing import Dict
from llama_cpp import Llama


MODEL_PATH = "/Users/edward/Downloads/LLMAnonymizer-Publication-main/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"  
DATA_FILE = "/Users/edward/Downloads/LLMAnonymizer-Publication-main/test_redacted.jsonl"

N_CTX = 4096
N_THREADS = 8       
N_GPU_LAYERS = 35    
MAX_TOKENS = 4       



def extract_answer_letter(text: str) -> str:

    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else None


def build_prompt(question: str, options: Dict[str, str]) -> str:
    prompt = (
        "You are a medical exam assistant.\n"
        "You MUST answer with only the letter of the correct option: A, B, C, D, or E.\n\n"
        f"Question:\n{question}\n\nOptions:\n"
    )
    for letter in sorted(options.keys()):
        prompt += f"{letter}. {options[letter]}\n"
    prompt += "\nAnswer with a single letter only."
    return prompt


def evaluate_with_qwen():
    print("Loading Qwen model...")
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=False,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")


    total = 0
    correct = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            question = item["question"]
            options = item["options"]
            gold = item["answer_idx"].strip().upper()

            prompt = build_prompt(question, options)


            out = llm(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                stop=["\n"], 
                echo=False,
            )
            output = out["choices"][0]["text"].strip()
            pred = extract_answer_letter(output)

            total += 1
            if pred == gold:
                correct += 1

            print(f"[{total}] Pred={pred}, Gold={gold}  |  raw='{output}'")

    accuracy = correct / total if total > 0 else 0.0
    print("\n===============================")
    print(f"Final Accuracy (Qwen): {accuracy:.4f}")
    print("===============================")

    return accuracy


if __name__ == "__main__":
    evaluate_with_qwen()
