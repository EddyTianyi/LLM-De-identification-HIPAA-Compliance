# gpt_fill_placeholders_minimal_llamacpp.py
import os, json, time, argparse, re
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from llama_cpp import Llama, LlamaGrammar  # pip install llama-cpp-python

SYSTEM_RULES = """You generate realistic but fictitious replacement strings for placeholders in a medical report.
Rules:
- Output ONLY valid JSON (no extra text).
- JSON shape must be: {"replacements": [{"ph":"<placeholder>", "value":"<string>"} , ...]}
- Use each placeholder from the sidecar EXACTLY once; do not invent new ones.
- Values must be realistic for the given semantic type (NAME, DATE, PHONE, EMAIL, ADDRESS, ID/MRN/SSN, URL/IP, etc.)
- Do not copy any sensitive real data: fabricate but keep formats plausible.
"""

USER_TEMPLATE = """MASKED REPORT:
---
{report}
---

PLACEHOLDER SIDECAR (JSON):
---
{sidecar}
---

Return ONLY the JSON object described in the rules.
"""

def build_grammar_from_sidecar(sidecar_json: str) -> str:
    """
    构造 llama.cpp Grammar，使输出严格为：
    {"replacements":[{"ph":"<PH>", "value":"<string>"} , ...]}
    其中 ph 只能取 sidecar 里的占位符枚举；value 为至少 1 个字符的 JSON 字符串。
    """
    try:
        sc = json.loads(sidecar_json)
        phs = [p.get("ph") for p in sc.get("placeholders", []) if isinstance(p, dict) and p.get("ph")]
    except Exception:
        phs = []

    # 基础 JSON 语法部件
    base = r"""
ws         ::= ([ \t\n\r])*
char       ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
string1    ::= "\"" (char)+ "\"" ws
string     ::= "\"" (char)* "\"" ws
lbrace     ::= "{" ws
rbrace     ::= "}" ws
lbrack     ::= "[" ws
rbrack     ::= "]" ws
colon      ::= ":" ws
comma      ::= "," ws
"""
    # ph 的枚举（必须精确匹配 sidecar 中的占位符；如果为空则允许任意字符串）
    if phs:
        ph_enum = " | ".join(json.dumps(p) for p in phs)  # ← 关键修复
        ph_rule = f'ph ::= {ph_enum} ws\n'

        items_rules = []
        items_seq = []
        for i, p in enumerate(phs):
            items_rules.append(
                f'item{i} ::= lbrace "\\\"ph\\\"" ws colon {json.dumps(p)} ws comma "\\\"value\\\"" ws colon string1 rbrace'
            )

            items_seq.append(f'item{i}')
        items_rule = (
            "\n".join(items_rules)
            + "\nitems ::= "
            + (items_seq[0] if len(items_seq) == 1 else " (comma ".join(items_seq) + ")" * (len(items_seq) - 1))
            + " \n"
        )
    else:
        ph_rule = 'ph ::= string ws\n'
        items_rule = 'items ::= /* empty (no placeholders) */\n'

    root = r"""
root ::= obj
obj  ::= lbrace "\"replacements\"" ws colon lbrack (items)? rbrack rbrace
"""
    # 通用 item（当 phs 为空时用）
    generic_item = r"""
item ::= lbrace "\"ph\"" ws colon string ws comma "\"value\"" ws colon string1 rbrace
"""

    if phs:
        # 固定顺序版本（严格一一对应，不新增不缺失）
        grammar = base + ph_rule + items_rule + root
    else:
        grammar = base + ph_rule + generic_item + root

    return grammar

def load_rows(input_path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load records from CSV or JSONL into a list of dicts.
    """
    rows: List[Dict[str, Any]] = []
    if input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
        records = df.to_dict(orient="records")
        if isinstance(max_rows, int) and max_rows > 0:
            records = records[:max_rows]
        rows.extend(records)
    elif input_path.lower().endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if isinstance(max_rows, int) and max_rows > 0 and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    continue
    else:
        raise ValueError("Unsupported input format, only .csv or .jsonl are allowed.")
    return rows


def save_output_rows(out_rows: List[Dict[str, Any]], output_path: Optional[str]) -> pd.DataFrame:
    """
    Save out_rows to CSV or JSONL depending on suffix.
    Always returns a DataFrame for compatibility.
    """
    df_out = pd.DataFrame(out_rows)
    if not output_path:
        return df_out

    if output_path.lower().endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df_out.iterrows():
                # convert each row to dict and dump as JSONL
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    else:
        # default to CSV
        df_out.to_csv(output_path, index=False)
    return df_out


def call_llamacpp_for_replacements(
    llm: Llama,
    masked_report: str,
    sidecar_json: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_output_tokens: int = 800
) -> Dict[str, str]:
    try:
        sc = json.loads(sidecar_json)
        phs = [p.get("ph") for p in sc.get("placeholders", []) if isinstance(p, dict) and p.get("ph")]
    except Exception:
        phs = []
    if not phs:
        return {}

    user_text = USER_TEMPLATE.format(report=masked_report, sidecar=sidecar_json)
    prompt = SYSTEM_RULES + "\n\n" + user_text

    grammar_text = build_grammar_from_sidecar(sidecar_json)
    grammar_obj = LlamaGrammar.from_string(grammar_text)

    out = llm(
        prompt=prompt,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=None,
        grammar=None,  #这里改成了不用grammar
        echo=False,
    )
    raw = out["choices"][0]["text"].strip() if out and out.get("choices") else ""
    if not raw:
        return {}

    # 尝试解析 JSON
    try:
        data = json.loads(raw)
    except Exception:
        # 容错：剪去前后非 JSON 噪声
        l, r = raw.find("{"), raw.rfind("}")
        try:
            data = json.loads(raw[l:r+1]) if (l != -1 and r != -1 and r > l) else {}
        except Exception:
            return {}

    if not isinstance(data, dict) or "replacements" not in data or not isinstance(data["replacements"], list):
        return {}

    allowed = set(phs)
    mapping: Dict[str, str] = {}
    for item in data["replacements"]:
        if not isinstance(item, dict):
            continue
        ph = item.get("ph")
        val = item.get("value")
        if isinstance(ph, str) and isinstance(val, str) and val.strip() and (ph in allowed):
            mapping[ph] = val.strip()
    return mapping

def parse_replacements_json(s: str) -> dict:
    try:
        obj = json.loads(s) if isinstance(s, str) else (s or {})
    except Exception:
        return {}
    reps = obj.get("replacements", [])
    mapping = {}
    for it in reps:
        ph = it.get("ph")
        val = it.get("value")
        if isinstance(ph, str) and isinstance(val, str) and val.strip():
            mapping[ph] = val.strip()
    return mapping

def apply_replacements(masked_report: str, mapping: dict) -> str:
    text = masked_report or ""
    for ph in sorted(mapping.keys(), key=len, reverse=True):
        val = mapping[ph]
        if not val:
            continue
        pattern = re.escape(ph)
        text = re.sub(pattern, val, text)
    return text



def extract_placeholders_from_text(text: str) -> List[str]:
    """
    从文本中抽取占位符，比如 [NAME_1], [HOSPITAL], [AGE] 等。
    这里假设占位符格式是全大写 + 下划线/数字，包在方括号里。
    你可以根据自己实际占位符风格调整正则。
    """
    if not text:
        return []
    pattern = r"\[[A-Z][A-Z0-9_]*\]"
    phs = re.findall(pattern, text)
    # 去重并排序，保证顺序稳定
    return sorted(set(phs))

def test_file_with_llamacpp(
    model_file: str,
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_output_tokens: int = 800,
    ctx_size: int = 4096,
    n_gpu_layers: int = 33,  # M2 建议尽量多放 GPU
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, int]:
    """
    只测试“能否按 sidecar 正确产出 JSON 映射”。不做文本替换之外的操作。
    输出列 id, masked_or_report, placeholder_sidecar, replacements_json, filled_report
    """
    # 1) 加载模型（一次）
    llm = Llama(
        model_path=model_file,
        n_ctx=ctx_size,
        n_gpu_layers=max(0, int(n_gpu_layers)),
        logits_all=False,
        verbose=False,
    )

    # 2) 读取数据
    df = pd.read_csv(input_csv_path)
    text_col = "masked_report_placeholders" if "masked_report_placeholders" in df.columns else "report"
    if "placeholder_sidecar" not in df.columns or text_col not in df.columns:
        raise ValueError("需要列 'placeholder_sidecar' 以及 'masked_report_placeholders' 或 'report'。")

    out_rows: List[Dict[str, Any]] = []
    err = 0
    t0 = time.time()

    it = df.iterrows()
    if isinstance(max_rows, int) and max_rows > 0:
        it = list(df.iterrows())[:max_rows]

    for i, row in it:
        rid = row["id"] if "id" in df.columns else f"row_{i}"
        masked = str(row[text_col]) if pd.notna(row[text_col]) else ""
        sidecar = str(row["placeholder_sidecar"]) if pd.notna(row["placeholder_sidecar"]) else ""

        # sidecar 校验
        try:
            sc = json.loads(sidecar)
            phs = sc.get("placeholders", [])
            if not isinstance(phs, list):
                raise ValueError("sidecar.placeholders not list")
        except Exception:
            err += 1
            out_rows.append({
                "id": rid,
                text_col: masked,
                "placeholder_sidecar": sidecar,
                "replacements_json": json.dumps({"replacements": []}, ensure_ascii=False),
                "filled_report": masked,
            })
            continue

        mapping = call_llamacpp_for_replacements(
            llm=llm,
            masked_report=masked,
            sidecar_json=sidecar,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens
        )

        if not mapping:
            err += 1
            rep_json = {"replacements": []}
            filled = masked
        else:
            # 保持 sidecar 顺序
            order = [p.get("ph") for p in json.loads(sidecar).get("placeholders", []) if isinstance(p, dict)]
            rep_json = {"replacements": [{"ph": ph, "value": mapping.get(ph, "")} for ph in order]}
            filled = apply_replacements(masked, mapping)

        out_rows.append({
            "id": rid,
            text_col: masked,
            "placeholder_sidecar": sidecar,
            "replacements_json": json.dumps(rep_json, ensure_ascii=False),
            "filled_report": filled,
        })

    df_out = pd.DataFrame(out_rows)
    if output_csv_path:
        df_out.to_csv(output_csv_path, index=False)
    print(f"Done {len(df_out)} rows in {time.time()-t0:.1f}s, errors={err}")
    return df_out, err

def relex_jsonl_questions_with_llamacpp(
    model_file: str,
    input_jsonl_path: str,
    output_jsonl_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_output_tokens: int = 800,
    ctx_size: int = 4096,
    n_gpu_layers: int = 33,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    版本：只针对 MedQA 风格 JSONL。
    对每一行：
      - 从 question 中自动抽取占位符（如 [NAME_1], [AGE]）
      - 临时构造 placeholder_sidecar
      - 调用 call_llamacpp_for_replacements 生成合成词
      - 用 apply_replacements 填回 question，生成 question_filled
      - 输出新的 JSONL（保留原字段 + placeholder_sidecar + replacements_json + question_filled）

    不再依赖 CSV，不再依赖 'masked_report_placeholders' / 'placeholder_sidecar' 原列。
    """

    # 1) 加载模型
    llm = Llama(
        model_path=model_file,
        n_ctx=ctx_size,
        n_gpu_layers=max(0, int(n_gpu_layers)),
        logits_all=False,
        verbose=False,
    )

    # 2) 读取 input_jsonl
    rows: List[Dict[str, Any]] = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if isinstance(max_rows, int) and max_rows > 0 and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue

    if not rows:
        raise ValueError("No valid JSON lines found in input file.")

    if "question" not in rows[0]:
        raise ValueError("Input JSONL must contain a 'question' field for each object.")

    out_rows: List[Dict[str, Any]] = []
    err = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        rid = row.get("id", f"row_{i}")
        q = str(row.get("question", "") or "")

        # 从 question 中抽取占位符
        phs = extract_placeholders_from_text(q)

        if not phs:
            # 没有占位符，就直接原样拷贝
            rep_json = {"replacements": []}
            filled = q
            row_out = dict(row)
            row_out.update({
                "id": rid,
                "placeholder_sidecar": json.dumps({"placeholders": []}, ensure_ascii=False),
                "replacements_json": json.dumps(rep_json, ensure_ascii=False),
                "question_filled": filled,
            })
            out_rows.append(row_out)
            continue

        # 构造 sidecar: {"placeholders":[{"ph":"[NAME_1]"}, ...]}
        sidecar_obj = {
            "placeholders": [{"ph": ph} for ph in phs]
        }
        sidecar = json.dumps(sidecar_obj, ensure_ascii=False)

        # 调用你已有的 Qwen+grammar 生成替换映射
        mapping = call_llamacpp_for_replacements(
            llm=llm,
            masked_report=q,
            sidecar_json=sidecar,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

        if not mapping:
            err += 1
            rep_json = {"replacements": []}
            filled = q
        else:
            # 保持 sidecar 中的顺序
            rep_json = {
                "replacements": [{"ph": ph, "value": mapping.get(ph, "")} for ph in phs]
            }
            filled = apply_replacements(q, mapping)

        row_out = dict(row)
        row_out.update({
            "id": rid,
            "placeholder_sidecar": sidecar,
            "replacements_json": json.dumps(rep_json, ensure_ascii=False),
            "question_filled": filled,
        })
        out_rows.append(row_out)

    df_out = pd.DataFrame(out_rows)

    # 写回 JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for _, r in df_out.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"Done {len(df_out)} rows in {time.time() - t0:.1f}s, errors={err}")
    return df_out, err





if __name__ == "__main__":
    # ==== 修改为你的 7B GGUF 路径 ====
    model_file = "/Users/edward/Downloads/LLMAnonymizer-Publication-main/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

    input_csv_path = "/Users/edward/Downloads/LLMAnonymizer-Publication-main/webapp/llm_processing/llm_place_holder_output.csv"
    output_csv_path = "/Users/edward/Downloads/LLMAnonymizer-Publication-main/webapp/llm_processing/test_replacements_mistral7b.csv"

    temperature = 0.2
    max_rows = 40
    ctx_size = 4096
    n_gpu_layers = 0  

    # test_file_with_llamacpp(
    #     model_file=model_file,
    #     input_csv_path=input_csv_path,
    #     output_csv_path=output_csv_path,
    #     temperature=temperature,
    #     max_output_tokens=2000,
    #     ctx_size=ctx_size,
    #     n_gpu_layers=n_gpu_layers,
    #     max_rows=(max_rows if max_rows and max_rows > 0 else None)
    # )
    df_out, err = relex_jsonl_questions_with_llamacpp(
    model_file=model_file,
    input_jsonl_path="/Users/edward/Downloads/LLMAnonymizer-Publication-main/test_redacted.jsonl",
    output_jsonl_path="test_relex.jsonl",
    max_rows=None,
)

