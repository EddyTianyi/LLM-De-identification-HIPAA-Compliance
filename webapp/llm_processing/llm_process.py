# llm_processing_backend.py
from __future__ import annotations
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple
import re
from typing import List, Tuple, Optional
from dateutil import parser as dtparser  
import os
import time
import pandas as pd
import re
import json
from utils import read_preprocessed_csv_from_zip, replace_personal_info, is_empty_string_nan_or_none

# pip install llama-cpp-python
from llama_cpp import Llama, LlamaGrammar

default_prompt = r"""You are a helpful medical assistant. Below you will find reports. Please extract the requested information verbatim from the report. Make sure you the content you ouput follows exactly the same format such as data and address as in the original report. If you do not find the information, respond with null. Please generate in the same format as in the text. 

This is the report:
{report}"""

# default_grammar = r"""
# root   ::= allrecords
# value  ::= object | array | string | number | ("true" | "false" | "null") ws

# allrecords ::= (
#   "{"
#   ws "\"patientLastName\":" ws string ","
#   ws "\"patientFirstName\":" ws string ","
#   ws "\"patientName\":" ws string ","
#   ws "\"patientHonorific\":" ws string ","
#   ws "\"patientBirthDate\":" ws string ","
#   ws "\"admissionDate\":" ws string ","
#   ws "\"dischargeDate\":" ws string ","
#   ws "\"deathDate\":" ws string ","
#   ws "\"otherDateElements\":" ws string ","
#   ws "\"patientAge\":" ws string ","
#   ws "\"patientphonenumber\":" ws string ","
#   ws "\"patientFaxNumber\":" ws string ","
#   ws "\"patientEmail\":" ws string ","
#   ws "\"patientSSN\":" ws string ","
#   ws "\"patientMRN\":" ws string ","
#   ws "\"healthPlanBeneficiaryNumber\":" ws string ","
#   ws "\"accountNumber\":" ws string ","
#   ws "\"certificateOrLicenseNumber\":" ws string "," 
#   ws "\"doctorName\":" ws string ","
#   ws "\"patientID\":" ws idlike ","
#   ws "\"patientStreet\":" ws string ","
#   ws "\"patientHouseNumber\":" ws string ","
#   ws "\"patientPostalCode\":" ws postalcode ","
#   ws "\"patientCity\":" ws string ","
#   ws "\"vehicleIdentifier\":" ws string ","
#   ws "\"vehicleVIN\":" ws string ","
#   ws "\"vehicleLicensePlate\":" ws string ","
#   ws "\"deviceIdentifier\":" ws string ","
#   ws "\"deviceSerialNumber\":" ws string ","
#   ws "\"url\":" ws string ","
#   ws "\"ipAddress\":" ws string ","
#   ws "\"otherUniqueIdOrCode\":" ws string ","
#   ws "}"
#   ws
# )

# record ::= (
#     "{"
#     ws "\"excerpt\":" ws ( string | "null" ) ","
#     ws "\"present\":" ws ("true" | "false") ws 
#     ws "}"
#     ws
# )

# object ::=
#   "{" ws (
#             string ":" ws value
#     ("," ws string ":" ws value)*
#   )? "}" ws

# array  ::=
#   "[" ws (
#             value
#     ("," ws value)*
#   )? "]" ws
# char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
# string ::=
#   "\"" (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char (char)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)?)? "\"" ws

# number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# postalcode ::= ("\"" [0-9][0-9][0-9][0-9][0-9] "\"" | "\"\"") ws
# idlike ::= ("\"" [0-9][0-9][0-9][0-9][0-9][0-9][0-9]?[0-9]? "\"" | "\"\"") ws

# # Optional space: by convention, applied in this grammar after literal chars when allowed
# ws ::= ([ \t\n])?
# """




default_grammar = r"""
root   ::= allrecords
value  ::= object | array | string | number | ("true" | "false" | "null") ws
ageval ::= "\"\"" ws | "\"" [0-9]{1,3} ( " "?( "years" | "year" | "yrs" | "yo" ) )? "\"" ws

# -------------------------
# All HIPAA identifiers as fields (strings or empty ""), plus existing fields
# -------------------------
allrecords ::= (
  "{"
  ws "\"patientFirstName\":" ws string ","
  ws "\"patientMiddleName\":" ws string ","
  ws "\"patientLastName\":" ws string ","
  ws "\"patientName\":" ws string ","
  ws "\"patientHonorific\":" ws string ","

  ws "\"patientBirthDate\":" ws string ","
  ws "\"admissionDate\":" ws string ","
  ws "\"dischargeDate\":" ws string ","
  ws "\"deathDate\":" ws string ","
  ws "\"otherDateElements\":" ws string ","
  ws "\"patientAge\":" ws ageval ","

  ws "\"patientPhoneNumber\":" ws string ","
  ws "\"patientFaxNumber\":" ws string ","
  ws "\"patientEmail\":" ws string ","

  ws "\"patientSSN\":" ws string ","
  ws "\"patientMRN\":" ws string ","
  ws "\"healthPlanBeneficiaryNumber\":" ws string ","
  ws "\"accountNumber\":" ws string ","
  ws "\"certificateOrLicenseNumber\":" ws string ","

  ws "\"patientAddress\":" ws string ","
  ws "\"patientStreet\":" ws string ","
  ws "\"patientHouseNumber\":" ws string ","
  ws "\"patientCity\":" ws string ","
  ws "\"patientCounty\":" ws string ","
  ws "\"patientPrecinct\":" ws string ","
  ws "\"patientPostalCode\":" ws postalcode ","
  ws "\"patientState\":" ws string ","
  ws "\"geocode\":" ws string ","

  ws "\"vehicleIdentifier\":" ws string ","
  ws "\"vehicleVIN\":" ws string ","
  ws "\"vehicleLicensePlate\":" ws string ","

  ws "\"deviceIdentifier\":" ws string ","
  ws "\"deviceSerialNumber\":" ws string ","

  ws "\"url\":" ws string ","
  ws "\"ipAddress\":" ws string ","

  ws "\"otherUniqueIdOrCode\":" ws string ","

  ws "\"patientID\":" ws idlike ","
  ws "\"doctorFullName\":" ws string
  ws "}"
)

# -------------------------
# JSON value forms
# -------------------------
record ::= (
    "{"
    ws "\"excerpt\":" ws ( string | "null" ) ","
    ws "\"present\":" ws ("true" | "false") ws
    ws "}"
    ws
)

object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws

# -------------------------
# Strings / numbers
# -------------------------
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" (char)* "\"" ws
number ::= "-"? ( [0-9] | [1-9] [0-9]* ) ("." [0-9]+)? ( [eE] [-+]? [0-9]+ )? ws

# -------------------------
# Patterns
# -------------------------
postalcode ::= ( "\"" [0-9]{5} ( "-" [0-9]{4} )? "\"" | "\"\"" ) ws
idlike     ::= ( "\"" [0-9]{7,8} "\"" | "\"\"" ) ws

# Optional space
ws     ::= ([ \t\n])?
"""



def _slice_outer_object(raw: str) -> str:
    l = raw.find("{")
    r = raw.rfind("}")
    return raw[l:r+1] if l != -1 and r != -1 and r > l else raw

def _try_json_loads_with_patch(raw: str):
    # First attempt
    try:
        return json.loads(_slice_outer_object(raw)), None
    except Exception as e1:
        # Patch the most common corruption: patientAge string with a raw newline or unclosed quotes
        patched = re.sub(
            r'"patientAge"\s*:\s*"(?:(?:\\.|[^"\\])*\n(?:.*?))?",',
            r'"patientAge": "",',
            _slice_outer_object(raw),
            flags=re.DOTALL
        )
        try:
            return json.loads(patched), "patched_patientAge"
        except Exception as e2:
            return None, f"json_error: {e2}"


# 只保留数字（可选保留前导 1）
def _phone_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

# 取“比对键”：优先取最后 10 位（北美号码），并返回 (digits10, digits_all)
def _phone_key(s: str):
    d = _phone_digits(s)
    last10 = d[-10:] if len(d) >= 10 else d
    return last10, d

# 在原文中找“像电话”的片段，返回 (raw_substring, key_last10)
_PHONE_SEP = r"[ \t\u00A0\u2009.\-]"          # 空格/不间断空格/窄空格/点/横
_AREA      = r"\(?\d{3}\)?"                   # (212) 或 212
_BLOCK3    = r"\d{3}"
_BLOCK4    = r"\d{4}"
_OPT_SEP   = rf"(?:{_PHONE_SEP})?"
_SEP       = rf"(?:{_PHONE_SEP})"

# 可匹配：+1 (212) 555-0211 / 212-555-0211 / 212 555 0211 / 212.555.0211 / (212)5550211 等
_PHONE_PATTERNS = [
    rf"(?:\+?1{_SEP})?{_AREA}{_OPT_SEP}{_BLOCK3}{_OPT_SEP}{_BLOCK4}(?:\s*(?:ext\.?|x)\s*\d{{1,6}})?",
]

def find_phone_spans_in_text(text: str):
    spans = []
    for pat in _PHONE_PATTERNS:
        for m in re.finditer(pat, text):
            raw = m.group(0)
            key10, _ = _phone_key(raw)
            if len(key10) >= 7:      # 简单阈值，避免匹配到短串
                spans.append((raw, key10))
    return spans

# 根据 LLM 给的“电话号码值”，回收原文中的等价片段（按最后10位对齐）
def collect_verbatim_phones_for_value(report_text: str, value: str):
    key10_val, _ = _phone_key(value)
    if not key10_val:
        return []
    outs = []
    for raw, key10 in find_phone_spans_in_text(report_text):
        if key10 == key10_val:
            outs.append(raw)
    return list(set(outs))



# from .utils import (
#     read_preprocessed_csv_from_zip,
#     replace_personal_info,
#     is_empty_string_nan_or_none,
# )

# -----------------------
# Progress helper (optional)
# -----------------------
def _format_time(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"

def default_progress_cb(done: int, total: int, eta: Optional[str] = None) -> None:
    if eta is None:
        print(f"[progress] {done}/{total}")
    else:
        print(f"[progress] {done}/{total}  ETA: {eta}")


def extract_from_report_inprocess(
    df: pd.DataFrame,
    model_file: str,
    symptoms: Iterable[str],
    n_predict: int,
    prompt: str = default_prompt,
    temperature: float = 0.1,
    grammar: str = default_grammar,
    ctx_size: int = 4096,
    n_gpu_layers: int = 0,
    progress_cb = default_progress_cb,
    job_id: Optional[str] = None,
    zip_file_path: Optional[str] = None,
    debug: bool = False,
    model_name_alias: str = "",
) -> Tuple[Tuple[pd.DataFrame, int], Optional[str]]:
    """
    Pure-backend version of the old server-based extract loop.
    Loads GGUF via llama-cpp-python, iterates df(report,id), applies symptoms-prompts,
    collects raw LLM JSON-like output, then calls postprocess_grammar(...) to produce final dataframe.
    Returns: ((aggregated_df, error_count), zip_file_path)
    """
    # 1) load model once
    llm = Llama(
        model_path=model_file,
        n_ctx=ctx_size,
        n_gpu_layers=max(0, int(n_gpu_layers)),
        logits_all=False,
        verbose=False,
    )

    # 2) iterate rows and query model
    total = len(df)
    start_t = time.time()

    results = {}
    skipped = 0

    for i, (report, _id) in enumerate(zip(df["report"], df["id"])):
        if is_empty_string_nan_or_none(report):
            skipped += 1
            done = (i + 1) - skipped
            eta = _format_time((time.time() - start_t) / max(done, 1) * (total - done))
            progress_cb(done, total - skipped, eta)
            continue

        for symptom in symptoms:
            prompt_formatted = prompt.format(symptom=symptom, report="".join(report))

            try:
                num_prompt_tokens = len(llm.tokenize(prompt_formatted.encode("utf-8"), add_bos=True))
                if num_prompt_tokens >= ctx_size - n_predict:
                    print(
                        f"[warn] prompt may be too long: prompt={num_prompt_tokens}, ctx={ctx_size}, n_predict={n_predict}"
                    )
            except Exception:
                pass

            # call model in-process
            grammar_obj = None
            if isinstance(grammar, str) and grammar.strip():
                grammar_obj = LlamaGrammar.from_string(grammar)

            # call model in-process
            out = llm(
                prompt=prompt_formatted,
                max_tokens=n_predict,
                temperature=temperature,
                stop=None,
                grammar=grammar_obj,   
                echo=False,
            )
            text = out["choices"][0]["text"]

            if _id not in results:
                results[_id] = {}
            results[_id]["report"] = report
            results[_id]["symptom"] = symptom
            results[_id]["summary"] = {"content": text}

        done = (i + 1) - skipped
        eta = _format_time((time.time() - start_t) / max(done, 1) * (total - done))
        progress_cb(done, total - skipped, eta)

    # 3) postprocess and return
    llm_metadata = {
        "model_name": model_name_alias if model_name_alias else os.path.basename(model_file),
        "prompt": prompt,
        "symptoms": list(symptoms),
        "temperature": temperature,
        "n_predict": n_predict,
        "ctx_size": ctx_size,
        "grammar": grammar,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    aggregated_df, err = postprocess_grammar(results, df, llm_metadata, debug)
    return (aggregated_df, err), zip_file_path





 # pip install python-dateutil

# 1) 把字符串解析为日期（尽量宽容）
def _try_parse_date(s: str) -> Optional[datetime]:
    try:
        # dayfirst/ yearfirst 视你的语料而定；加 fuzzy=True 宽容非日期词
        return dtparser.parse(s, fuzzy=True)
    except Exception:
        return None

# 2) 在原文中找“像日期”的片段，并返回(原文子串, 解析后的日期对象)
#    这里可先用多个正则快速初筛，再对命中进行 parse 验证
_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_MONTHS_LONG = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_ORD = r"(?:st|nd|rd|th)?"
_WDAY = r"(?:Mon|Tue|Tues|Wed|Thu|Thur|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_APOS = r"(?:'|')"  # 直引号或弯引号
_TAIL = r"(?:[.,;)]?)"  # 允许一个尾随标点（用于回收时也可不含）

_DATE_REGEXES = [
    # Mon, May 20, 2024 / Monday, May 20, 2024 / on May 20, 2024
    rf"(?:\b(?:{_WDAY})\,?\s+)?(?:on\s+)?\b{_MONTHS}\.?\s+\d{{1,2}}{_ORD}\,?\s+\d{{2,4}}\b{_TAIL}",
    # January 1st, 1980
    rf"\b{_MONTHS_LONG}\s+\d{{1,2}}{_ORD}\,?\s+\d{{4}}\b{_TAIL}",
    # 1 January 1980
    rf"\b\d{{1,2}}\s+{_MONTHS_LONG}\s+\d{{4}}\b{_TAIL}",
    # May 14–20, 2024 / May 14-20, 2024 (范围)
    rf"\b{_MONTHS_LONG}|{_MONTHS}\.?\s+\d{{1,2}}{_ORD}\s*(?:\-|-|—)\s*\d{{1,2}}{_ORD}\,?\s+\d{{4}}\b{_TAIL}",
    # 05/21/2024, 5-21-24
    r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b" + _TAIL,
    # 2024/05/21, 2024-05-21
    r"\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b" + _TAIL,
    # May 20, '24 / May 20, ’24（撇号年份）
    rf"\b{_MONTHS_LONG}|{_MONTHS}\.?\s+\d{{1,2}}{_ORD}\,?\s+{_APOS}\d{{2}}\b{_TAIL}",
]

def find_date_spans_in_text(text: str) -> List[Tuple[str, datetime]]:
    spans = []
    for pat in _DATE_REGEXES:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw = m.group(0)
            dt = _try_parse_date(raw)
            if dt:
                spans.append((raw, dt))
    return spans

# 3) 根据“语义日期”（LLM 输出的标准化或可解析值），回收原文中的全部等价片段
def collect_verbatim_dates_for_value(report_text: str, value: str) -> List[str]:
    dt_val = _try_parse_date(value)
    if not dt_val:
        return []
    # 统一到“年月日三元组”比较；避免时区/时间成分干扰
    ymd_val = (dt_val.year, dt_val.month, dt_val.day)

    verbatims = []
    for raw, dt in find_date_spans_in_text(report_text):
        if (dt.year, dt.month, dt.day) == ymd_val:
            verbatims.append(raw)
    return list(set(verbatims))  # 去重





# -----------------------
# Postprocess (ported from routes.py, no socketio)
# -----------------------
def postprocess_grammar(result: dict[Any, Any], df: pd.DataFrame, llm_metadata: dict, debug: bool = False):
    print("POSTPROCESSING GRAMMAR")

    extracted_data = []
    error_count = 0

    for i, (_id, info) in enumerate(result.items()):
        print(f"Processing report {i} of {len(result)}")
        content = info["summary"]["content"]

        # Parse LLM output into dict
        if content.endswith("<|eot_id|>"):
            content = content[:-len("<|eot_id|>")]
        if content.endswith("</s>"):
            content = content[:-len("</s>")]

        info_dict_raw, parse_note = _try_json_loads_with_patch(content)

        if info_dict_raw is None:
            print(f"Failed to parse LLM output. ({content=})")
            if debug:
                import traceback; traceback.print_exc()
            info_dict = {}
            error_count += 1
        else:
            # 统一空值
            info_dict = {
                k: ("" if is_empty_string_nan_or_none(v) else v)
                for k, v in info_dict_raw.items()
            }
            print("Successfully parsed LLM output." + (f" ({parse_note})" if parse_note else ""))

        # attach metadata from df (id match)
        meta = df[df["id"] == _id]["metadata"].iloc[0]
        import ast as _ast
        meta = _ast.literal_eval(meta)
        meta["llm_processing"] = llm_metadata

        import json
        row_out = {"report": info["report"], "id": _id, "metadata": json.dumps(meta)}

        # 先把 LLM 抽取的字段写入
        for k, v in info_dict.items():
            row_out[k] = v

        # 再补充：基于 LLM 的“语义日期值”，回收原文中的等价 verbatim 片段
        date_like_keys = {
            "patientBirthDate","admissionDate","dischargeDate","deathDate","otherDateElements"
        }
        row_verbatim_boosts = []
        for k, v in info_dict.items():
            if k in date_like_keys and isinstance(v, str) and v.strip():
                row_verbatim_boosts.extend(
                    collect_verbatim_dates_for_value(info["report"], v)
                )
        # 存一个列表列，后面聚合时会一并拍平到 personal_info_list
        row_out["_verbatim_date_spans"] = list(set(row_verbatim_boosts))
        # 在 row_out 组装处，紧跟 row_out["_verbatim_date_spans"] 之后加：
        row_out["_all_date_spans"] = list({raw for raw, _ in find_date_spans_in_text(info["report"])})

# ——电话字段的语义回收（和日期同理）——
        phone_like_keys = {"patientPhoneNumber", "patientFaxNumber", "patientphonenumber"}  # 兼容大小写差异

        row_phone_verbatims = []
        for k, v in info_dict.items():
            if k in phone_like_keys and isinstance(v, str) and v.strip():
                row_phone_verbatims.extend(collect_verbatim_phones_for_value(info["report"], v))
        row_out["_verbatim_phone_spans"] = list(set(row_phone_verbatims))

        # 兜底：把原文中所有“像电话”的片段也收集（即使 LLM 没提到）
        row_out["_all_phone_spans"] = list({raw for raw, _ in find_phone_spans_in_text(info["report"])})



        extracted_data.append(row_out)

    out_df = pd.DataFrame(extracted_data)

    # merge chunks back by base_id (same logic as original)
    def extract_base_id(x: str) -> str:
        parts = x.split("$")
        base_id = parts[0]
        if len(parts) > 1:
            subparts = parts[1].split("_")
            if len(subparts) > 1 and subparts[-1].isdigit():
                return base_id + "$" + "_".join(subparts[:-1])
        return x

    out_df["base_id"] = out_df["id"].apply(extract_base_id)

    # aggregated_df = out_df.groupby("base_id").agg(
    #     lambda col: col.tolist() if col.name != "report" else " ".join(col)
    # ).reset_index()

    def _agg_flat(col):
        if col.name == "report":
            return " ".join(col)
        # 把每个单元里的 list 拿出来连成一层
        flat = []
        for v in col:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat

    aggregated_df = out_df.groupby("base_id").agg(_agg_flat).reset_index()

    # personal_info_list 组装（排除非字段列，包含 _verbatim_date_spans）
    cols_to_exclude = {"id", "base_id", "report", "metadata"}
    aggregated_df["personal_info_list"] = aggregated_df.apply(
        lambda row: [
            item
            for col, lst in row.items()
            if col not in cols_to_exclude
            for item in (lst if isinstance(lst, list) else [lst])
            if isinstance(item, str) and item.strip()
        ],
        axis=1,
    )

    def _prepare_pi_list(pi_list):
    # 去重 + 按长度降序
        uniq = sorted(set(x for x in pi_list if isinstance(x, str) and x.strip()), key=len, reverse=True)
        return uniq

    aggregated_df["masked_report"] = aggregated_df.apply(
        lambda row: replace_personal_info(row["report"], _prepare_pi_list(row["personal_info_list"]), []),
        axis=1
    )



    aggregated_df.drop(columns=["id"], inplace=True)
    aggregated_df.rename(columns={"base_id": "id"}, inplace=True)
    aggregated_df["metadata"] = aggregated_df["metadata"].apply(lambda x: x[0])

    return aggregated_df, error_count



if __name__ == "__main__":
    import sys

    try:
        preproc_path = input("Enter preprocessed CSV or ZIP path: ").strip()
        if not preproc_path:
            print("No preprocessed file provided.")
            sys.exit(1)

        model_file = input("Enter model .gguf path: ").strip()
        if not model_file:
            print("No model file provided.")
            sys.exit(1)

        if preproc_path.lower().endswith(".zip"):
            df_in = read_preprocessed_csv_from_zip(preproc_path)
            if df_in is None:
                print("Could not find a 'preprocessed_*.csv' inside the ZIP.")
                sys.exit(1)
        else:
            df_in = pd.read_csv(preproc_path)


        required_cols = {"report", "id", "metadata"}
        if not required_cols.issubset(df_in.columns):
            print(f"CSV missing required columns: {required_cols - set(df_in.columns)}")
            sys.exit(1)


        syms_raw = input("Enter symptoms (comma-separated, default: Patienteninfos): ").strip()
        symptoms = [s.strip() for s in syms_raw.split(",") if s.strip()] or ["Patienteninfos"]

        use_default_grammar = input("Use default grammar? [y/N]: ").strip().lower() == "y"
        grammar_str = default_grammar if use_default_grammar else ""

        prompt_override = input("Use default prompt? [Y/n]: ").strip().lower()
        prompt_str = default_prompt if prompt_override in ("", "y", "yes") else input("Enter prompt template: ")

        def _get_float(msg, default):
            v = input(f"{msg} (default {default}): ").strip()
            return float(v) if v else default

        def _get_int(msg, default):
            v = input(f"{msg} (default {default}): ").strip()
            return int(v) if v else default

        temperature = _get_float("Temperature", 0.1)
        n_predict   = _get_int("n_predict", 384)
        ctx_size    = _get_int("ctx_size", 4096)
        n_gpu_layers= _get_int("n_gpu_layers", 0)


        (df_out, err_cnt), _ = extract_from_report_inprocess(
            df=df_in,
            model_file=model_file,
            symptoms=symptoms,
            n_predict=n_predict,
            prompt=prompt_str,
            temperature=temperature,
            grammar=grammar_str,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            job_id=None,
            zip_file_path=preproc_path if preproc_path.lower().endswith(".zip") else None,
            debug=False,
            model_name_alias=os.path.basename(model_file),
        )

        out_csv = "llm_output.csv"
        df_out.to_csv(out_csv, index=False)


       

        print(f"\n[OK] wrote: {out_csv}")
        print(f"[INFO] parse errors: {err_cnt}")
        print("\n[HEAD]")
        print(df_out.head().to_string(index=False))

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

