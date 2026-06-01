import os
from typing import Dict, Optional, Tuple


OPTION_LABELS = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
}
OPTION_FIELDS = {
    "A": "opa",
    "B": "opb",
    "C": "opc",
    "D": "opd",
}


def infer_dataset_name(input_file: str) -> str:
    name = os.path.basename(input_file)
    stem = os.path.splitext(name)[0].lower()
    return stem.split("_")[0] if "_" in stem else stem


def get_dataset_name(data: Dict, input_file: str) -> str:
    if "dataset_name" in data and isinstance(data["dataset_name"], str):
        return data["dataset_name"].lower()
    return infer_dataset_name(input_file)



def get_query_text(data: Dict) -> str:
    return data.get("question") or data.get("input") or ""


def get_option_qa_options(data: Dict) -> Optional[Dict[str, str]]:
    opts = data.get("options")
    if not isinstance(opts, dict):
        return None
    cleaned = {}
    for k, v in opts.items():
        if k is None or v is None:
            continue
        key = str(k).strip().upper()
        if not key:
            continue
        cleaned[key] = str(v).strip()
    return cleaned or None


def is_option_qa(data: Dict, dataset_name: str) -> bool:
    opts = get_option_qa_options(data)
    ans = data.get("answer")
    if not opts or ans is None:
        return False
    ans_label = str(ans).strip().upper()
    return ans_label in opts




def format_option_qa_question(data: Dict) -> str:
    question = data.get("question") or data.get("input") or ""
    opts = get_option_qa_options(data) or {}
    keys = sorted(opts.keys())
    options_str = "\n".join([f"{k}. {opts[k]}" for k in keys])
    if options_str:
        return f"Question: {question}\n\nOptions:\n{options_str}"
    return f"Question: {question}"


def format_question_for_prompt(data: Dict, dataset_name: str) -> str:
    if is_option_qa(data, dataset_name):
        return format_option_qa_question(data)
    return get_query_text(data)




def get_ground_truth(data: Dict, dataset_name: str) -> str:
    if is_option_qa(data, dataset_name):
        # For option QA datasets, ground truth is the option label.
        return str(data.get("answer", "")).strip().upper()
    if "answer" in data and isinstance(data.get("answer"), str):
        return data.get("answer", "")
    output = data.get("output", [{}])
    if isinstance(output, list) and output:
        return output[0].get("answer", "")
    return ""


def get_passage_docid(p: Dict):
    return p.get("docid") or p.get("id")


def get_passage_score(p: Dict) -> Optional[float]:
    score = p.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def build_system_prompt(use_context: bool, dataset_name: str) -> str:


    if use_context:
        return (
            "You are a helpful assistant. Use the context to answer the question briefly. "
            "Put the final answer in \\boxed{}. Please don't think too long."
        )
    return (
        "You are a helpful assistant. Answer the question briefly. "
        "Put the final answer in \\boxed{}. Please don't think too long."
    )
