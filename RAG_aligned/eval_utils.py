import re
import string
from collections import Counter
from typing import Optional


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("\\text{", "").replace("}", "").replace("$", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_option_label(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    boxed = re.search(r"\\boxed\s*\{([^}]*)\}", text, re.IGNORECASE)
    if boxed:
        text = boxed.group(1)

    patterns = [
        r"^\s*([ABCD])\b",
        r"\boption\s*([ABCD])\b",
        r"\banswer\s*(?:is|:)\s*([ABCD])\b",
        r"\bchoose\s*([ABCD])\b",
        r"\b([ABCD])\s*\.\s*",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # fallback: first standalone A/B/C/D
    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def get_item_dataset_name(item: dict) -> str:
    return str(item.get("dataset_name", "")).lower()


def is_medmcqa_item(item: dict) -> bool:
    if get_item_dataset_name(item) == "medmcqa":
        return True
    return bool(item.get("ground_truth_option"))


def is_option_qa_item(item: dict) -> bool:
    ds = get_item_dataset_name(item)
    if ds in {"pubmedqa", "bioasq"}:
        return True
    gt = str(item.get("ground_truth", "")).strip().upper()
    return gt in {"A", "B", "C", "D", "E"}


# def evaluate_prediction(item: dict) -> dict:
#     pred = item.get("prediction", "")
#     gt = item.get("ground_truth", "")

#     if is_medmcqa_item(item) or is_option_qa_item(item):
#         gt_option = str(item.get("ground_truth_option") or gt or "").strip().upper()
#         pred_option = extract_option_label(pred)
#         correct = float(pred_option == gt_option and gt_option != "")
#         empty = float(pred_option == "")
#         return {
#             "em": correct,
#             "f1": correct,
#             "empty": empty,
#             "pred_option": pred_option,
#             "gt_option": gt_option,
#             "metric_type": "option_match",
#         }

#     return {
#         "em": exact_match_score(pred, gt),
#         "f1": f1_score(pred, gt),
#         "empty": float(normalize_answer(pred) == ""),
#         "pred_option": "",
#         "gt_option": "",
#         "metric_type": "qa_em_f1",
#     }

def evaluate_prediction(item: dict) -> dict:
    pred = item.get("prediction", "")
    gt = item.get("ground_truth", "")

    if is_medmcqa_item(item) or is_option_qa_item(item):
        gt_option = str(item.get("ground_truth_option") or gt or "").strip().upper()
        pred_option = extract_option_label(pred)
        correct = float(pred_option == gt_option and gt_option != "")
        empty = float(pred_option == "")
        return {
            "em": correct,
            "f1": correct,
            "empty": empty,
            "pred_option": pred_option,
            "gt_option": gt_option,
            "metric_type": "option_match",
        }

    # Keep the existing mature str logic unchanged.
    if not isinstance(gt, list):
        return {
            "em": exact_match_score(pred, gt),
            "f1": f1_score(pred, gt),
            "empty": float(normalize_answer(pred) == ""),
            "pred_option": "",
            "gt_option": "",
            "metric_type": "qa_em_f1",
        }

    # New logic: multiple acceptable gold answers.
    valid_gts = [g for g in gt if g is not None]

    if not valid_gts:
        valid_gts = [""]

    return {
        "em": max(exact_match_score(pred, g) for g in valid_gts),
        "f1": max(f1_score(pred, g) for g in valid_gts),
        "empty": float(normalize_answer(pred) == ""),
        "pred_option": "",
        "gt_option": "",
        "metric_type": "qa_em_f1",
    }
