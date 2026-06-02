import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM
from tqdm import tqdm
import argparse
import math

from agent_core import BudgetedTierAgent, KnowledgeStore, extract_boxed_answer

def parse_budget(x):

    if x.lower() in ("inf", "infinity"):

        return math.inf

    return int(x)

def normalize_budget(budget): 
    if isinstance(budget, float) and math.isinf(budget):
        return 10**18
    return int(budget)

def parse_args():
    ap = argparse.ArgumentParser(description="Budgeted tiered retrieval agent runner")
    ap.add_argument("--input_jsonl", required=True, help="Input dataset jsonl (must contain id and question fields)")
    ap.add_argument("--tiered_retrieval_jsonl", required=True, help="Tiered retrieval jsonl produced by tiering script")
    ap.add_argument("--output_jsonl", required=True)

    ap.add_argument("--model_path", default="")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.4)

    ap.add_argument("--budget", type=parse_budget, default=20, help="Budget per question")
    ap.add_argument("--tier_costs", type=str, default="0:0,1:1,2:4", help="Comma list like 0:0,1:1,2:4")

    ap.add_argument("--max_steps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048)

    return ap.parse_args()


def parse_tier_costs(s: str) -> dict:
    out = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        k, v = part.split(":", 1)
        out[int(k)] = int(v)
    return out


def infer_dataset_name(input_path: str) -> str:
    base = os.path.basename(input_path).lower()
    for name in ["medqa", "mmlu", "hotpotqa", "nq", "triviaqa"]:
        if name in base:
            return name
    return ""


def format_question(obj: dict) -> str:
    question = str(obj.get("question") or obj.get("input") or "")
    options = obj.get("options")
    if not isinstance(options, dict) or not options:
        return question

    option_lines = []
    for key, value in options.items():
        option_lines.append(f"{key}. {value}")
    return f"{question}\n\nOptions:\n" + "\n".join(option_lines)


def extract_ground_truth(obj: dict):
    if obj.get("answer") is not None:
        return obj.get("answer")
    if obj.get("ground_truth") is not None:
        return obj.get("ground_truth")

    output = obj.get("output")
    if isinstance(output, list):
        answers = []
        for item in output:
            if isinstance(item, dict) and item.get("answer") is not None:
                answers.append(item.get("answer"))
        if answers:
            return answers[0] if len(answers) == 1 else answers
    return None


def main():
    args = parse_args()
    default_dataset_name = infer_dataset_name(args.input_jsonl)

    store = KnowledgeStore(args.tiered_retrieval_jsonl)
    store.load()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )

    agent = BudgetedTierAgent(
        llm=llm,
        tokenizer=tokenizer,
        tier_costs=parse_tier_costs(args.tier_costs),
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as f_out:
        for line in tqdm(f_in, desc="Agent", unit="q"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id") or obj.get("qid") or "")
            question = format_question(obj)
            if not qid:
                raise KeyError("Missing id in input_jsonl")
            budget = normalize_budget(args.budget)
            result = agent.run_one(qid=qid, question=str(question), budget=budget, store=store) 
            result["prediction"] = extract_boxed_answer(result.get("prediction", ""))
            result["dataset_name"] = obj.get("dataset_name") or default_dataset_name
            result["ground_truth"] = extract_ground_truth(obj)
            result["question_raw"] = obj.get("question") or obj.get("input") or ""
            result["options"] = obj.get("options")
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
