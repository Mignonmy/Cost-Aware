import argparse
import csv
import json
import os
import sys
from glob import glob
from typing import Dict, List

import matplotlib.pyplot as plt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_ALIGNED_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "RAG_aligned"))
if RAG_ALIGNED_DIR not in sys.path:
    sys.path.insert(0, RAG_ALIGNED_DIR)

from eval_utils import evaluate_prediction  # noqa: E402


def infer_dataset_name(path: str) -> str:
    base = os.path.basename(path).lower()
    for name in ["medqa", "mmlu", "hotpotqa", "nq", "triviaqa"]:
        if name in base:
            return name
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_output", "")


def evaluate_one_file(path: str) -> Dict:
    count = 0
    em_sum = 0.0
    f1_sum = 0.0
    cost_sum = 0.0
    num_passages_sum = 0.0
    num_steps_sum = 0.0
    empty_pred = 0
    metric_types = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item = dict(item)
            item["actual_cost"] = float(item.get("spent", 0.0) or 0.0)

            result = evaluate_prediction(item)
            em_sum += result["em"]
            f1_sum += result["f1"]
            cost_sum += float(item.get("actual_cost", 0.0))
            num_passages_sum += float(item.get("num_passages", 0) or 0)
            num_steps_sum += float(len(item.get("trajectory") or []))
            if result["empty"]:
                empty_pred += 1
            metric_types.add(result["metric_type"])
            count += 1

    if count == 0:
        return {
            "count": 0,
            "avg_cost": 0.0,
            "avg_num_passages": 0.0,
            "avg_num_steps": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "empty_prediction_rate": 0.0,
            "metric_type": "unknown",
        }

    return {
        "count": count,
        "avg_cost": cost_sum / count,
        "avg_num_passages": num_passages_sum / count,
        "avg_num_steps": num_steps_sum / count,
        "em": em_sum / count,
        "f1": f1_sum / count,
        "empty_prediction_rate": empty_pred / count,
        "metric_type": ",".join(sorted(metric_types)),
    }


def save_csv(rows: List[Dict], output_csv: str):
    fields = [
        "group",
        "dataset",
        "method_dir",
        "kind",
        "setting",
        "method",
        "file",
        "count",
        "avg_cost",
        "avg_num_passages",
        "avg_num_steps",
        "em",
        "f1",
        "empty_prediction_rate",
        "metric_type",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot(rows: List[Dict], output_png: str, x_axis: str, title: str, annotate_field: str):
    if not rows:
        return

    x = [r["avg_cost"] if x_axis == "cost" else r["avg_num_passages"] for r in rows]
    y_f1 = [r["f1"] * 100 for r in rows]
    y_em = [r["em"] * 100 for r in rows]
    labels = [r[annotate_field] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_f1, marker="o", label="F1 (%)")
    plt.plot(x, y_em, marker="s", label="EM (%)")

    for xi, yi, lb in zip(x, y_f1, labels):
        plt.annotate(lb, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.xlabel("Average Actual Cost" if x_axis == "cost" else "Average Number of Passages")
    plt.ylabel("Performance (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Agent-side evaluator aligned to RAG_aligned/evaluate_unified.py for one result jsonl file or a directory of result jsonl files."
    )
    ap.add_argument("--input_jsonl", type=str, default=None)
    ap.add_argument("--result_dir", type=str, default=None)
    ap.add_argument("--pattern", type=str, default="*_output.jsonl")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--method_dir", type=str, default="")
    ap.add_argument("--setting", type=str, default="default")
    ap.add_argument("--method", type=str, default="budgeted_tier_agent")
    ap.add_argument("--x_axis", type=str, default="cost", choices=["cost", "num_passages"])
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--output_png", type=str, default=None)
    args = ap.parse_args()

    if not args.input_jsonl and not args.result_dir:
        raise ValueError("Provide either --input_jsonl or --result_dir")
    if args.input_jsonl and args.result_dir:
        raise ValueError("Provide only one of --input_jsonl or --result_dir")

    if args.input_jsonl:
        input_files = [args.input_jsonl]
        stem = os.path.splitext(args.input_jsonl)[0] + "_evaluation_unified"
        default_method_dir = args.method_dir or os.path.dirname(args.input_jsonl)
    else:
        input_files = sorted(glob(os.path.join(args.result_dir, args.pattern)))
        if not input_files:
            raise FileNotFoundError(f"No files found: {os.path.join(args.result_dir, args.pattern)}")
        stem = os.path.join(args.result_dir, "evaluation_unified_all")
        default_method_dir = args.method_dir or args.result_dir

    rows = []
    for path in input_files:
        metrics = evaluate_one_file(path)
        basename = os.path.basename(path)
        dataset = args.dataset or infer_dataset_name(path)
        rows.append(
            {
                "group": dataset if args.result_dir else args.setting,
                "dataset": dataset,
                "method_dir": default_method_dir,
                "kind": "budget",
                "setting": args.setting,
                "method": args.method,
                "file": basename,
                "count": metrics["count"],
                "avg_cost": metrics["avg_cost"],
                "avg_num_passages": metrics["avg_num_passages"],
                "avg_num_steps": metrics["avg_num_steps"],
                "em": metrics["em"],
                "f1": metrics["f1"],
                "empty_prediction_rate": metrics["empty_prediction_rate"],
                "metric_type": metrics["metric_type"],
            }
        )

    output_csv = args.output_csv or f"{stem}.csv"
    output_json = args.output_json or f"{stem}.json"
    output_png = args.output_png or f"{stem}.png"

    save_csv(rows, output_csv)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    plot(rows, output_png, x_axis=args.x_axis, title="Evaluation (budget)", annotate_field="group")

    print(f"Evaluated files: {len(rows)}")
    print(f"CSV saved to: {output_csv}")
    print(f"JSON saved to: {output_json}")
    print(f"Figure saved to: {output_png}")
    print("")
    for r in rows:
        print(
            f"- group={r['group']:<24} | n={r['count']:4d} | avg_cost={r['avg_cost']:.2f} | "
            f"avg_num_passages={r['avg_num_passages']:.2f} | avg_num_steps={r['avg_num_steps']:.2f} | EM={r['em']*100:.2f}% | "
            f"F1={r['f1']*100:.2f}% | metric={r['metric_type']} | file={r['file']}"
        )


if __name__ == "__main__":
    main()
