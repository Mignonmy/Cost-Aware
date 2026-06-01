import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from eval_utils import evaluate_prediction


@dataclass(frozen=True)
class ResultFileInfo:
    path: str
    basename: str
    dataset: str  # derived from path relative to result_dir
    method_dir: str  # derived from path relative to result_dir (directory name)
    kind: str  # budget|topk|no_passage|other
    setting: str  # e.g. "5", "infinite", "topk_10", "no_passage"
    method: str  # e.g. "mmr_cost_aware", "relevance_only", "topk", "no_passage", "unknown"


def derive_dataset_and_method_dir(result_dir: str, path: str) -> Dict[str, str]:
    """
    Supports both layouts:
    - Flat: result_dir/results*.jsonl
    - Nested: result_dir/<dataset>/<method>/results*.jsonl (or deeper; we take first two segments)
    """
    rel = os.path.relpath(path, result_dir)
    parts = [p for p in rel.split(os.sep) if p and p not in (".", "..")]
    dataset = parts[0] if len(parts) >= 2 else "all"
    method_dir = parts[1] if len(parts) >= 3 else "all"
    return {"dataset": dataset, "method_dir": method_dir}


def parse_result_filename(result_dir: str, path: str) -> ResultFileInfo:
    name = os.path.basename(path)
    meta = derive_dataset_and_method_dir(result_dir, path)

    if re.search(r"results_no_passage\.jsonl$", name):
        return ResultFileInfo(
            path=path,
            basename=name,
            dataset=meta["dataset"],
            method_dir=meta["method_dir"],
            kind="no_passage",
            setting="no_passage",
            method="no_passage",
        )

    m = re.search(r"results_topk_(\d+)\.jsonl$", name)
    if m:
        k = m.group(1)
        return ResultFileInfo(
            path=path,
            basename=name,
            dataset=meta["dataset"],
            method_dir=meta["method_dir"],
            kind="topk",
            setting=f"topk_{k}",
            method="topk",
        )

    # Budgeted results.
    # Supports:
    # - results_budget_5.jsonl
    # - results_budget_5_relevance_only.jsonl
    # - results_budget_5_mmr_cost_aware.jsonl
    # - results_budget_infinite_redundancy_aware.jsonl
    m = re.match(r"results_budget_([^_]+)(?:_(.+?))?\.jsonl$", name)
    if m:
        budget = m.group(1)
        suffix = m.group(2) or ""
        method = suffix or "budget"
        return ResultFileInfo(
            path=path,
            basename=name,
            dataset=meta["dataset"],
            method_dir=meta["method_dir"],
            kind="budget",
            setting=budget,
            method=method,
        )

    return ResultFileInfo(
        path=path,
        basename=name,
        dataset=meta["dataset"],
        method_dir=meta["method_dir"],
        kind="other",
        setting=os.path.splitext(name)[0],
        method="unknown",
    )


def setting_sort_key(label: str) -> float:
    if label == "no_passage":
        return -1.0
    if label == "infinite":
        return float("inf")
    m = re.match(r"topk_(\d+)$", label)
    if m:
        return float(m.group(1))
    try:
        return float(label)
    except ValueError:
        return float("inf")


def group_key(info: ResultFileInfo, group_by: str) -> str:
    if group_by == "file":
        return info.basename
    if group_by == "dataset":
        return info.dataset
    if group_by == "setting":
        return info.setting
    if group_by == "method":
        return info.method
    if group_by == "method_dir":
        return info.method_dir
    if group_by == "setting_method":
        return f"{info.setting}::{info.method}"
    if group_by == "dataset_method_dir":
        return f"{info.dataset}::{info.method_dir}"
    if group_by == "dataset_setting":
        return f"{info.dataset}::{info.setting}"
    if group_by == "dataset_setting_method":
        return f"{info.dataset}::{info.setting}::{info.method}"
    raise ValueError(f"Unknown group_by: {group_by}")


def evaluate_one_file(path: str) -> Dict:
    count = 0
    em_sum = 0.0
    f1_sum = 0.0
    cost_sum = 0.0
    num_passages_sum = 0.0
    empty_pred = 0
    metric_types = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cost_sum += float(item.get("actual_cost", 0.0))
            num_passages_sum += float(item.get("num_passages", 0) or 0)
            result = evaluate_prediction(item)
            em_sum += result["em"]
            f1_sum += result["f1"]
            if result["empty"]:
                empty_pred += 1
            metric_types.add(result["metric_type"])
            count += 1

    if count == 0:
        return {
            "count": 0,
            "avg_cost": 0.0,
            "avg_num_passages": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "empty_prediction_rate": 0.0,
            "metric_type": "unknown",
        }

    return {
        "count": count,
        "avg_cost": cost_sum / count,
        "avg_num_passages": num_passages_sum / count,
        "em": em_sum / count,
        "f1": f1_sum / count,
        "empty_prediction_rate": empty_pred / count,
        "metric_type": ",".join(sorted(metric_types)),
    }


def save_csv(rows: List[Dict], output_csv: str):
    if not rows:
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
            "em",
            "f1",
            "empty_prediction_rate",
            "metric_type",
        ]
    else:
        fields = list(rows[0].keys())

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
    ap = argparse.ArgumentParser(description="Unified evaluator for RAG_aligned result jsonl files.")
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="results*.jsonl")
    ap.add_argument(
        "--kind",
        type=str,
        default="all",
        choices=["all", "budget", "topk", "no_passage"],
        help="Filter which result file kinds to include.",
    )
    ap.add_argument(
        "--group_by",
        type=str,
        default="setting",
        choices=[
            "setting",
            "setting_method",
            "method",
            "file",
            "dataset",
            "method_dir",
            "dataset_method_dir",
            "dataset_setting",
            "dataset_setting_method",
        ],
        help="How to group/label rows.",
    )
    ap.add_argument(
        "--x_axis",
        type=str,
        default="cost",
        choices=["cost", "num_passages"],
        help="X-axis for plot.",
    )
    ap.add_argument(
        "--sort_by",
        type=str,
        default="setting",
        choices=["setting", "avg_cost"],
        help="Sorting for output rows/plot.",
    )
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--output_png", type=str, default=None)
    args = ap.parse_args()

    # Support nested directory layout by scanning recursively.
    files = sorted(glob(os.path.join(args.result_dir, "**", args.pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No files found: {os.path.join(args.result_dir, '**', args.pattern)}")

    infos = [parse_result_filename(args.result_dir, p) for p in files]
    if args.kind != "all":
        infos = [i for i in infos if i.kind == args.kind]

    if not infos:
        raise FileNotFoundError("No files left after filtering by --kind")

    rows: List[Dict] = []
    for info in infos:
        m = evaluate_one_file(info.path)
        rows.append(
            {
                "group": group_key(info, args.group_by),
                "dataset": info.dataset,
                "method_dir": info.method_dir,
                "kind": info.kind,
                "setting": info.setting,
                "method": info.method,
                "file": info.basename,
                "count": m["count"],
                "avg_cost": m["avg_cost"],
                "avg_num_passages": m["avg_num_passages"],
                "em": m["em"],
                "f1": m["f1"],
                "empty_prediction_rate": m["empty_prediction_rate"],
                "metric_type": m["metric_type"],
            }
        )

    if args.sort_by == "avg_cost":
        rows = sorted(rows, key=lambda r: r["avg_cost"])
    else:
        rows = sorted(rows, key=lambda r: setting_sort_key(r["setting"]))

    stem = f"evaluation_unified_{args.kind}_{args.group_by}_{args.x_axis}_{args.sort_by}"
    output_csv = args.output_csv or os.path.join(args.result_dir, f"{stem}.csv")
    output_json = args.output_json or os.path.join(args.result_dir, f"{stem}.json")
    output_png = args.output_png or os.path.join(args.result_dir, f"{stem}.png")

    save_csv(rows, output_csv)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    plot_title = "Evaluation" if args.kind == "all" else f"Evaluation ({args.kind})"
    plot(rows, output_png, x_axis=args.x_axis, title=plot_title, annotate_field="group")

    print(f"Evaluated files: {len(rows)}")
    print(f"CSV saved to: {output_csv}")
    print(f"JSON saved to: {output_json}")
    print(f"Figure saved to: {output_png}")
    print("")
    for r in rows:
        print(
            f"- group={r['group']:<24} | n={r['count']:4d} | avg_cost={r['avg_cost']:.2f} | "
            f"avg_num_passages={r['avg_num_passages']:.2f} | EM={r['em']*100:.2f}% | "
            f"F1={r['f1']*100:.2f}% | metric={r['metric_type']} | file={r['file']}"
        )


if __name__ == "__main__":
    main()
