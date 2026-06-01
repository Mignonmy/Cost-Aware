from pathlib import Path

import pandas as pd


paths = [
    "Agent/result_optimized_budget10/evaluation_unified_all.csv",
    "Agent/result_optimized/evaluation_unified_all.csv",
    "Agent/result_optimized_budget30/evaluation_unified_all.csv",
    "Agent/result_optimized_budget40/evaluation_unified_all.csv",
    "Agent/result_optimized_budgetinf/evaluation_unified_all.csv",
    "Agent/result_optimized_llama_budget10/evaluation_unified_all.csv",
    "Agent/result_optimized_llama/evaluation_unified_all.csv",
    "Agent/result_optimized_llama_budget30/evaluation_unified_all.csv",
    "Agent/result_optimized_llama_budget40/evaluation_unified_all.csv",
    "Agent/result_optimized_llama_budgetinf/evaluation_unified_all.csv",
]

budgets = [
    10,
    20,
    30,
    40,
    "inf",
    10,
    20,
    30,
    40,
    "inf",
]

models = [
    "Qwen3-8B",
    "Qwen3-8B",
    "Qwen3-8B",
    "Qwen3-8B",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct",
]


KEEP_COLUMNS = [
    "dataset",
    "avg_cost",
    "avg_num_passages",
    "avg_num_steps",
    "em",
    "f1",
]

FINAL_COLUMNS = [
    "model",
    "dataset",
    "budget",
    "avg_cost",
    "avg_num_passages",
    "avg_num_steps",
    "em",
    "f1",
]

output_path = "Agent/result_budget/merged_results.csv"


def read_one_table(path, budget, model):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {path}: {missing_cols}\n"
            f"Existing columns: {list(df.columns)}"
        )

    out = df[KEEP_COLUMNS].copy()

    out["model"] = model
    out["budget"] = budget

    out = out[FINAL_COLUMNS]

    return out


def merge_tables(paths, budgets, models, output_path):
    if not (len(paths) == len(budgets) == len(models)):
        raise ValueError(
            f"Length mismatch: "
            f"paths={len(paths)}, budgets={len(budgets)}, models={len(models)}"
        )

    all_dfs = []

    for path, budget, model in zip(paths, budgets, models):
        print(f"Reading: {path}")
        print(f"  model = {model}, budget = {budget}")

        one_df = read_one_table(
            path=path,
            budget=budget,
            model=model,
        )

        all_dfs.append(one_df)

    merged = pd.concat(all_dfs, ignore_index=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_path, index=False)

    print(f"\nSaved merged table to: {output_path}")
    print(f"Total rows: {len(merged)}")
    print("\nPreview:")
    print(merged.head())

    return merged


if __name__ == "__main__":
    merged_df = merge_tables(
        paths=paths,
        budgets=budgets,
        models=models,
        output_path=output_path,
    )