import json
import csv
from pathlib import Path


def extract_dataset_name(file_path: Path) -> str:
    """
    triviaqa_mini_output.jsonl -> triviaqa
    hotpotqa_mini_output.jsonl -> hotpotqa
    medqa_mini_output.jsonl -> medqa
    """
    filename = file_path.name

    if filename.endswith("_mini_output.jsonl"):
        return filename.replace("_mini_output.jsonl", "")

    return file_path.stem.split("_")[0]


def load_jsonl_file(jsonl_path: Path):
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                records.append(item)
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse JSON in {jsonl_path}, line {line_num}: {e}")

    return records


def extract_records_from_folder(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    all_records = []

    jsonl_files = sorted(folder_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"[Warning] No *_mini_output.jsonl files found in {folder_path}")

    for jsonl_file in jsonl_files:
        dataset_name = extract_dataset_name(jsonl_file)
        data_items = load_jsonl_file(jsonl_file)

        for idx, item in enumerate(data_items):
            used_tiers = item.get("used_tiers", [])

            if used_tiers is None:
                used_tiers = []
            
            batch_allocations = item.get("batch_allocations", [])

            record = {
                "dataset_name": dataset_name,
                "sample_index": idx,
                "id": item.get("id", ""),
                "budget": item.get("budget", None),
                "spent": item.get("spent", None),
                "len_used_tiers": len(used_tiers),
                "used_tiers": json.dumps(used_tiers, ensure_ascii=False),
                "batch_id": item.get("batch_id", None),
                "batch_index":item.get("batch_index", None),
                "batch_size":item.get("batch_size", None),
                "batch_total_budget":item.get("batch_total_budget", None),
                "allocated_budget":item.get("allocated_budget", None),
                "budget_allocation_source":item.get("budget_allocation_source", None),
                "budget_allocation_detail":item.get("budget_allocation_detail", None),
                "budget_allocation_reason":item.get("budget_allocation_reason", None),
                "budget_allocation_parsed":item.get("budget_allocation_parsed", None),
                # "":,
            }

            all_records.append(record)

    return all_records


def save_to_csv(records, output_csv_path):
    fieldnames = [
        "dataset_name",
        "sample_index",
        "id",
        "budget",
        "spent",
        "len_used_tiers",
        "used_tiers",
        "batch_id",
        "batch_index",
        "batch_size",
        "batch_total_budget",
        "allocated_budget",
        "budget_allocation_source",
        "budget_allocation_detail",
        "budget_allocation_reason",
        "budget_allocation_parsed",
        
    ]

    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} records to {output_csv_path}")


if __name__ == "__main__":
    folder_path = "Agent/result_optimized_llama_budget40"

    output_csv_path = Path(folder_path) / "conclusion_all.csv"

    records = extract_records_from_folder(folder_path)
    save_to_csv(records, output_csv_path)