import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM

from agent_core import BudgetedTierAgent, KnowledgeStore, build_tier_descriptions, extract_boxed_answer
from run_agent import extract_ground_truth, format_question, infer_dataset_name, parse_tier_costs


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Batch-budget runner: allocate one shared budget across a batch of questions, "
            "then run the existing budgeted tier agent with each allocated budget."
        )
    )
    ap.add_argument("--input_jsonl", required=True, help="Input dataset jsonl (must contain id and question fields)")
    ap.add_argument("--tiered_retrieval_jsonl", required=True, help="Tiered retrieval jsonl produced by tiering script")
    ap.add_argument("--output_jsonl", required=True)

    ap.add_argument("--model_path", default="")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.4)

    ap.add_argument("--batch_size", type=int, default=5, help="Number of questions per shared-budget batch")
    ap.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Alias for --budget_per_question, kept for compatibility with run_agent.py commands",
    )
    ap.add_argument(
        "--budget_per_question",
        type=int,
        default=20,
        help="Used to compute default shared budget: actual_batch_size * budget_per_question",
    )
    ap.add_argument(
        "--total_budget",
        type=int,
        default=None,
        help="Optional fixed shared budget for every batch. Default: actual_batch_size * budget_per_question",
    )
    ap.add_argument("--min_budget", type=int, default=0, help="Minimum allocated budget for each question")
    ap.add_argument("--max_budget", type=int, default=None, help="Optional maximum allocated budget for each question")
    ap.add_argument("--tier_costs", type=str, default="0:0,1:1,2:4", help="Comma list like 0:0,1:1,2:4")

    ap.add_argument("--max_steps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperature for per-question agent decisions/answers")
    ap.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for per-question agent decisions/answers")

    ap.add_argument("--allocation_temperature", type=float, default=0.7, help="Temperature for batch budget allocation")
    ap.add_argument("--allocation_max_tokens", type=int, default=1024, help="Max tokens for batch budget allocation")
    ap.add_argument(
        "--allocation_question_max_chars",
        type=int,
        default=1200,
        help="Max characters of each formatted question shown to the budget allocator",
    )
    ap.add_argument(
        "--save_allocation_raw",
        action="store_true",
        help="Store the raw allocator model output in every result row for debugging",
    )

    args = ap.parse_args()
    if args.budget is not None:
        args.budget_per_question = args.budget
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.budget_per_question < 0:
        raise ValueError("--budget_per_question must be non-negative")
    if args.total_budget is not None and args.total_budget < 0:
        raise ValueError("--total_budget must be non-negative")
    if args.min_budget < 0:
        raise ValueError("--min_budget must be non-negative")
    if args.max_budget is not None and args.max_budget < args.min_budget:
        raise ValueError("--max_budget must be >= --min_budget")
    return args


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def call_llm(llm, tokenizer, system: str, user: str, temperature: float, max_tokens: int) -> str:
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    out = llm.generate([prompt], sampling_params)[0]
    return out.outputs[0].text


def iter_batches(input_jsonl: str, batch_size: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id") or obj.get("qid") or "")
            if not qid:
                raise KeyError(f"Missing id in input_jsonl at line {line_no}")
            batch.append(
                {
                    "qid": qid,
                    "question": str(format_question(obj)),
                    "obj": obj,
                    "line_no": line_no,
                }
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_allocation_prompt(
    batch: List[dict],
    total_budget: int,
    tier_costs: Dict[int, int],
    min_budget: int,
    max_budget: Optional[int],
    question_max_chars: int,
) -> Tuple[str, str]:
    max_text = "none" if max_budget is None else str(max_budget)
    tier_desc = build_tier_descriptions(tier_costs)

    question_lines = []
    for idx, item in enumerate(batch, start=1):
        question = truncate_text(item["question"], question_max_chars)
        question_lines.append(f"{idx}. id={item['qid']}\n{question}")

    system = (
        "You are a budget allocation controller for a retrieval-augmented QA system. "
        "Before retrieval starts, allocate integer retrieval budgets across all questions in the batch. "
        "Each tier of retrieval has a known cost, and the downstream agent will spend the allocated budget on retrieval to gather evidence for answering. "
        "Questions that likely need multi-hop, official, professional, or domain-specific evidence should receive more budget. "
        "Questions that can likely be answered from the prompt or cheap evidence can receive less budget. "
        "Return only the final tagged TSV allocation block."
    )
    user = (
        f"Batch size: {len(batch)}\n"
        f"Total shared budget: {total_budget}\n"
        f"Per-question bounds: min={min_budget}, max={max_text}\n"
        f"Retrieval tier options:\n{tier_desc}\n\n"
        "Questions:\n"
        + "\n\n".join(question_lines)
        + "\n\n"
        "Allocate budgets with these strict rules:\n"
        "- Include every id exactly once.\n"
        "- Each budget must be a non-negative integer within the per-question bounds.\n"
        "- The sum of all budgets must equal the total shared budget.\n"
        "- Budget is spent only by retrieval; unused budget may remain after the downstream agent answers.\n\n"
        "Return the final allocation as plain text, not JSON. Use exactly this TSV block format:\n"
        "BEGIN_BUDGET_ALLOCATION\n"
        "<question id>\\t<integer budget>\\t<short reason>\n"
        "...\n"
        "END_BUDGET_ALLOCATION\n"
        "Do not put Markdown tables, bullets, or JSON inside the final block."
    )
    return system, user


def coerce_budget(value: Any, default: int = 1) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, dict):
        for key in ("budget", "allocated_budget", "retrieval_budget", "value"):
            if key in value:
                return coerce_budget(value.get(key), default=default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return int(round(float(match.group(0))))
                except ValueError:
                    pass
        return default


def json_candidates_from_text(raw: str) -> List[Any]:
    """Extract JSON-like objects/lists from messy LLM output."""
    text = "" if raw is None else str(raw)
    decoder = json.JSONDecoder()
    texts = [text.strip()]
    texts.extend(m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE))

    expanded_texts: List[str] = []
    seen_texts = set()
    for candidate_text in texts:
        queue = [candidate_text]
        for _ in range(3):
            if not queue:
                break
            current = queue.pop(0)
            if current in seen_texts:
                continue
            seen_texts.add(current)
            expanded_texts.append(current)
            try:
                loaded = json.loads(current)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, str) and loaded != current:
                queue.append(loaded.strip())
            unescaped = re.sub(r"\\+([{}\[\]\"])", r"\1", current).strip()
            if unescaped and unescaped != current:
                queue.append(unescaped)

    values: List[Any] = []
    seen_values = set()
    for candidate_text in expanded_texts:
        for start in [m.start() for m in re.finditer(r"[\{\[]", candidate_text)]:
            try:
                value, _ = decoder.raw_decode(candidate_text[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, (dict, list)):
                continue
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key not in seen_values:
                seen_values.add(key)
                values.append(value)
    return values


def first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d.get(key)
    return None


def qid_from_index(value: Any, qids: List[str]) -> Optional[str]:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= len(qids):
        return qids[idx - 1]
    if 0 <= idx < len(qids):
        return qids[idx]
    return None


def qid_from_item(item: Dict[str, Any], qids: List[str], default_idx: Optional[int]) -> Optional[str]:
    qid_keys = ("id", "qid", "question_id", "questionId", "sample_id", "query_id")
    index_keys = ("index", "idx", "question_index", "question_idx", "question_number", "question_no", "no")

    for key in qid_keys:
        if key not in item:
            continue
        value = item.get(key)
        if isinstance(value, (dict, list)):
            continue
        qid = str(value)
        if qid in qids:
            return qid
        by_index = qid_from_index(value, qids)
        if by_index is not None:
            return by_index

    for key in index_keys:
        if key in item:
            by_index = qid_from_index(item.get(key), qids)
            if by_index is not None:
                return by_index

    if default_idx is not None and 0 <= default_idx < len(qids):
        return qids[default_idx]
    return None


def budget_from_item(item: Dict[str, Any]) -> Optional[int]:
    budget_keys = (
        "budget",
        "allocated_budget",
        "allocation",
        "allocated",
        "retrieval_budget",
        "assigned_budget",
        "budget_units",
        "cost_budget",
    )
    for key in budget_keys:
        if key in item:
            return coerce_budget(item.get(key))
    return None


def reason_from_item(item: Dict[str, Any]) -> str:
    reason = first_present(item, ("reason", "rationale", "justification", "why"))
    return "" if reason is None else str(reason)


def collect_allocations(value: Any, qids: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    budgets: Dict[str, int] = {}
    reasons: Dict[str, str] = {}

    def record(qid: Optional[str], budget_value: Any, reason: str = "") -> None:
        if qid not in qids:
            return
        budgets[qid] = coerce_budget(budget_value)
        if reason:
            reasons[qid] = reason

    def walk(node: Any, default_idx: Optional[int] = None) -> None:
        if isinstance(node, list):
            for idx, item in enumerate(node):
                walk(item, idx)
            return

        if not isinstance(node, dict):
            if default_idx is not None and 0 <= default_idx < len(qids):
                record(qids[default_idx], node)
            return

        container_keys = (
            "allocations",
            "allocation",
            "budget_allocation",
            "budget_allocations",
            "budgets",
            "question_budgets",
            "questionBudgets",
            "results",
            "items",
            "questions",
            "data",
        )
        for key in container_keys:
            if key in node:
                walk(node.get(key), default_idx)

        for key, item in node.items():
            key_qid = str(key)
            if key_qid in qids:
                if isinstance(item, dict):
                    budget_value = budget_from_item(item)
                    if budget_value is None:
                        budget_value = item
                    record(key_qid, budget_value, reason_from_item(item))
                else:
                    record(key_qid, item)

        item_qid = qid_from_item(node, qids, default_idx)
        item_budget = budget_from_item(node)
        if item_qid is not None and item_budget is not None:
            record(item_qid, item_budget, reason_from_item(node))

    walk(value)
    return budgets, reasons


ALLOCATION_BEGIN = "BEGIN_BUDGET_ALLOCATION"
ALLOCATION_END = "END_BUDGET_ALLOCATION"


def strip_think_blocks(raw: str) -> str:
    text = "" if raw is None else str(raw)
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def allocation_text_blocks(raw: str) -> List[str]:
    text = strip_think_blocks(raw)
    pattern = re.compile(
        rf"{re.escape(ALLOCATION_BEGIN)}\s*([\s\S]*?)\s*{re.escape(ALLOCATION_END)}",
        re.IGNORECASE,
    )
    blocks = [m.group(1).strip() for m in pattern.finditer(text)]
    return blocks if blocks else [text]


def parse_text_allocator_output(raw: str, qids: List[str]) -> Tuple[Dict[str, int], Dict[str, str], bool]:
    budgets: Dict[str, int] = {}
    reasons: Dict[str, str] = {}
    qid_set = set(qids)

    def record(qid: Optional[str], budget_text: str, reason_text: str = "") -> None:
        if qid not in qid_set:
            return
        budgets[qid] = coerce_budget(budget_text)
        if reason_text:
            reasons[qid] = reason_text.strip(" \t|,-")

    for block in allocation_text_blocks(raw):
        for raw_line in block.splitlines():
            line = raw_line.strip().strip("|").strip()
            if not line:
                continue
            if line.lower().startswith(("question id", "id\t", "id |", "qid", "---")):
                continue
            if ALLOCATION_BEGIN.lower() in line.lower() or ALLOCATION_END.lower() in line.lower():
                continue

            recorded_from_qid = False
            for qid in qids:
                pos = line.find(qid)
                if pos < 0:
                    continue
                after_qid = line[pos + len(qid):]
                number = re.search(r"(?:budget\s*)?(?:=|:)?\s*(-?\d+(?:\.\d+)?)", after_qid, re.IGNORECASE)
                if number:
                    record(qid, number.group(1), after_qid[number.end():])
                    recorded_from_qid = True

            if recorded_from_qid:
                continue

            index_match = re.match(
                r"^\s*(?:[-*]\s*)?(?:Q(?:uestion)?\s*)?(\d+)\s*(?:[\).:\-|,\t]+|\s+)(.*)$",
                line,
                re.IGNORECASE,
            )
            if index_match:
                qid = qid_from_index(index_match.group(1), qids)
                rest = index_match.group(2)
                number = re.search(r"-?\d+(?:\.\d+)?", rest)
                if qid is not None and number:
                    record(qid, number.group(0), rest[number.end():])

    return budgets, reasons, bool(budgets)


def parse_allocator_output(raw: str, qids: List[str]) -> Tuple[Dict[str, int], Dict[str, str], bool]:
    text_budgets, text_reasons, text_ok = parse_text_allocator_output(raw, qids)
    if text_ok:
        return text_budgets, text_reasons, True

    best_budgets: Dict[str, int] = {}
    best_reasons: Dict[str, str] = {}
    best_score = (-1, -1)

    for value in json_candidates_from_text(raw):
        budgets, reasons = collect_allocations(value, qids)
        exact_matches = sum(1 for qid in qids if qid in budgets)
        score = (exact_matches, len(budgets))
        if score > best_score:
            best_score = score
            best_budgets = budgets
            best_reasons = reasons

    return best_budgets, best_reasons, bool(best_budgets)


def equal_allocation(qids: List[str], total_budget: int, min_budget: int, max_budget: Optional[int]) -> Dict[str, int]:
    validate_budget_bounds(len(qids), total_budget, min_budget, max_budget)
    budgets = {qid: min_budget for qid in qids}
    remaining = total_budget - min_budget * len(qids)
    while remaining > 0:
        changed = False
        for qid in qids:
            if max_budget is None or budgets[qid] < max_budget:
                budgets[qid] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("Cannot create equal allocation with the provided bounds")
    return budgets


def validate_budget_bounds(n_questions: int, total_budget: int, min_budget: int, max_budget: Optional[int]) -> None:
    if n_questions <= 0:
        return
    min_total = min_budget * n_questions
    if total_budget < min_total:
        raise ValueError(
            f"Total budget {total_budget} is smaller than min allocation {min_budget} * {n_questions}"
        )
    if max_budget is not None and total_budget > max_budget * n_questions:
        raise ValueError(
            f"Total budget {total_budget} is larger than max allocation {max_budget} * {n_questions}"
        )


def append_detail(details: Dict[str, str], qid: str, detail: str) -> None:
    current = details.get(qid, "")
    if current:
        details[qid] = current + "+" + detail
    else:
        details[qid] = detail


def normalize_allocations(
    raw_budgets: Dict[str, int],
    qids: List[str],
    total_budget: int,
    min_budget: int,
    max_budget: Optional[int],
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
    validate_budget_bounds(len(qids), total_budget, min_budget, max_budget)

    present_qids = [qid for qid in qids if qid in raw_budgets]
    missing_qids = [qid for qid in qids if qid not in raw_budgets]

    seed_budgets: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    details: Dict[str, str] = {}

    for qid in present_qids:
        raw_budget = coerce_budget(raw_budgets.get(qid), default=min_budget)
        budget = raw_budget
        detail_parts: List[str] = []

        if budget < min_budget:
            budget = min_budget
            detail_parts.append("clamped_to_min")

        if max_budget is not None and budget > max_budget:
            budget = max_budget
            detail_parts.append("clamped_to_max")

        seed_budgets[qid] = budget

        if detail_parts:
            sources[qid] = "LLM_output_modified"
            details[qid] = "+".join(detail_parts)
        else:
            sources[qid] = "LLM_output_pure"
            details[qid] = "explicit_unchanged"

    remaining_for_missing = total_budget - sum(seed_budgets.values())
    missing_can_fit = remaining_for_missing >= min_budget * len(missing_qids)
    if max_budget is not None:
        missing_can_fit = missing_can_fit and remaining_for_missing <= max_budget * len(missing_qids)

    if missing_qids and missing_can_fit:
        missing_allocations = equal_allocation(missing_qids, remaining_for_missing, min_budget, max_budget)
        seed_budgets.update(missing_allocations)

        for qid in missing_qids:
            sources[qid] = "passive"
            details[qid] = "missing_qid_equal_fill"

    elif missing_qids:
        seed_budgets = equal_allocation(qids, total_budget, min_budget, max_budget)

        for qid in qids:
            sources[qid] = "passive"
            details[qid] = "missing_qid_global_equal_init"

        for qid in present_qids:
            raw_budget = coerce_budget(raw_budgets.get(qid), default=seed_budgets[qid])
            budget = raw_budget
            detail_parts: List[str] = []

            if budget < min_budget:
                budget = min_budget
                detail_parts.append("clamped_to_min")

            if max_budget is not None and budget > max_budget:
                budget = max_budget
                detail_parts.append("clamped_to_max")

            seed_budgets[qid] = budget

            if detail_parts:
                sources[qid] = "LLM_output_modified"
                details[qid] = "+".join(detail_parts)
            else:
                sources[qid] = "LLM_output_pure"
                details[qid] = "explicit_unchanged_after_global_equal_init"

    budgets: Dict[str, int] = {qid: seed_budgets[qid] for qid in qids}
    order_values: Dict[str, int] = dict(budgets)

    index = {qid: i for i, qid in enumerate(qids)}

    def add_order() -> List[str]:
        return sorted(qids, key=lambda qid: (-order_values.get(qid, 0), index[qid]))

    def remove_order() -> List[str]:
        return sorted(qids, key=lambda qid: (-budgets[qid], index[qid]))

    diff = total_budget - sum(budgets.values())
    while diff > 0:
        changed = False
        for qid in add_order():
            if max_budget is None or budgets[qid] < max_budget:
                budgets[qid] += 1

                if sources.get(qid) == "LLM_output_pure":
                    sources[qid] = "LLM_output_modified"
                append_detail(details, qid, "increased_to_match_total")

                diff -= 1
                changed = True
                if diff == 0:
                    break
        if not changed:
            raise ValueError("Cannot increase allocations to match total budget with the provided bounds")

    while diff < 0:
        changed = False
        for qid in remove_order():
            if budgets[qid] > min_budget:
                budgets[qid] -= 1

                if sources.get(qid) == "LLM_output_pure":
                    sources[qid] = "LLM_output_modified"
                append_detail(details, qid, "decreased_to_match_total")

                diff += 1
                changed = True
                if diff == 0:
                    break
        if not changed:
            raise ValueError("Cannot reduce allocations to match total budget with the provided bounds")

    return budgets, sources, details


def allocate_batch_budgets(
    llm,
    tokenizer,
    batch: List[dict],
    total_budget: int,
    tier_costs: Dict[int, int],
    min_budget: int,
    max_budget: Optional[int],
    question_max_chars: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str], Dict[str, str], str, bool]:
    qids = [item["qid"] for item in batch]
    validate_budget_bounds(len(qids), total_budget, min_budget, max_budget)

    system, user = build_allocation_prompt(
        batch=batch,
        total_budget=total_budget,
        tier_costs=tier_costs,
        min_budget=min_budget,
        max_budget=max_budget,
        question_max_chars=question_max_chars,
    )
    raw = call_llm(
        llm=llm,
        tokenizer=tokenizer,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed_budgets, reasons, parsed_ok = parse_allocator_output(raw, qids)
    if not parsed_ok:
        fallback = equal_allocation(qids, total_budget, min_budget, max_budget)
        fallback_reasons = {qid: "fallback_equal_allocation_after_unparsed_allocator_output" for qid in qids}
        fallback_sources = {qid: "fallback" for qid in qids}
        fallback_details = {qid: "unparsed_allocator_output_equal_allocation" for qid in qids}
        return fallback, fallback_reasons, fallback_sources, fallback_details, raw, False

    allocations, allocation_sources, allocation_details = normalize_allocations(
        parsed_budgets, qids, total_budget, min_budget, max_budget
    )
    return allocations, reasons, allocation_sources, allocation_details, raw, True


def batch_total_budget(args, batch: List[dict]) -> int:
    if args.total_budget is not None:
        return int(args.total_budget)
    return int(len(batch) * args.budget_per_question)


def main():
    args = parse_args()
    tier_costs = parse_tier_costs(args.tier_costs)
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
        tier_costs=tier_costs,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for batch_id, batch in enumerate(tqdm(iter_batches(args.input_jsonl, args.batch_size), desc="Agent batches", unit="batch"), start=1):
            total_budget = batch_total_budget(args, batch)
            allocations, reasons, allocation_sources, allocation_details, allocation_raw, allocation_parsed = allocate_batch_budgets(
                llm=llm,
                tokenizer=tokenizer,
                batch=batch,
                total_budget=total_budget,
                tier_costs=tier_costs,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                question_max_chars=args.allocation_question_max_chars,
                temperature=args.allocation_temperature,
                max_tokens=args.allocation_max_tokens,
            )
            allocation_records = [
                {
                    "id": item["qid"],
                    "budget": int(allocations[item["qid"]]),
                    "reason": reasons.get(item["qid"], ""),
                    "source": allocation_sources.get(item["qid"], ""),
                    "detail": allocation_details.get(item["qid"], ""),
                }
                for item in batch
            ]

            for batch_index, item in enumerate(batch):
                qid = item["qid"]
                obj = item["obj"]
                allocated_budget = int(allocations[qid])
                result = agent.run_one(qid=qid, question=item["question"], budget=allocated_budget, store=store)
                result["prediction"] = extract_boxed_answer(result.get("prediction", ""))
                result["dataset_name"] = obj.get("dataset_name") or default_dataset_name
                result["ground_truth"] = extract_ground_truth(obj)
                result["question_raw"] = obj.get("question") or obj.get("input") or ""
                result["options"] = obj.get("options")
                result["allocated_budget"] = allocated_budget
                result["batch_id"] = batch_id
                result["batch_index"] = batch_index
                result["batch_size"] = len(batch)
                result["batch_total_budget"] = total_budget
                result["batch_allocations"] = allocation_records
                result["budget_allocation_reason"] = reasons.get(qid, "")
                result["budget_allocation_source"] = allocation_sources.get(qid, "")
                result["budget_allocation_detail"] = allocation_details.get(qid, "")
                result["budget_allocation_parsed"] = allocation_parsed
                if args.save_allocation_raw:
                    result["budget_allocation_raw"] = allocation_raw
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()


if __name__ == "__main__":
    main()