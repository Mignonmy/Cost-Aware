import json
import os
import re

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from dataset_utils import (
    build_system_prompt,
    format_question_for_prompt,
    get_dataset_name,
    get_ground_truth,
    get_medmcqa_ground_truth_option,
    get_passage_docid,
    get_passage_score,
    is_medmcqa,
)

from unified_runner_methods import (
    parse_cost_map,
    build_items,
    greedy_pick,
    knapsack_pick,
    mmr_pick,
    redundancy_pick,
    similarity_matrix,
)


def extract_boxed_answer(text: str) -> str:
    m = re.search(r"\\boxed\s*\{([^}]*)\}", str(text), re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in str(text).split("\n") if l.strip()]
    return lines[-1] if lines else ""


def build_passage_text(p: dict) -> str:
    title = (p.get("title") or "").strip()
    seg = (p.get("contents") or p.get("text") or p.get("segment") or "").strip()
    if title and seg:
        return f"Title: {title}\n{seg}"
    return f"Title: {title}" if title else seg


def select_passages_under_token_budget(passages: list[dict], tokenizer, max_tokens: int) -> list[dict]:
    """
    Keep passages in order while fitting within a token budget.
    - Always respects token budget (even for tier0/zero-cost passages).
    - Does not attempt partial passage truncation; prompt-level truncation handles edge cases.
    """
    if max_tokens <= 0 or not passages:
        return []
    selected = []
    used = 0
    for p in passages:
        t = build_passage_text(p)
        n = len(tokenizer.encode(t))
        if used + n <= max_tokens:
            selected.append(p)
            used += n
    return selected


def get_passages(data: dict):
    r = data.get("retrieval", {})
    if isinstance(r, dict) and "results" in r:
        return r.get("results", [])
    return data.get("passage", [])


def parse_budgets(s: str):
    out = []
    for x in [p.strip() for p in s.split(",") if p.strip()]:
        out.append(float("inf") if x.lower() in {"inf", "infinite"} else float(int(float(x))))
    return out


def budget_label(b: float) -> str:
    return "infinite" if b == float("inf") else str(int(b))


def load_docid_to_tier(path: str):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {str(k): str(v) for k, v in m.items()}


def prompt_with_context(tokenizer, dataset_name, query, passages, safe_limit):
    """
    Try to include as many passages as possible under `safe_limit`.

    If even 1 passage makes the prompt too long, we truncate the final prompt instead of returning an empty
    prompt or dropping context entirely.
    """

    def build_prompt(sel):
        ctx = "\n\n".join([f"Doc {i + 1}:\n{build_passage_text(p)}" for i, p in enumerate(sel)])
        msgs = [
            {"role": "system", "content": build_system_prompt(True, dataset_name)},
            {"role": "user", "content": f"Context:\n{ctx}\n\n{query}"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    selected = list(passages)
    while True:
        prompt = build_prompt(selected)
        ids = tokenizer.encode(prompt)

        if len(ids) <= safe_limit:
            return prompt, selected

        if not selected:
            # No passages left but prompt still too long (extreme query). Truncate prompt.
            return tokenizer.decode(ids[:safe_limit], skip_special_tokens=False), []

        if len(selected) == 1:
            # Keep at least one passage; truncate the prompt rather than dropping context.
            return tokenizer.decode(ids[:safe_limit], skip_special_tokens=False), selected

        selected = selected[:-1]


def prompt_no_context(tokenizer, dataset_name, query, safe_limit):
    msgs = [
        {"role": "system", "content": build_system_prompt(False, dataset_name)},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt)
    if len(ids) > safe_limit:
        prompt = tokenizer.decode(ids[:safe_limit], skip_special_tokens=False)
    return prompt


class UnifiedRunner:
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        gen_tokens: int,
        safe_margin: int,
        context_token_margin: int = 1000,
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.safe_limit = int(max_model_len) - int(gen_tokens) - int(safe_margin)
        if self.safe_limit <= 0:
            raise ValueError(
                "Invalid prompt budget: safe_limit <= 0. "
                f"Got max_model_len={int(max_model_len)}, gen_tokens={int(gen_tokens)}, safe_margin={int(safe_margin)}."
            )
        # Token budget reserved for context passages before building full chat prompt.
        self.context_token_budget = max(1, self.safe_limit - int(context_token_margin))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
        )
        self.sampling_params = SamplingParams(temperature=float(temperature), max_tokens=int(gen_tokens))

    def run(
        self,
        method: str,
        input_jsonl: str,
        output_dir: str,
        mapping_file: str | None,
        budgets: list[float],
        max_candidates: int,
        topk_list: list[int],
        generation_batch_size: int,
        embedding_model=None,
        embedding_batch_size: int = 4,
        embedding_max_chars: int = 2000,
        redundancy_lambda: float = 0.2,
        mmr_lambda: float = 0.7,
        mmr_gamma: float = 0.1,
        tier_costs: str = "0:0,1:1,2:4",
    ):
        os.makedirs(output_dir, exist_ok=True)

        if method == "with_docids":
            method = "relevance_only"

        need_mapping = method in {"relevance_only", "greedy_cost_aware", "knapsack", "redundancy_aware", "mmr"}
        docid_to_tier = load_docid_to_tier(mapping_file) if (need_mapping and mapping_file) else {}

        cost_map = parse_cost_map(tier_costs)

        handlers = {}
        if method == "no_passage":
            handlers["no_passage"] = open(os.path.join(output_dir, "results_no_passage.jsonl"), "w", encoding="utf-8")
        elif method == "topk":
            for k in topk_list:
                handlers[k] = open(os.path.join(output_dir, f"results_topk_{k}.jsonl"), "w", encoding="utf-8")
        else:
            for b in budgets:
                lb = budget_label(b)
                handlers[lb] = open(os.path.join(output_dir, f"results_budget_{lb}_{method}.jsonl"), "w", encoding="utf-8")

        def flush(prompts, metas):
            if not prompts:
                return
            outs = self.llm.generate(prompts, self.sampling_params)
            for out, (key, meta) in zip(outs, metas):
                raw = out.outputs[0].text
                meta["prediction"] = extract_boxed_answer(raw)
                meta["raw_output"] = raw
                handlers[key].write(json.dumps(meta, ensure_ascii=False) + "\n")
            prompts.clear()
            metas.clear()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        prompts, metas = [], []

        with open(input_jsonl, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        for ln in lines:
            data = json.loads(ln)
            dataset_name = get_dataset_name(data, input_jsonl)
            query = format_question_for_prompt(data, dataset_name)
            gt = get_ground_truth(data, dataset_name)
            gt_opt = get_medmcqa_ground_truth_option(data) if is_medmcqa(data, dataset_name) else ""
            passages = get_passages(data)

            base = {
                "id": data.get("id"),
                "question": query,
                "dataset_name": dataset_name,
                "ground_truth": gt,
                "ground_truth_option": gt_opt,
            }

            if method == "no_passage":
                prompt = prompt_no_context(self.tokenizer, dataset_name, query, self.safe_limit)
                meta = {**base, "num_passages": 0, "used_docids": [], "used_titles": [], "actual_cost": 0.0}
                prompts.append(prompt)
                metas.append(("no_passage", meta))

            elif method == "topk":
                for k in topk_list:
                    raw_sel = select_passages_under_token_budget(
                        passages[:k],
                        self.tokenizer,
                        self.context_token_budget,
                    )
                    prompt, sel = prompt_with_context(self.tokenizer, dataset_name, query, raw_sel, self.safe_limit)
                    meta = {
                        **base,
                        "topk": k,
                        "num_passages": len(sel),
                        "used_docids": [str(get_passage_docid(p)) for p in sel],
                        "used_titles": [p.get("title", "") for p in sel],
                        "actual_cost": 0.0,
                    }
                    prompts.append(prompt)
                    metas.append((k, meta))

            else:
                if need_mapping and not docid_to_tier:
                    raise ValueError("mapping_file is required for this method")

                items = build_items(passages, docid_to_tier, get_passage_docid, get_passage_score, max_candidates, cost_map)
                sim = None
                if method in {"redundancy_aware", "mmr"}:
                    if embedding_model is None:
                        raise ValueError("embedding_model is required for redundancy_aware/mmr")
                    sim = similarity_matrix(items, embedding_model, build_passage_text, embedding_batch_size, embedding_max_chars)

                for b in budgets:
                    lb = budget_label(b)
                    if method == "relevance_only":
                        picked, spent = [], 0.0
                        for i, it in enumerate(items):
                            c = it["cost"]
                            if c == 0 or b == float("inf") or spent + c <= float(int(b)):
                                picked.append(i)
                                if c != 0 and b != float("inf"):
                                    spent += c
                    elif method == "greedy_cost_aware":
                        picked = greedy_pick(items, b)
                    elif method == "knapsack":
                        picked = knapsack_pick(items, b)
                    elif method == "redundancy_aware":
                        picked = redundancy_pick(items, sim, b, redundancy_lambda)
                    elif method == "mmr":
                        picked = mmr_pick(items, sim, b, mmr_lambda, mmr_gamma)
                    else:
                        raise ValueError(f"unknown method: {method}")

                    picked_docids = {items[i]["docid"] for i in picked}
                    raw_sel = [p for p in passages[:max_candidates] if str(get_passage_docid(p)) in picked_docids]
                    raw_sel = select_passages_under_token_budget(raw_sel, self.tokenizer, self.context_token_budget)
                    prompt, sel = prompt_with_context(self.tokenizer, dataset_name, query, raw_sel, self.safe_limit)

                    used = [str(get_passage_docid(p)) for p in sel]
                    used_set = set(used)
                    stats = [
                        {"docid": it["docid"], "title": it["title"], "relevance": it["rel"], "cost": it["cost"]}
                        for it in items
                        if it["docid"] in used_set
                    ]
                    actual_cost = float(sum(float(s["cost"]) for s in stats))

                    meta = {
                        **base,
                        "budget": lb,
                        "actual_cost": actual_cost,
                        "num_passages": len(sel),
                        "used_docids": used,
                        "used_titles": [p.get("title", "") for p in sel],
                        "selected_passage_stats": stats,
                        "all_candidate_stats": [
                            {"docid": it["docid"], "title": it["title"], "relevance": it["rel"], "cost": it["cost"]}
                            for it in items
                        ],
                    }
                    prompts.append(prompt)
                    metas.append((lb, meta))

            if len(prompts) >= int(generation_batch_size):
                flush(prompts, metas)

        flush(prompts, metas)

        for h in handlers.values():
            h.close()
