import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_TIER_LABELS = {0: "tier0", 1: "tier1", 2: "tier2"}
_TIER_DESCRIPTIONS = {
    0: "Open community resources (e.g., Wikipedia, public forums).",
    1: "General open-web content (blogs, tutorials, news sites).",
    2: (
        "Curated, official, or professional sources with moderate-to-high friction "
        "(official vendor docs and API references, support knowledge bases, standards or "
        "government technical docs, and restricted special-domain knowledge such as medical, "
        "biological, or financial sources)."
    ),
}


def _json_string_to_dict(value: Any, max_depth: int = 3) -> Optional[dict]:
    for _ in range(max_depth):
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _iter_json_like_candidates(s: str) -> List[str]:
    candidates = [s.strip()]
    candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE))

    for candidate in list(candidates):
        current = candidate
        for _ in range(3):
            # Some models return JSON that is escaped as plain text, e.g.
            # {\"action\":\"retrieve\"} or \{\"action\":\"retrieve\"\}.
            unescaped = re.sub(r"\\+([{}\"])", r"\1", current).strip()
            if unescaped == current:
                break
            candidates.append(unescaped)
            control_unescaped = unescaped.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
            if control_unescaped != unescaped:
                candidates.append(control_unescaped)
            current = unescaped

    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def safe_json_extract(text: str) -> Optional[dict]:
    if text is None:
        return None
    if isinstance(text, dict):
        return text

    decoder = json.JSONDecoder()
    matches: List[dict] = []

    for candidate in _iter_json_like_candidates(str(text)):
        parsed = _json_string_to_dict(candidate)
        if parsed is not None:
            matches.append(parsed)
            continue

        for start in [m.start() for m in re.finditer(r"\{", candidate)]:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            parsed = _json_string_to_dict(value)
            if parsed is not None:
                matches.append(parsed)

    if not matches:
        return None

    for obj in matches:
        if str(obj.get("action", "")).lower().strip() in {"answer", "retrieve"}:
            return obj
    return matches[0]


def extract_boxed_answer(text: str) -> str:
    s = str(text)
    m = re.search(r"(?:\\\\|\\)?boxed\s*\{([^}]*)\}", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in s.split("\n") if l.strip()]
    return lines[-1] if lines else ""


@dataclass
class Step:
    idx: int
    action: str  # decide, retrieve, answer
    spent: int
    remaining: int
    choice: Optional[dict]
    retrieved: Optional[dict]
    model_raw: str


class Trajectory:
    def __init__(self):
        self.steps: List[Step] = []

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def to_prompt_block(self) -> str:
        if not self.steps:
            return "(empty)"
        lines: List[str] = []
        for s in self.steps:
            lines.append(f"Step {s.idx} | action={s.action} | spent={s.spent} | remaining={s.remaining}")
            if s.choice is not None:
                lines.append(f"  choice: {json.dumps(s.choice, ensure_ascii=False)}")
            if s.retrieved is not None:
                rid = s.retrieved.get("docid") or s.retrieved.get("id")
                score = s.retrieved.get("score")
                tier = s.retrieved.get("tier")
                title = (s.retrieved.get("title") or "")[:120]
                lines.append(f"  retrieved: tier={tier} docid={rid} score={score} title={title}")
        return "\n".join(lines)


class KnowledgeStore:
    """Provide tiered passages per question id.

    Expected input jsonl per question:
    - id: str
    - retrieval_by_tier: {"0": [passages...], "1": [...], "2": [...]}
      where each passage has at least: docid/id, title/text/contents, score
    """

    def __init__(self, tiered_jsonl_path: str):
        self.path = tiered_jsonl_path
        self._by_qid: Dict[str, dict] = {}

    def load(self) -> None:
        by_qid: Dict[str, dict] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj.get("id") or obj.get("qid") or "")
                if not qid:
                    raise KeyError(f"Missing id at line {line_no} in {self.path}")
                by_qid[qid] = obj
        self._by_qid = by_qid

    def get_question(self, qid: str) -> dict:
        return self._by_qid[str(qid)]

    def pop_next(self, qid: str, tier: int) -> Optional[dict]:
        item = self.get_question(qid)
        tb = item.get("retrieval_by_tier") or {}
        key = str(int(tier))
        arr = tb.get(key) or []
        if not arr:
            return None
        p = arr.pop(0)
        p = dict(p)
        p["tier"] = int(tier)
        return p


def passage_to_text(p: dict) -> str:
    title = (p.get("title") or "").strip()
    seg = (p.get("contents") or p.get("text") or p.get("segment") or "").strip()
    if title and seg:
        return f"Title: {title}\n{seg}"
    if title:
        return f"Title: {title}"
    return seg


def build_tier_descriptions(costs: Dict[int, int]) -> str:
    parts = []
    for t in sorted(costs.keys()):
        desc = _TIER_DESCRIPTIONS.get(t, "")
        parts.append(f"- tier {t} ({_TIER_LABELS.get(t, str(t))}): cost={costs[t]} per passage | {desc}")
    return "\n".join(parts)


class BudgetedTierAgent:
    def __init__(
        self,
        llm,
        tokenizer,
        tier_costs: Dict[int, int],
        max_steps: int = 50,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.tier_costs = {int(k): int(v) for k, v in tier_costs.items()}
        self.max_steps = int(max_steps)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def _call_llm(self, system: str, user: str) -> str:
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
        out = self.llm.generate([prompt], sp)[0]
        return out.outputs[0].text

    def _generate_final_answer(self, question: str, ctx_block: str) -> tuple[str, str]:
        system = (
            "You are a QA answer generator. "
            "Use the provided evidence if available. "
            "Give only the final answer with no explanation. "
            "If the question is multiple-choice, output only the option label in \\boxed{}. "
            "If the question is open-domain QA, output only the short answer in \\boxed{}."
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{ctx_block if ctx_block else '(none)'}\n\n"
            # "Return only the final answer in \\boxed{}."
        )
        raw = self._call_llm(system, user)
        final = extract_boxed_answer(raw)
        return (final if final else raw.strip(), raw)

    def _finalize_with_answer(
        self,
        qid: str,
        question: str,
        budget: int,
        spent: int,
        contexts: List[dict],
        traj: Trajectory,
        step_idx: int,
        remaining: int,
        choice: Optional[dict],
        decision_raw: str,
    ) -> Dict[str, Any]:
        ctx_block = "\n\n".join([f"[Passage {i+1} | tier={p['tier']}]\n{passage_to_text(p)}" for i, p in enumerate(contexts)])
        final, answer_raw = self._generate_final_answer(question=question, ctx_block=ctx_block)
        traj.add(
            Step(
                idx=step_idx,
                action="answer",
                spent=spent,
                remaining=remaining,
                choice=choice,
                retrieved=None,
                model_raw=json.dumps(
                    {
                        "decision_raw": decision_raw,
                        "answer_raw": answer_raw,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        return {
            "id": qid,
            "question": question,
            "budget": budget,
            "spent": spent,
            "num_passages": len(contexts),
            "used_docids": [p.get("docid") or p.get("id") for p in contexts],
            "used_tiers": [p.get("tier") for p in contexts],
            "prediction": final,
            "trajectory": [s.__dict__ for s in traj.steps],
        }

    def run_one(self, qid: str, question: str, budget: int, store: KnowledgeStore) -> Dict[str, Any]:
        traj = Trajectory()
        spent = 0
        contexts: List[dict] = []

        tier_desc = build_tier_descriptions(self.tier_costs)

        for step_idx in range(self.max_steps):
            remaining = budget - spent

            ctx_block = "\n\n".join([f"[Passage {i+1} | tier={p['tier']}]\n{passage_to_text(p)}" for i, p in enumerate(contexts)])
            traj_block = traj.to_prompt_block()

            system = (
                "You are a budgeted retrieval-augmented QA agent. "
                "At each step, first decide whether the question requires external knowledge beyond the current context. "
                "If external knowledge is needed and budget allows, retrieve exactly ONE passage from the most appropriate tier. "
                "Actually, you can retrieve the passages from the same tier many times."
                "Only answer when the current context is sufficient, the question can be answered confidently without retrieval, "
                "or the remaining budget cannot support a useful retrieval. "
                "Do not answer merely because the current context is empty; at the beginning, empty context is normal. "
                "Follow the budget strictly. Output MUST be a single JSON object."
            )
            user = (
                f"Budget: {budget} | Spent: {spent} | Remaining: {remaining}\n\n"
                f"Question:\n{question}\n\n"
                f"Tier options:\n{tier_desc}\n\n"
                f"Trajectory so far:\n{traj_block}\n\n"
                f"Current context:\n{ctx_block if ctx_block else '(none)'}\n\n"
                "Decision objective:\n"
                "- First judge whether answering requires outside knowledge.\n"
                "- If outside knowledge is needed, retrieve one passage from the best tier.\n"
                "- Continue retrieving across steps until the evidence is sufficient, or the budget is exhausted.\n"
                "- If the question can already be answered confidently from the question itself or the accumulated context, answer directly.\n\n"
                "Decide the next action. Choose one:\n"
                "1) answer now\n"
                "2) retrieve one passage from a tier (0/1/2)\n\n"
                "Return JSON in one line with schema:\n"
                "{\"action\":\"answer\"|\"retrieve\", \"tier\":0|1|2|null, \"reason\":string}\n"
                "Rules:\n"
                "- If action=answer, set tier=null.\n"
                "- If action=retrieve, set tier to 0/1/2.\n"
                "- If Remaining < cost(tier), you MUST answer.\n"
                "- Do not answer just because Current context is empty.\n"
                "- If the information is insufficient, retrieve before answering.\n"
                "- Prefer the cheapest tier likely to be sufficient, but use higher tiers when the question likely needs official, professional, or restricted-domain knowledge.\n"
            )
            # system = (
            #     "You are a budgeted retrieval-augmented QA agent. "
            #     "At each step, first decide whether the question requires external knowledge beyond the current context. "
            #     "If external knowledge is needed and budget allows, retrieve exactly ONE passage from the most appropriate tier. "
            #     "Only answer when the current context is sufficient, the question can be answered confidently without retrieval, "
            #     "or the remaining budget cannot support a useful retrieval. "
            #     "Do not answer merely because the current context is empty; at the beginning, empty context is normal. "
            #     "Follow the budget strictly. Output MUST be a single JSON object."
            #     "When you give the final answer, be concise and directly answer the question (only choice optionor word answer)."
            # )

            # user = (
            #     f"Budget: {budget} | Spent: {spent} | Remaining: {remaining}\n\n"
            #     f"Question:\n{question}\n\n"
            #     f"Tier options:\n{tier_desc}\n\n"
            #     f"Trajectory so far:\n{traj_block}\n\n"
            #     f"Current context:\n{ctx_block if ctx_block else '(none)'}\n\n"
            #     "Decision objective:\n"
            #     "- First judge whether answering requires outside knowledge.\n"
            #     "- If outside knowledge is needed, retrieve one passage from the best tier.\n"
            #     "- Continue retrieving across steps until the evidence is sufficient, or the budget is exhausted.\n"
            #     "- If the question can already be answered confidently from the question itself or the accumulated context, answer directly.\n\n"
            #     "Decide the next action. Choose one:\n"
            #     "1) answer now\n"
            #     "2) retrieve one passage from a tier (0/1/2)\n\n"
            #     "Return JSON in one line with schema:\n"
            #     "{\"action\":\"answer\"|\"retrieve\", \"tier\":0|1|2|null, \"final\": string|null, \"reason\":string}\n"
            #     "Rules:\n"
            #     "- If action=answer, set final to the final answer string and tier=null.\n"
            #     "- If action=retrieve, set tier to 0/1/2 and final=null.\n"
            #     "- If Remaining < cost(tier), you MUST answer.\n"
            #     "- Do not answer just because Current context is empty.\n"
            #     "- If the question is multiple-choice and the information is insufficient, retrieve before answering.\n"
            #     "- Prefer the cheapest tier likely to be sufficient, but use higher tiers when the question likely needs official, professional, or restricted-domain knowledge.\n"
            # )

            raw = self._call_llm(system, user)
            choice = safe_json_extract(raw) or {}
            action = str(choice.get("action", "")).lower().strip()

            if action not in {"answer", "retrieve"}:
                # fallback: answer with empty
                action = "answer"
                choice = {"action": "answer", "tier": None, "reason": "unparsed"}

            if action == "retrieve":
                tier = choice.get("tier")
                try:
                    tier_i = int(tier)
                except (TypeError, ValueError):
                    tier_i = 2

                tier_i = 2 if tier_i not in {0, 1, 2} else tier_i
                cost = int(self.tier_costs.get(tier_i, 0))

                if remaining < cost:
                    action = "answer"
                    choice = {"action": "answer", "tier": None, "reason": "insufficient_budget"}

            if action == "answer":
                return self._finalize_with_answer(
                    qid=qid,
                    question=question,
                    budget=budget,
                    spent=spent,
                    contexts=contexts,
                    traj=traj,
                    step_idx=step_idx,
                    remaining=remaining,
                    choice=choice,
                    decision_raw=raw,
                )

            tier_i = int(choice.get("tier"))
            cost = int(self.tier_costs.get(tier_i, 0))
            retrieved = store.pop_next(qid, tier_i)
            if retrieved is None:
                # nothing left, must answer
                traj.add(
                    Step(
                        idx=step_idx,
                        action="retrieve",
                        spent=spent,
                        remaining=remaining,
                        choice={**choice, "note": "no_more_passages_in_tier"},
                        retrieved=None,
                        model_raw=raw,
                    )
                )
                return self._finalize_with_answer(
                    qid=qid,
                    question=question,
                    budget=budget,
                    spent=spent,
                    contexts=contexts,
                    traj=traj,
                    step_idx=step_idx,
                    remaining=remaining,
                    choice={"action": "answer", "tier": None, "reason": "no_more_passages_in_tier"},
                    decision_raw=raw,
                )

            spent += cost
            contexts.append(retrieved)

            traj.add(
                Step(
                    idx=step_idx,
                    action="retrieve",
                    spent=spent,
                    remaining=budget - spent,
                    choice=choice,
                    retrieved={
                        "tier": retrieved.get("tier"),
                        "docid": retrieved.get("docid") or retrieved.get("id"),
                        "score": retrieved.get("score"),
                        "title": retrieved.get("title", ""),
                    },
                    model_raw=raw,
                )
            )

            if spent >= budget:
                # force answer next
                continue

        return self._finalize_with_answer(
            qid=qid,
            question=question,
            budget=budget,
            spent=spent,
            contexts=contexts,
            traj=traj,
            step_idx=self.max_steps,
            remaining=budget - spent,
            choice={"action": "answer", "tier": None, "reason": "max_steps_reached"},
            decision_raw="",
        )
