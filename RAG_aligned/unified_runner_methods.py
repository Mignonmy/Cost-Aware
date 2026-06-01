import numpy as np


COST_MAP = {"0": 0, "1": 1, "2": 4}


def parse_cost_map(s: str) -> dict[str, int]:
    out = dict(COST_MAP)
    for part in [x.strip() for x in str(s).split(",") if x.strip()]:
        k, v = part.split(":", 1)
        out[str(int(k))] = int(v)
    return out


def build_items(passages, docid_to_tier, get_docid, get_score, max_candidates=30, cost_map=None):
    cost_map = COST_MAP if cost_map is None else {str(k): int(v) for k, v in cost_map.items()}
    items = []
    for p in passages[:max_candidates]:
        docid = str(get_docid(p))
        tier = str(docid_to_tier.get(docid, "2"))
        cost = int(cost_map.get(tier, cost_map.get("2", 4)))
        rel = float(get_score(p) or 0.0)
        items.append({"p": p, "docid": docid, "title": p.get("title", ""), "rel": rel, "cost": cost})
    return items


def greedy_pick(items, budget):
    scored = []
    for i, it in enumerate(items):
        ratio = float("inf") if it["cost"] == 0 else it["rel"] / it["cost"]
        scored.append((ratio, i))
    scored.sort(reverse=True, key=lambda x: x[0])
    picked, spent = [], 0.0
    for _, i in scored:
        c = items[i]["cost"]
        if c == 0 or budget == float("inf") or spent + c <= float(int(budget)):
            picked.append(i)
            if c != 0 and budget != float("inf"):
                spent += c
    return picked


def knapsack_pick(items, budget):
    zero = [i for i, it in enumerate(items) if it["cost"] == 0]
    pos = [(i, it) for i, it in enumerate(items) if it["cost"] > 0]

    if budget == float("inf"):
        return zero + [i for i, _ in pos]

    b = int(budget)
    n = len(pos)
    dp = [[0.0] * (b + 1) for _ in range(n + 1)]
    keep = [[False] * (b + 1) for _ in range(n + 1)]

    for r in range(1, n + 1):
        idx, it = pos[r - 1]
        v, c = float(it["rel"]), int(it["cost"])
        for w in range(b + 1):
            dp[r][w] = dp[r - 1][w]
            if c <= w:
                cand = dp[r - 1][w - c] + v
                if cand > dp[r][w]:
                    dp[r][w] = cand
                    keep[r][w] = True

    chosen, w = [], b
    for r in range(n, 0, -1):
        if keep[r][w]:
            idx, it = pos[r - 1]
            chosen.append(idx)
            w -= int(it["cost"])

    return zero + list(reversed(chosen))


def similarity_matrix(items, embedding_model, build_passage_text, batch_size=4, max_chars=2000):
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    texts = []
    for it in items:
        t = build_passage_text(it["p"])
        texts.append(t if max_chars <= 0 else t[:max_chars])
    embs = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return embs @ embs.T


def redundancy_pick(items, sim, budget, lam=0.2):
    picked = [i for i, it in enumerate(items) if it["cost"] == 0]
    picked_set = set(picked)
    remain = float("inf") if budget == float("inf") else float(int(budget))

    while True:
        best_i, best_pri = None, None
        for i, it in enumerate(items):
            if i in picked_set:
                continue
            if budget != float("inf") and it["cost"] > remain:
                continue
            penalty = float(sum(float(sim[i, j]) for j in picked))
            gain = float(it["rel"]) - lam * penalty
            pri = gain if it["cost"] == 0 else gain / it["cost"]
            if best_pri is None or pri > best_pri:
                best_i, best_pri = i, pri

        if best_i is None or (best_pri is not None and best_pri <= 0):
            break

        picked.append(best_i)
        picked_set.add(best_i)
        if budget != float("inf"):
            remain -= int(items[best_i]["cost"])

    return picked


def mmr_pick(items, sim, budget, mmr_lambda=0.7, gamma=0.1):
    picked, picked_set = [], set()
    remain = float("inf") if budget == float("inf") else float(int(budget))

    while True:
        best_i, best_score = None, None
        for i, it in enumerate(items):
            if i in picked_set:
                continue
            if budget != float("inf") and it["cost"] > remain:
                continue
            red = 0.0 if not picked else max(float(sim[i, j]) for j in picked)
            score = mmr_lambda * float(it["rel"]) - (1 - mmr_lambda) * red - gamma * float(it["cost"])
            if best_score is None or score > best_score:
                best_i, best_score = i, score

        if best_i is None or (best_score is not None and best_score <= 0):
            break

        picked.append(best_i)
        picked_set.add(best_i)
        if budget != float("inf"):
            remain -= int(items[best_i]["cost"])

    return picked
