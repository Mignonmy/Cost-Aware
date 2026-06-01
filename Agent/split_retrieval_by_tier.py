import argparse
import json
import os


def load_docid_to_tier(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {str(k): int(v) for k, v in m.items()}


def get_results(obj: dict) -> list:
    r = obj.get("retrieval", {})
    if isinstance(r, dict) and "results" in r:
        return r.get("results", [])
    return obj.get("passage", [])


def get_docid(p: dict) -> str:
    return str(p.get("docid") or p.get("id") or "")


def get_score(p: dict) -> float:
    try:
        return float(p.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Split retrieval results into tiers (0/1/2), sort within tier by score, and report tier distribution."
    )
    ap.add_argument("--input_jsonl", required=True, help="Retrieved dataset jsonl (has retrieval.results)")
    ap.add_argument("--docid_tier_mapping", required=True, help="docid_tier_mapping.json")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--default_tier", type=int, default=2)
    args = ap.parse_args()

    mapping = load_docid_to_tier(args.docid_tier_mapping)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    # Stats across all queries
    total_passages = 0
    tier_counts = {"0": 0, "1": 0, "2": 0}

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id")
            if qid is None:
                raise KeyError(f"Missing id at line {line_no}")

            results = get_results(obj)
            buckets = {"0": [], "1": [], "2": []}

            for p in results:
                docid = get_docid(p)
                if not docid:
                    continue
                tier = int(mapping.get(docid, int(args.default_tier)))
                tier = tier if tier in {0, 1, 2} else int(args.default_tier)
                pp = dict(p)
                pp["docid"] = docid
                pp["tier"] = tier
                k = str(tier)
                buckets[k].append(pp)
                tier_counts[k] += 1
                total_passages += 1

            for k in ["0", "1", "2"]:
                buckets[k].sort(key=get_score, reverse=True)

            out = dict(obj)
            out["retrieval_by_tier"] = buckets
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved tiered retrieval to: {args.output_jsonl}")
    print("\nTier distribution (all passages across all queries):")
    if total_passages == 0:
        print("- No passages found.")
        return
    for k in ["0", "1", "2"]:
        c = tier_counts[k]
        pct = c / total_passages * 100.0
        print(f"- tier{k}: {c} ({pct:.2f}%)")
    print(f"- total: {total_passages}")


if __name__ == "__main__":
    main()
