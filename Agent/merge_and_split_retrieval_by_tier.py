import argparse
import json
import os
from typing import Dict, List, Tuple


def load_docid_to_tier(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {str(k): int(v) for k, v in m.items()}


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def get_results(obj: dict) -> List[dict]:
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


def merge_results(a: List[dict], b: List[dict]) -> List[dict]:
    """Merge two retrieval lists.

    If the same docid appears in both lists, keep the higher score.
    """
    by_id: Dict[str, dict] = {}
    for p in list(a) + list(b):
        docid = get_docid(p)
        if not docid:
            continue
        score = get_score(p)
        if docid not in by_id or score > get_score(by_id[docid]):
            pp = dict(p)
            pp["docid"] = docid
            by_id[docid] = pp
    merged = list(by_id.values())
    merged.sort(key=get_score, reverse=True)
    return merged


def split_by_tier(results: List[dict], mapping: dict, default_tier: int) -> Dict[str, List[dict]]:
    buckets = {"0": [], "1": [], "2": []}
    for p in results:
        docid = get_docid(p)
        if not docid:
            continue
        tier = int(mapping.get(docid, default_tier))
        tier = tier if tier in {0, 1, 2} else default_tier
        pp = dict(p)
        pp["docid"] = docid
        pp["tier"] = tier
        buckets[str(tier)].append(pp)

    for k in ["0", "1", "2"]:
        buckets[k].sort(key=get_score, reverse=True)
    return buckets


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge two retrieved jsonl files (e.g., msmarco + textbooks) per query id, "
            "then split merged results into tiers (0/1/2) and sort within tier by score." 
        )
    )
    ap.add_argument("--msmarco_jsonl", required=True, help="Retrieved jsonl from msmarco")
    ap.add_argument("--textbooks_jsonl", required=True, help="Retrieved jsonl from textbooks")
    ap.add_argument("--docid_tier_mapping", required=True, help="docid_tier_mapping.json")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--default_tier", type=int, default=2)
    args = ap.parse_args()

    mapping = load_docid_to_tier(args.docid_tier_mapping)

    # Load msmarco side keyed by qid
    left: Dict[str, dict] = {}
    for line_no, obj in iter_jsonl(args.msmarco_jsonl):
        qid = obj.get("id")
        if qid is None:
            raise KeyError(f"Missing id at line {line_no} in {args.msmarco_jsonl}")
        left[str(qid)] = obj

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    written = 0
    missing_left = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, obj_r in iter_jsonl(args.textbooks_jsonl):
            qid = obj_r.get("id")
            if qid is None:
                raise KeyError(f"Missing id at line {line_no} in {args.textbooks_jsonl}")
            qid = str(qid)
            obj_l = left.get(qid)
            if obj_l is None:
                missing_left += 1
                obj_l = {}

            res_l = get_results(obj_l)
            res_r = get_results(obj_r)
            merged = merge_results(res_l, res_r)
            buckets = split_by_tier(merged, mapping, int(args.default_tier))

            out = dict(obj_r)  # keep right side fields by default
            out["retrieval_merged"] = {
                "sources": {"msmarco": len(res_l), "textbooks": len(res_r)},
                "results": merged,
            }
            out["retrieval_by_tier"] = buckets
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved: {args.output_jsonl}")
    print(f"Wrote queries: {written}")
    if missing_left:
        print(f"Warning: {missing_left} queries missing in msmarco_jsonl (used textbooks only)")


if __name__ == "__main__":
    main()
