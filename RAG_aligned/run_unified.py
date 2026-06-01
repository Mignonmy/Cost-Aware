import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

from unified_runner_core import UnifiedRunner, parse_budgets


def main():
    ap = argparse.ArgumentParser(description="Unified runner for RAG_aligned methods")
    ap.add_argument(
        "--method",
        required=True,
        choices=[
            "relevance_only",
            "greedy_cost_aware",
            "knapsack",
            "redundancy_aware",
            "mmr",
            "topk",
            "no_passage",
        ],
    )
    ap.add_argument("--input", required=True, help="Input jsonl file")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mapping_file", default="corpus/meta_data/docid_tier_mapping_addtextbooks.json", help="docid_tier_mapping.json for budgeted methods")

    ap.add_argument("--model_path", default="pretrained_models/Qwen3-8B")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gen_tokens", type=int, default=512)
    ap.add_argument("--safe_margin", type=int, default=50)
    ap.add_argument(
        "--context_token_margin",
        type=int,
        default=1000,
        help="Reserve this many tokens from safe prompt budget; remaining used for context passages (like SAFE_PROMPT_LIMIT-1000 in original script).",
    )
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--budgets", default="0,5,10,15,20,25,30,inf")
    ap.add_argument("--max_candidates", type=int, default=30)
    ap.add_argument("--topk_list", default="10,20,30")
    ap.add_argument("--generation_batch_size", type=int, default=32)
    ap.add_argument("--tier_costs", default="0:0,1:1,2:4", help="Comma list like 0:0,1:1,2:4 for tier costs")

    ap.add_argument("--embedding_model_path", default="pretrained_models/Qwen3-Embedding-0.6B")
    ap.add_argument("--embedding_device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--embedding_batch_size", type=int, default=4)
    ap.add_argument("--embedding_max_chars", type=int, default=2000)
    ap.add_argument("--embedding_max_seq_length", type=int, default=512)

    ap.add_argument("--redundancy_lambda", type=float, default=0.2)
    ap.add_argument("--mmr_lambda", type=float, default=0.7)
    ap.add_argument("--mmr_gamma", type=float, default=0.1)

    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
    topk_list = [int(x.strip()) for x in args.topk_list.split(",") if x.strip()]

    emb = None
    if args.method in {"redundancy_aware", "mmr"}:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for redundancy_aware/mmr")
        if not args.embedding_model_path:
            raise ValueError("--embedding_model_path is required for redundancy_aware/mmr")
        if args.embedding_device == "auto":
            dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            dev = args.embedding_device
        emb = SentenceTransformer(args.embedding_model_path, device=dev)
        emb.max_seq_length = int(args.embedding_max_seq_length)

    runner = UnifiedRunner(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        gen_tokens=args.gen_tokens,
        safe_margin=args.safe_margin,
        context_token_margin=args.context_token_margin,
        temperature=args.temperature,
    )

    runner.run(
        method=args.method,
        input_jsonl=args.input,
        output_dir=args.output_dir,
        mapping_file=args.mapping_file,
        budgets=budgets,
        max_candidates=args.max_candidates,
        topk_list=topk_list,
        generation_batch_size=args.generation_batch_size,
        embedding_model=emb,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_chars=args.embedding_max_chars,
        redundancy_lambda=args.redundancy_lambda,
        mmr_lambda=args.mmr_lambda,
        mmr_gamma=args.mmr_gamma,
        tier_costs=args.tier_costs,
    )


if __name__ == "__main__":
    main()
