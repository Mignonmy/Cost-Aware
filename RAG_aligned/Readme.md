# Run Command Handbook
# Inference
## 1) relevance_only, greedy_cost_aware, knapsack
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --input /path/to/pubmedqa_retrieved.jsonl \
  --output_dir /path/to/out/pubmedqa_relevance_only \
  --mapping_file /path/to/docid_tier_mapping.json \
  --model_path /path/to/Qwen3-8B

## 2) redundancy_aware / mmr_cost_aware（require embedding model）
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --input /path/to/bioasq_retrieved.jsonl \
  --output_dir /path/to/out/bioasq_redundancy \
  --mapping_file /path/to/docid_tier_mapping.json \
  --model_path /path/to/Qwen3-8B \
  --embedding_model_path /path/to/Qwen3-Embedding-0.6B
## 3) topk / no_passage（no need mapping file）
python RAG_aligned/run_unified.py \
  --method topk \
  --input /path/to/pubmedqa_retrieved.jsonl \
  --output_dir /path/to/out/pubmedqa_topk \
  --model_path /path/to/Qwen3-8B \
  --topk_list 10,20,30

# evaluate
python RAG_aligned/evaluate_unified.py \
  --result_dir RAG_aligned/result \
  --pattern "results*.jsonl" \
  --group_by dataset_setting_method

## 1) Evaluate all results in the given directory（budget + topk + no_passage）
python RAG_aligned/evaluate_unified.py \
  --result_dir /path/to/result_dir \
  --kind all \
  --group_by setting \
  --x_axis cost

## 2) Evaluate only the budget results, and group by different methods (budget results mean fixed evidence selection methods)
python .../evaluate_unified.py \
  --result_dir /path/to/result_dir \
  --kind budget \
  --group_by setting_method \
  --x_axis cost

## 3) Evaluate only topk
python .../evaluate_unified.py \
  --result_dir /path/to/result_dir \
  --kind topk \
  --group_by setting \
  --x_axis num_passages