set -e  


echo "===== Hotpotqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/new_datasets/mini_datasets/hotpotqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/hotpotqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized/hotpotqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \

echo "===== NQ Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/new_datasets/mini_datasets/nq_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/nq_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized/nq_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192

echo "===== Triviaqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/new_datasets/mini_datasets/triviaqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/triviaqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized/triviaqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192

echo "===== MMLU Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/new_datasets/mini_datasets/mmlu_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/mmlu_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized/mmlu_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192

echo "===== Medqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/new_datasets/mini_datasets/medqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/medqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized/medqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192