set -e 


echo "===== Hotpotqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/mini_datasets/hotpotqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/hotpotqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized_llama/hotpotqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \
    --model_path pretrained_models/Llama-3.1-8B-Instruct

echo "===== NQ Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/mini_datasets/nq_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/nq_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized_llama/nq_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \
    --model_path pretrained_models/Llama-3.1-8B-Instruct

echo "===== Triviaqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/mini_datasets/triviaqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/triviaqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized_llama/triviaqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \
    --model_path pretrained_models/Llama-3.1-8B-Instruct

echo "===== MMLU Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/mini_datasets/mmlu_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/mmlu_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized_llama/mmlu_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \
    --model_path pretrained_models/Llama-3.1-8B-Instruct

echo "===== Medqa Dataset ====="
python Agent/run_agent.py \
    --input_jsonl test_dataset/mini_datasets/medqa_mini.jsonl \
    --tiered_retrieval_jsonl Agent/tier_split_result/medqa_mini_tier_split.jsonl \
    --output_jsonl Agent/result_optimized_llama/medqa_mini_output.jsonl \
    --max_steps 30 \
    --max_tokens 8192 \
    --model_path pretrained_models/Llama-3.1-8B-Instruct