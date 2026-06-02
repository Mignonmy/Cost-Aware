set -e  
set -o pipefail
output_base="Agent/result_batch_qwen"
mkdir -p "${output_base}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${output_base}/run_all_${timestamp}.log"
model_path="<output_path>"

{
echo "===== Hotpotqa Dataset ====="
python Agent/run_agent_batch.py \
  --input_jsonl test_dataset/mini_datasets/hotpotqa_mini.jsonl \
  --tiered_retrieval_jsonl Agent/tier_split_result/hotpotqa_mini_tier_split.jsonl \
  --output_jsonl "${output_base}/hotpotqa_mini_output.jsonl" \
  --batch_size 5 \
  --budget_per_question 20 \
  --max_steps 30 \
  --max_tokens 8192 \
  --allocation_max_tokens 8192 \
  --model_path $model_path \
  --save_allocation_raw

echo "===== NQ Dataset ====="
python Agent/run_agent_batch.py \
  --input_jsonl test_dataset/mini_datasets/nq_mini.jsonl \
  --tiered_retrieval_jsonl Agent/tier_split_result/nq_mini_tier_split.jsonl \
  --output_jsonl "${output_base}/nq_mini_output.jsonl" \
  --batch_size 5 \
  --budget_per_question 20 \
  --max_steps 30 \
  --max_tokens 8192 \
  --allocation_max_tokens 8192 \
  --model_path $model_path \
  --save_allocation_raw

echo "===== TriviaQA Dataset ====="
python Agent/run_agent_batch.py \
  --input_jsonl test_dataset/mini_datasets/triviaqa_mini.jsonl \
  --tiered_retrieval_jsonl Agent/tier_split_result/triviaqa_mini_tier_split.jsonl \
  --output_jsonl "${output_base}/triviaqa_mini_output.jsonl" \
  --batch_size 5 \
  --budget_per_question 20 \
  --max_steps 30 \
  --max_tokens 8192 \
  --allocation_max_tokens 8192 \
  --model_path $model_path \
  --save_allocation_raw


echo "===== MMLU Dataset ====="
python Agent/run_agent_batch.py \
  --input_jsonl test_dataset/mini_datasets/mmlu_mini.jsonl \
  --tiered_retrieval_jsonl Agent/tier_split_result/mmlu_mini_tier_split.jsonl \
  --output_jsonl "${output_base}/mmlu_mini_output.jsonl" \
  --batch_size 5 \
  --budget_per_question 20 \
  --max_steps 30 \
  --max_tokens 8192 \
  --allocation_max_tokens 8192 \
  --model_path $model_path \
  --save_allocation_raw


echo "===== Medqa Dataset ====="
python Agent/run_agent_batch.py \
  --input_jsonl test_dataset/mini_datasets/medqa_mini.jsonl \
  --tiered_retrieval_jsonl Agent/tier_split_result/medqa_mini_tier_split.jsonl \
  --output_jsonl "${output_base}/medqa_mini_output.jsonl" \
  --batch_size 5 \
  --budget_per_question 20 \
  --max_steps 30 \
  --max_tokens 8192 \
  --allocation_max_tokens 8192 \
  --model_path $model_path \
  --save_allocation_raw
} 2>&1 | tee -a "${log_file}"