#!/bin/bash

set -e  

echo "===== Medqa Dataset ====="
echo "===== Running no_passage ====="
python RAG_aligned/run_unified.py \
  --method no_passage \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_no_passage

echo "===== Running topk ====="
python RAG_aligned/run_unified.py \
  --method topk \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_topk \
  --topk_list 5,10,15,20,25,30

echo "===== Running Relevance Only ====="
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_relevance_only

echo "===== Running Greedy Cost Aware ====="
python RAG_aligned/run_unified.py \
  --method greedy_cost_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_greedy_cost_aware

echo "===== Running Knapsack ====="
python RAG_aligned/run_unified.py \
  --method knapsack \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_knapsack

echo "===== Running Redundancy Aware ====="
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_redundancy_aware

echo "===== Running MMR ====="
python RAG_aligned/run_unified.py \
  --method mmr \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/medqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/medqa_mini/medqa_mini_mmr



echo "===== MMLU Dataset ====="
echo "===== Running no_passage ====="
python RAG_aligned/run_unified.py \
  --method no_passage \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_no_passage

echo "===== Running topk ====="
python RAG_aligned/run_unified.py \
  --method topk \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_topk \
  --topk_list 5,10,15,20,25,30

echo "===== Running Relevance Only ====="
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_relevance_only

echo "===== Running Greedy Cost Aware ====="
python RAG_aligned/run_unified.py \
  --method greedy_cost_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_greedy_cost_aware

echo "===== Running Knapsack ====="
python RAG_aligned/run_unified.py \
  --method knapsack \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_knapsack

echo "===== Running Redundancy Aware ====="
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_redundancy_aware

echo "===== Running MMR ====="
python RAG_aligned/run_unified.py \
  --method mmr \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/mmlu_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/mmlu_mini/mmlu_mini_mmr


echo "===== Hotpotqa Dataset ====="
echo "===== Running no_passage ====="
python RAG_aligned/run_unified.py \
  --method no_passage \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_no_passage

echo "===== Running topk ====="
python RAG_aligned/run_unified.py \
  --method topk \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_topk \
  --topk_list 5,10,15,20,25,30

echo "===== Running Relevance Only ====="
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_relevance_only

echo "===== Running Greedy Cost Aware ====="
python RAG_aligned/run_unified.py \
  --method greedy_cost_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_greedy_cost_aware

echo "===== Running Knapsack ====="
python RAG_aligned/run_unified.py \
  --method knapsack \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_knapsack

echo "===== Running Redundancy Aware ====="
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_redundancy_aware

echo "===== Running MMR ====="
python RAG_aligned/run_unified.py \
  --method mmr \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/hotpotqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/hotpotqa_mini/hotpotqa_mini_mmr


echo "===== NQ Dataset ====="
echo "===== Running no_passage ====="
python RAG_aligned/run_unified.py \
  --method no_passage \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_no_passage

echo "===== Running topk ====="
python RAG_aligned/run_unified.py \
  --method topk \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_topk \
  --topk_list 5,10,15,20,25,30

echo "===== Running Relevance Only ====="
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_relevance_only

echo "===== Running Greedy Cost Aware ====="
python RAG_aligned/run_unified.py \
  --method greedy_cost_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_greedy_cost_aware

echo "===== Running Knapsack ====="
python RAG_aligned/run_unified.py \
  --method knapsack \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_knapsack

echo "===== Running Redundancy Aware ====="
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_redundancy_aware

echo "===== Running MMR ====="
python RAG_aligned/run_unified.py \
  --method mmr \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/nq_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/nq_mini/nq_mini_mmr


echo "===== Triviaqa Dataset ====="
echo "===== Running no_passage ====="
python RAG_aligned/run_unified.py \
  --method no_passage \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_no_passage

echo "===== Running topk ====="
python RAG_aligned/run_unified.py \
  --method topk \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_topk \
  --topk_list 5,10,15,20,25,30

echo "===== Running Relevance Only ====="
python RAG_aligned/run_unified.py \
  --method relevance_only \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_relevance_only

echo "===== Running Greedy Cost Aware ====="
python RAG_aligned/run_unified.py \
  --method greedy_cost_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_greedy_cost_aware

echo "===== Running Knapsack ====="
python RAG_aligned/run_unified.py \
  --method knapsack \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_knapsack

echo "===== Running Redundancy Aware ====="
python RAG_aligned/run_unified.py \
  --method redundancy_aware \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_redundancy_aware

echo "===== Running MMR ====="
python RAG_aligned/run_unified.py \
  --method mmr \
  --model_path pretrained_models/Llama-3.1-8B-Instruct \
  --input test_dataset/retrieved_result_on_marco_textbooks/triviaqa_mini_retrieved.jsonl \
  --output_dir RAG_aligned/result_on_msmarco_textbooks/llama/triviaqa_mini/triviaqa_mini_mmr



echo "===== Done ====="