#!/usr/bin/env python3
"""
Randomly sample lines from a JSONL file with a fixed seed for reproducibility.
"""

import argparse
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample a JSONL file")
    parser.add_argument("--input_jsonl", type=str, default="test_dataset/hotpotqa-dev-kilt.jsonl", help="Input JSONL file path")
    parser.add_argument("--output_jsonl", type=str, default="test_dataset/mini_datasets/hotpotqa_mini.jsonl", help="Output sampled JSONL file path")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of lines to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def sample_jsonl(input_path: str, output_path: str, sample_size: int, seed: int):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read all lines
    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Check sample size
    if sample_size > len(lines):
        raise ValueError(f"Sample size {sample_size} exceeds total lines {len(lines)}")

    # Set random seed
    random.seed(seed)

    # Randomly sample lines
    sampled_lines = random.sample(lines, sample_size)

    # Write to output
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    print(f"Sampled {sample_size} lines from {input_path} -> {output_path} with seed={seed}")


def main():
    args = parse_args()
    sample_jsonl(args.input_jsonl, args.output_jsonl, args.sample_size, args.seed)


if __name__ == "__main__":
    main()