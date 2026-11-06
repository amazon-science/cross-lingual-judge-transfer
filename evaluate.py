# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

from llm import BedrockLLM
from ucs import UCS

# Dataset configurations
DATASETS = {
    "memerag": {
        "data_path": "data/memerag_ext_w_majority_vote/{lang}.jsonl",
        "context_col": "context",
        "answer_col": "sentence",
        "label_col": "factuality",
        "languages": ["en", "de", "fr", "es", "hi"],
    },
    "mface": {
        "data_path": "data/mface/{lang}.jsonl",
        "context_col": "context",
        "answer_col": "sentence",
        "label_col": "factuality",
        "languages": ['english', 'amharic', 'burmese', 'french', 'swahili', 'thai', 'arabic', 'hindi', 'spanish'],
    },
}


def load_prompts(prompts_file: str, base_path: Path) -> dict:
    """Load prompt templates from file."""
    prompts_path = base_path / "prompts" / f"{prompts_file}.json"
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts not found: {prompts_path}")
    
    with open(prompts_path) as f:
        return json.load(f)


def load_dataset(dataset: str, lang: str, base_path: Path) -> pd.DataFrame:
    """Load dataset from JSONL file and expand nested answer column."""
    config = DATASETS[dataset]
    data_path = base_path / config["data_path"].format(lang=lang)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(df)} rows (before expansion)")
    
    # Explode answer list into separate rows
    df = df.explode("answer")
    
    # Extract all keys from answer dicts as columns
    all_keys = set().union(*(d.keys() for d in df['answer']))
    for key in all_keys:
        df[key] = df['answer'].apply(lambda x: x.get(key, None))
    
    df = df.drop('answer', axis=1)
    print(f"Expanded to {len(df)} samples")
    return df

def main(args):
    # Setup paths
    base_path = Path(__file__).parent
    checklist_dir = base_path / args.output_dir / args.dataset / args.model
    checklist_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = checklist_dir / "checklist.json"
    results_dir = checklist_dir / args.lang
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "scores.csv"

    if results_path.exists() and not args.rerun:
        print(f"Results already exist: {results_path}")
        return
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Set random seeds to {args.seed}\n")

    # Load and prepare raw data
    raw_df = load_dataset(args.dataset, args.lang, base_path)
    
    # Apply max_samples if specified
    if args.max_samples and args.max_samples < len(raw_df):
        random.seed(args.seed)
        np.random.seed(args.seed)
        sampled_indices = sorted(random.sample(range(len(raw_df)), args.max_samples))
        raw_df = raw_df.iloc[sampled_indices].reset_index(drop=True)
        print(f"Sampled {len(raw_df)} samples using seed {args.seed}\n")
    
    # Create LLM
    llm = BedrockLLM(
        model=args.model,
        region=args.bedrock_region,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=2000,
        max_retries=args.num_retries,
        profile=args.aws_profile
    )
    
    # Load prompts and evaluate
    prompts = load_prompts(args.prompts, base_path)
    ucs = UCS(llm, prompts)
    results_df = ucs.evaluate(raw_df, checklist_cache_path=str(checklist_path), num_workers=args.num_proc)
    
    # Save results
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--dataset")
    parser.add_argument("--model", default="qwen235b")
    parser.add_argument("--lang", default='en')
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--bedrock_region", default="us-east-1")
    parser.add_argument("--aws_profile", default=None)
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--num_retries", default=5, type=int)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--prompts", required=True, help="Prompt file name without extension (e.g., memerag_ucs)")
    
    args = parser.parse_args()
    main(args)
