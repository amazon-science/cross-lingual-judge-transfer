#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Cross-lingual judge transfer with UCS using NN."""

import json
import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def extract_concept_features(results_df, invert_scores=True):
    """Extract concept-level scores from universal questions results.
    
    Args:
        results_df: DataFrame with raw_llm_output column
        invert_scores: If True, invert scores (10-score) so higher = Not Supported
    """
    all_concept_avg_scores = []
    
    def average(numbers, default_avg=0.0):
        valid = [x for x in numbers if x is not None]
        return round(sum(valid) / len(valid), 1) if valid else default_avg

    for _, row in results_df.iterrows():
        concept_scores = row['concept_scores']
        if isinstance(concept_scores, str):
            try:
                concept_scores = json.loads(concept_scores)
            except json.JSONDecodeError:
                concept_scores = ast.literal_eval(concept_scores)

        concept_avg_scores = {concept: average(scores, 5.0) for concept, scores in concept_scores.items()}
        if invert_scores:
            concept_avg_scores = {concept: 10.0 - score for concept, score in concept_avg_scores.items()}
        all_concept_avg_scores.append(concept_avg_scores)
    
    return pd.DataFrame(all_concept_avg_scores)

LANGS = ['en', 'de', 'fr', 'es', 'hi']


def load_data(path: Path):
    """Load CSV and extract features and labels."""
    df = pd.read_csv(path)
    features = extract_concept_features(df, invert_scores=True)
    labels = (df['factuality'] == 'Not Supported').astype(int).values
    return features.values, labels


def get_file_path(base_path: Path, dataset: str, model: str, lang: str) -> Path:
    """Construct data file path using new structure: output/{dataset}/{model}/{lang}/scores.csv"""
    return base_path / dataset / model / lang / "scores.csv"


def run_transfer(dataset: str, model: str, source: str, targets: list = None, output_dir: str = "output"):
    """Train NN on source language, evaluate on target languages."""
    base_path = Path(__file__).parent / output_dir
    langs = [source] + [t for t in (targets or LANGS) if t != source]
    
    # Load and train on source
    src_file = get_file_path(base_path, dataset, model, source)
    if not src_file.exists():
        print(f"File not found: {src_file}")
        return None
    
    X_src, y_src = load_data(src_file)
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src)
    
    m = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=42)
    m.fit(X_src_scaled, y_src)
    
    # Evaluate all languages
    results = []
    for lang in langs:
        path = get_file_path(base_path, dataset, model, lang)
        if not path.exists():
            continue
        
        X, y = load_data(path)
        pred = m.predict(scaler.transform(X))
        results.append({
            'lang': lang.upper() + ('*' if lang == source else ''),
            'ba': balanced_accuracy_score(y, pred),
            'kappa': cohen_kappa_score(y, pred),
            'n': len(y)
        })
    
    # Print results
    print(f"\n{'Lang':<8} {'BA':>7} {'Kappa':>7} {'N':>5}")
    print("-" * 30)
    for r in results:
        print(f"{r['lang']:<8} {r['ba']:>7.4f} {r['kappa']:>7.4f} {r['n']:>5}")
    
    test = [r for r in results if '*' not in r['lang']]
    if test:
        print("-" * 30)
        print(f"{'AVG':<8} {np.mean([r['ba'] for r in test]):>7.4f} {np.mean([r['kappa'] for r in test]):>7.4f}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='memerag')
    p.add_argument('--model', default='qwen235b')
    p.add_argument('--source', default='en')
    p.add_argument('--targets', nargs='+')
    p.add_argument('--output_dir', default='output')
    args = p.parse_args()
    
    run_transfer(args.dataset, args.model, args.source, args.targets, args.output_dir)
