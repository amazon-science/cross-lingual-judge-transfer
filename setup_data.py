#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Setup datasets for the UCS evaluation framework."""

import argparse
import csv
import json
import urllib.request
import ssl
from pathlib import Path

# MEMERAG configuration
MEMERAG_BASE_URL = "https://raw.githubusercontent.com/amazon-science/MEMERAG/main/data/memerag_ext_w_majority_vote"
MEMERAG_LANGUAGES = ["en", "de", "fr", "es", "hi"]
MEMERAG_OUTPUT_DIR = "data/memerag_ext_w_majority_vote"

# mFACE configuration
MFACE_LANGUAGES = ['english', 'amharic', 'burmese', 'french', 'swahili', 'thai', 'arabic', 'hindi', 'spanish']
MFACE_INPUT_FILE = "data/mface/mface_annotated_data.csv"
MFACE_OUTPUT_DIR = "data/mface"


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output path."""
    try:
        # Create SSL context that doesn't verify certificates (for GitHub raw)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        print(f"  Downloading: {output_path.name}...", end=" ")
        with urllib.request.urlopen(url, context=ctx) as response:
            content = response.read()
            output_path.write_bytes(content)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def setup_memerag():
    """Download MEMERAG dataset files."""
    print("\n=== Setting up MEMERAG Dataset ===\n")
    
    # Create output directory
    output_dir = Path(__file__).parent / MEMERAG_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Download each language file
    success_count = 0
    for lang in MEMERAG_LANGUAGES:
        url = f"{MEMERAG_BASE_URL}/{lang}.jsonl"
        output_path = output_dir / f"{lang}.jsonl"
        
        if download_file(url, output_path):
            success_count += 1
    
    print(f"\nDownloaded {success_count}/{len(MEMERAG_LANGUAGES)} files")
    
    if success_count == len(MEMERAG_LANGUAGES):
        print("✓ MEMERAG setup complete!")
        return True
    else:
        print("⚠ Some files failed to download. Please check the URLs or try again.")
        return False


def setup_mface():
    """Process mFACE CSV into per-language JSONL files."""
    print("\n=== Setting up mFACE Dataset ===\n")
    
    base_path = Path(__file__).parent
    input_path = base_path / MFACE_INPUT_FILE
    output_dir = base_path / MFACE_OUTPUT_DIR
    
    # Check if input file exists
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        print("\nPlease download the mFACE data first:")
        print("  1. Go to: https://github.com/google-research/google-research/tree/master/mface")
        print(f"  2. Place the file at: {input_path}")
        print("  3. Run this command again")
        return False
    
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and process CSV
    language_data = {lang: [] for lang in MFACE_LANGUAGES}
    total_rows = 0
    filtered_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            lang = row['lang'].lower()
            
            # Skip languages we don't need
            if lang not in MFACE_LANGUAGES:
                continue
            
            # Skip rows where q2_avg == 0.5
            try:
                q2_avg = float(row['q2_avg'])
            except (ValueError, KeyError):
                continue
            
            if q2_avg == 0.5:
                filtered_rows += 1
                continue
            
            # Transform to target JSON structure
            record = {
                "query_id": f"mface_{row['id']}",
                "query": "Is the summary faithful to the source document?",
                "context": [{"text": row['input']}],
                "answer": [{
                    "sentence": row['summary'],
                    "sentence_id": 0,
                    "factuality": "Supported" if q2_avg > 0.5 else "Not Supported"
                }]
            }
            
            language_data[lang].append(record)
    
    print(f"Total rows in CSV: {total_rows}")
    print(f"Filtered (q2_avg=0.5): {filtered_rows}")
    print()
    
    # Write per-language JSONL files
    success_count = 0
    for lang in MFACE_LANGUAGES:
        records = language_data[lang]
        if not records:
            print(f"  {lang}: 0 samples (skipped)")
            continue
        
        output_path = output_dir / f"{lang}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"  {lang}: {len(records)} samples ✓")
        success_count += 1
    
    print(f"\nProcessed {success_count} language files")
    print("✓ mFACE setup complete!")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup datasets for UCS evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_data.py memerag    # Download MEMERAG dataset
  python setup_data.py mface      # Process mFACE CSV to JSONL
        """
    )
    parser.add_argument(
        'dataset',
        choices=['memerag', 'mface'],
        help='Dataset to setup: memerag (downloads from GitHub) or mface (processes local CSV)'
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("UCS Evaluation Framework - Data Setup")
    print("=" * 50)
    
    if args.dataset == 'memerag':
        success = setup_memerag()
    else:
        success = setup_mface()
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print(f"✓ {args.dataset.upper()} is ready to use!")
    else:
        print(f"✗ {args.dataset.upper()} setup failed. See errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()