# Cross-Lingual LLM-Judge Transfer via Evaluation Decomposition

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-b31b1b)](https://aclanthology.org/2026.emnlp-main.1742/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.18557-b31b1b)](https://arxiv.org/abs/2603.18557)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](#license)

This repository contains the code for the EMNLP 2026 paper [*Cross-Lingual LLM-Judge Transfer via Evaluation Decomposition*](https://aclanthology.org/2026.emnlp-main.1742/) by Ivaxi Sheth, Zeno Jonke, Amin Mantrach and Saab Mansour.

As large language models are increasingly deployed across diverse real-world applications, extending automated evaluation beyond English has become a critical challenge. Existing evaluation approaches are predominantly English-focused, and adapting them to other languages is hindered by the scarcity and cost of human-annotated judgments in most languages. We introduce a decomposition-based evaluation framework built around a Universal Criteria Set (UCS). UCS consists of a shared, language-agnostic set of evaluation dimensions, producing an interpretable intermediate representation that supports cross-lingual transfer with minimal supervision. Experiments on multiple faithfulness tasks across languages and model backbones demonstrate consistent improvements over strong baselines without requiring target-language annotations.

> **Release note.** This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.

## Method Overview

The UCS framework operates in three stages:

1. **Concept Generation** — LLM generates language-agnostic verification concepts
2. **Question Generation** — For each concept, targeted evaluation questions are generated
3. **Question Application** — Questions are applied to samples, producing per-concept scores

The resulting concept-level scores form a latent representation that transfers across languages via a lightweight neural classifier.

## Installation

```bash
git clone https://github.com/amazon-science/cross-lingual-judge-transfer.git
cd cross-lingual-judge-transfer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- AWS credentials configured for Bedrock access

## Data

### MEMERAG

Downloads MEMERAG data automatically from GitHub:

```bash
python setup_data.py memerag
```

This downloads all language files (en, de, fr, es, hi) to `data/memerag_ext_w_majority_vote/`.

**Manual alternative:** Download from https://github.com/amazon-science/MEMERAG/tree/main/data/memerag_ext_w_majority_vote

### mFACE

mFACE requires manual download, then automated processing:

1. Download the mFACE human-judgment spreadsheet linked from: https://github.com/google-research/google-research/tree/master/mface
2. Save it as CSV at: `data/mface/mface_annotated_data.csv`
3. Run the processing script:

```bash
python setup_data.py mface
```

This processes the CSV and creates per-language JSONL files in `data/mface/`:
- Languages: english, amharic, burmese, french, swahili, thai, arabic, hindi, spanish
- Filters out samples where `q2_avg == 0.5` (ambiguous labels)

## Quick Start

### MEMERAG Dataset

```bash
# Generate UCS scores for English
python evaluate.py --dataset memerag --lang en --prompts memerag_factuality --model qwen235b --num_proc 30

# Generate UCS scores for German
python evaluate.py --dataset memerag --lang de --prompts memerag_factuality --model qwen235b --num_proc 30

# Cross-lingual transfer (train on EN, evaluate on DE)
python transfer.py --dataset memerag --model qwen235b --source en --targets de
```

### mFACE Dataset

```bash
# Generate UCS scores for English
python evaluate.py --dataset mface --lang english --prompts mface_factuality --model qwen235b --num_proc 30

# Generate UCS scores for Amharic
python evaluate.py --dataset mface --lang amharic --prompts mface_factuality --model qwen235b --num_proc 30

# Cross-lingual transfer (train on English, evaluate on Amharic)
python transfer.py --dataset mface --model qwen235b --source english --targets amharic
```

> **Note:** If using AWS profiles and a specific Bedrock region, add `--aws_profile` and `--bedrock_region` arguments:
> ```bash
> python evaluate.py --dataset memerag --lang en --prompts memerag_factuality --model qwen235b --num_proc 30 --aws_profile prod --bedrock_region us-west-2
> ```

## Usage

### Step 1: Generate UCS Scores

Run evaluation on a dataset to generate per-concept scores:

```bash
python evaluate.py \
    --dataset memerag \
    --lang en \
    --model qwen235b \
    --prompts memerag_factuality \
    --num_proc 4
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | - | Dataset to evaluate (`memerag`, `mface`) |
| `--lang` | `en` | Language code (memerag: en, de, fr, es, hi; mface: english, amharic, burmese, french, swahili, thai, arabic, hindi, spanish) |
| `--model` | `qwen235b` | LLM model (see Supported Models) |
| `--prompts` | (required) | Prompt file name without extension (e.g., `memerag_factuality`, `mface_factuality`) |
| `--output_dir` | `output` | Output directory |
| `--num_proc` | `1` | Number of parallel workers |
| `--temperature` | `0.0` | LLM sampling temperature |
| `--top_p` | `1.0` | LLM nucleus sampling parameter |
| `--num_retries` | `5` | Retries per LLM call on throttling |
| `--seed` | `0` | Random seed for sample selection |
| `--max_samples` | None | Limit number of samples (for testing) |
| `--rerun` | False | Overwrite existing results |
| `--aws_profile` | None | AWS profile name |
| `--bedrock_region` | `us-east-1` | AWS Bedrock region |

### Step 2: Cross-lingual Transfer

Train on source language, evaluate on targets:

```bash
python transfer.py \
    --dataset memerag \
    --model qwen235b \
    --source en \
    --targets de fr es hi
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `memerag` | Dataset name |
| `--model` | `qwen235b` | Model used for evaluation (same as in `evaluate.py`) |
| `--source` | `en` | Source language for training |
| `--targets` | MEMERAG languages (`en de fr es hi`) | Target languages for evaluation |
| `--output_dir` | `output` | Directory with the scores from `evaluate.py` (same as in `evaluate.py`) |

## Project Structure

```
├── evaluate.py          # Main evaluation script
├── setup_data.py        # Data setup script
├── ucs.py               # UCS evaluation module
├── llm.py               # Bedrock LLM interface
├── transfer.py          # Cross-lingual transfer module
├── requirements.txt     # Python dependencies
├── prompts/             # Prompt templates
│   ├── memerag_factuality.json
│   └── mface_factuality.json
├── data/                # Input datasets
│   ├── memerag_ext_w_majority_vote/
│   │   ├── en.jsonl
│   │   ├── de.jsonl
│   │   └── ...
│   └── mface/
│       ├── english.jsonl
│       ├── amharic.jsonl
│       └── ...
└── output/              # Generated results
    └── {dataset}/{model}/
        ├── checklist.json   # Cached UCS checklist, shared by all languages
        └── {lang}/
            └── scores.csv   # Evaluation results
```

## Supported Models

Any AWS Bedrock model can be used. Pre-configured model shortcuts are defined in `llm.py`:

```bash
# List available model shortcuts
python -c "from llm import MODELS; print(list(MODELS.keys()))"
```

Default model: `qwen235b`

**Adding new models:** Edit the `MODELS` dictionary in `llm.py` with the Bedrock model ID and appropriate response extractor function.

## Citation

This work was published at EMNLP 2026 ([paper](https://aclanthology.org/2026.emnlp-main.1742/), [arXiv](https://arxiv.org/abs/2603.18557)). If you use UCS in your work, please cite:

```bibtex
@inproceedings{sheth-etal-2026-cross,
    title = "Cross-Lingual {LLM}-Judge Transfer via Evaluation Decomposition",
    author = "Sheth, Ivaxi and Jonke, Zeno and Mantrach, Amin and Mansour, Saab",
    editor = "Lee, Hung-Yi and Dev, Sunipa and Li, Junyi Jessy and Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1742/",
    doi = "10.18653/2026.emnlp-main.1742",
    pages = "39727--39745",
    ISBN = "979-8-89176-459-0",
}
```

## License

Released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
You are free to share and adapt the code for non-commercial purposes, provided appropriate credit is given.
See [LICENSE](LICENSE) for the full text.

## Contact

For research inquiries, please contact the paper authors. Pull requests are not accepted: this repository is maintained as a frozen research artifact matching the published paper.
