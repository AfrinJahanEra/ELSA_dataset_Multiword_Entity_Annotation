# ReSearch

A small Python toolkit for working with the ELSA Bangla review dataset and extracting multi-word product entities using BIO tagging.

## Project Overview

This repository contains two main scripts:

- `csv_to_json.py` — converts the raw ELSA CSV dataset into a JSON file.
- `elsa_multiword_entity_bio.py` — processes the ELSA dataset to detect product entities using a custom BIO tagging scheme, identify multi-word entities, and generate analysis reports.

The project is designed for Bangla (Bengali) product review data and focuses on extracting entities tagged with `_NE_` labels.

## Repository Structure

- `csv_to_json.py` — CSV to JSON conversion utility
- `elsa_multiword_entity_bio.py` — multi-word entity detection and report generation
- `datasets/ELSA_Dataset - ELSA_10K.csv` — source dataset
- `datasets/ELSA_Dataset - ELSA_10K.json` — generated JSON dataset from `csv_to_json.py`
- `result/elsa_bio_tagged_sentences.txt` — BIO-tagged sentences output
- `result/elsa_multiword_entity_report.txt` — full entity analysis report

## Requirements

- Python 3.7+
- `pandas`

Install dependencies with:

```bash
pip install pandas
```

## Usage

### 1. Convert CSV to JSON

```bash
python csv_to_json.py datasets/ELSA_Dataset - ELSA_10K.csv datasets/ELSA_Dataset - ELSA_10K.json
```

If run without arguments, it defaults to:

- Input: `datasets/ELSA_Dataset - ELSA_10K.csv`
- Output: `datasets/ELSA_Dataset - ELSA_10K.json`

### 2. Generate BIO tags and analysis report

```bash
python elsa_multiword_entity_bio.py
```

This script automatically finds the default dataset file in the local directory or `datasets/` folder and writes results to `result/`.

## What the analysis script does

- Loads the ELSA dataset from `ELSA_Dataset - ELSA_10K.csv`
- Detects entity tokens prefixed with `_NE_`
- Converts tagged reviews into BIO labels:
  - `B-PRODUCT` for the first token of an entity
  - `I-PRODUCT` for continuation tokens in multi-word entities
  - `O` for non-entity tokens
- Uses a configurable list of Bangla and English product continuation tokens to identify multi-word entities
- Collects statistics for sentiments, entity counts, and token distributions
- Writes:
  - `result/elsa_bio_tagged_sentences.txt`
  - `result/elsa_multiword_entity_report.txt`

## Output Files

- `result/elsa_bio_tagged_sentences.txt`
  - Contains sentence-level BIO tagging with tokens and assigned tags.
- `result/elsa_multiword_entity_report.txt`
  - Contains dataset statistics, entity summaries, sentiment breakdowns, examples, and top entity frequencies.

## Notes

- The script is tailored for Bangla review data and the ELSA dataset format.
- Multi-word entity detection is based on `_NE_` prefixes and a continuation-word dictionary for product-related terms.
- If the dataset file is not found, `elsa_multiword_entity_bio.py` will raise a clear error showing the expected filename.

## License

This repository does not include a license file. Add one if you want to publish or share the project more broadly.
