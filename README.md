# SwiftIOC-ML-Scoring-Engine

A practical machine learning pipeline that scores and prioritizes threat intelligence indicators collected from SwiftIOC.

The goal of this project is to turn raw IOC exports into analyst-friendly outputs by:

- loading and cleaning SwiftIOC data
- generating weak supervision labels
- building model features
- training a family-based scoring model
- scoring new indicators
- exporting ranked results for review

## What this project does

This repo takes a SwiftIOC CSV export and produces prioritized IOC outputs such as:

- `high_value`
- `medium_value`
- `low_value`

It uses a family-based approach instead of treating all indicators the same. This helps separate very different IOC types such as:

- `ipv4`
- `cve`
- `url`
- `sha256`

The production scoring flow also applies lightweight source-aware rules for known operational cases such as:

- `tor_exit_nodes`
- `feodo_ipblocklist`
- `openphish_feed`

## Project structure

```text
SwiftIOC-ML-Scoring-Engine/
├── data/
│   ├── raw/
│   ├── processed/
│   └── labeled/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── benchmark_production_model.py
│   ├── export_ranked_results.py
│   ├── features.py
│   ├── inspect_model.py
│   ├── label_data.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── run_pipeline.py
│   ├── score_iocs.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Data flow

```text
data/raw/latest.csv
        ↓
load_data.py
        ↓
data/processed/processed_iocs.csv
        ↓
label_data.py
        ↓
data/labeled/labeled_iocs.csv
        ↓
features.py
        ↓
data/processed/featured_iocs.csv
        ↓
train_model.py
        ↓
models/ioc_priority_family_bundle_v1.joblib
        ↓
score_iocs.py
        ↓
reports/scored_iocs_family_v4.csv
        ↓
export_ranked_results.py
        ↓
reports/exports/
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input format

Place the latest SwiftIOC export here:

```text
data/raw/latest.csv
```

The pipeline expects these core columns:

- `indicator`
- `type`
- `source`
- `first_seen`
- `last_seen`
- `confidence`
- `tags`
- `reference`
- `context`

## How to run

### Full production pipeline

```bash
python src/run_pipeline.py
```

### Start from scoring only

```bash
python src/run_pipeline.py --from-step score
```

### Export only

```bash
python src/run_pipeline.py --only export
```

### Benchmark only

```bash
python src/run_pipeline.py --only benchmark
```

### Show available steps

```bash
python src/run_pipeline.py --list-steps
```

## Pipeline steps

### 1. `load_data.py`

Loads the raw SwiftIOC CSV, validates required columns, applies preprocessing, and saves the cleaned dataset.

**Output:**
- `data/processed/processed_iocs.csv`

### 2. `label_data.py`

Applies weak scoring logic and creates training labels.

**Adds:**
- `weak_score`
- `label`

**Output:**
- `data/labeled/labeled_iocs.csv`

### 3. `features.py`

Builds model-ready features from the labeled dataset.

**Feature groups include:**
- indicator family
- text richness
- keyword hit counts
- time features
- confidence features
- family diagnostics

**Output:**
- `data/processed/featured_iocs.csv`

### 4. `train_model.py`

Trains the family-based production model bundle and saves summary files.

**Outputs:**
- `models/ioc_priority_family_bundle_v1.joblib`
- `reports/training_summary_family_v1.csv`
- `reports/training_summary_family_v1.txt`

### 5. `score_iocs.py`

Scores the latest IOC dataset using the trained family-based model and source-aware production rules.

**Output:**
- `reports/scored_iocs_family_v4.csv`

### 6. `export_ranked_results.py`

Creates analyst-facing CSV exports and a summary report.

**Outputs:**
- `reports/exports/all_ranked_iocs.csv`
- `reports/exports/high_priority_iocs.csv`
- `reports/exports/medium_priority_iocs.csv`
- `reports/exports/top_100_for_review.csv`
- `reports/exports/summary.txt`

## Benchmarking

Use the production benchmark script to evaluate how the scoring logic performs across held-out source groups:

```bash
python src/benchmark_production_model.py
```

Benchmarking is intentionally separate from the main production pipeline so the scoring model can be trained on the full labeled dataset while validation stays a distinct workflow.

## Current modeling approach

### Weak supervision

This repo currently uses rule-based weak labels instead of analyst-verified ground truth.

That means the project should be understood as a practical IOC prioritization engine, not a perfect research-grade classifier.

### Family-based scoring

Indicators are routed by family so the model does not try to learn CVEs, URLs, hashes, and IPv4s as if they were the same problem.

### Source-aware operational rules

A few targeted post-model rules are used where they improve operational usefulness, especially for known feed behavior.

Examples:
- pure `tor_exit_nodes` feed → `low_value`
- pure `feodo_ipblocklist` feed → `high_value`
- pure `openphish_feed` URL rows → `high_value`

## Main outputs

### Production model

```text
models/ioc_priority_family_bundle_v1.joblib
```

### Scored IOC results

```text
reports/scored_iocs_family_v4.csv
```

### Analyst-facing exports

```text
reports/exports/
```

## Suggested workflow

1. Export the latest SwiftIOC data to `data/raw/latest.csv`
2. Run the full pipeline
3. Review:
   - `reports/scored_iocs_family_v4.csv`
   - `reports/exports/high_priority_iocs.csv`
   - `reports/exports/top_100_for_review.csv`
   - `reports/exports/summary.txt`
4. Run the benchmark when validating changes to labels, features, scoring rules, or model logic

## Notes

- Labels in this project are generated from weak scoring logic, not analyst-reviewed truth.
- This makes the project highly useful for prioritization, triage, and operational ranking, but not a final authority on threat severity.
- The benchmark script should be used to validate current production scoring behavior across different source groups.
- Family diagnostics in `features.py` help catch type-vs-indicator mismatches, especially for hash data.

## Future improvements

Possible future improvements:

- analyst-reviewed training labels
- richer IPv4 enrichment features
- better family-specific models
- confidence calibration
- automated scheduled runs
- dashboard or UI for reviewing ranked IOCs
