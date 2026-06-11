# Turkish Extended Named Entity Recognition (ENER)

This repository contains a Turkish Extended Named Entity Recognition (ENER) workflow for corpus analysis, quality control, model training, and comparative evaluation. It supports CoNLL-style annotation parsing, corpus statistics, visualization, BERT-based token classification, transformer embedding extraction, a CRF baseline, and a Common Vector Approach (CVA) classifier.

## Project Overview

The project is organized around two complementary goals:

1. Characterize the annotated corpus through statistics, tagset checks, and visual summaries.
2. Compare a fine-tuned BERT NER model against a CRF baseline and a CVA-based similarity classifier using shared BIO-aligned evaluation with 4-fold cross-validation.

## Key Features

- CoNLL annotation parsing for INCEpTION-style document folders.
- Corpus statistics for tokens, sentences, entity density, label counts, and entity-type distribution.
- Tagset quality control against `data/ENER-tagset.tsv`.
- 4-fold cross-validation for both BERT and CRF, with per-fold precision/recall/F1 and mean±std aggregates.
- BERT-based token classification with first-subtoken alignment.
- CRF baseline with the same fold splits for a fair comparison.
- Transformer embedding extraction for CVA class-vector construction.
- CVA similarity classification using cosine similarity over class vectors.
- Context-only vs CVA vs Combined classification experiments.
- OOV entity experiment with top-3 nearest-label retrieval.
- PCA/t-SNE/UMAP visualizations with prototype overlay and explained variance.
- Top-N frequent-label confusion matrices (in addition to full confusion matrices).
- Thesis-ready Markdown and LaTeX table generator covering all experiments.

## Project Structure

```text
turkish-extended-ner/
├── data/
│   ├── annotation/             # CoNLL annotation folders
│   ├── ENER-tagset.tsv         # Canonical tagset used for QC
│   ├── _smoke_train.conll      # Small training sample for quick runs
│   ├── _smoke_eval.conll       # Small evaluation sample for quick runs
│   ├── full_train.conll        # Full training split
│   └── full_eval.conll         # Full evaluation split
├── ner_stats/                  # Core utilities
│   ├── cva.py                  # CVA class vectors and cosine similarity (top-k)
│   ├── data_utils.py           # CoNLL parsing, BIO validation, label maps
│   ├── embeddings.py           # TransformerEmbedder (word-aligned extraction)
│   ├── embedding_analysis.py   # Embedding aggregation and class vector building
│   ├── evaluation.py           # Token-level precision/recall/F1/confusion
│   ├── evaluation_report.py    # Report generation (JSON, CSV, PNG, top-N confusion)
│   ├── statistics.py           # Corpus-level statistics
│   ├── visualization.py        # PCA, t-SNE, UMAP, prototype overlay plots
│   └── ...                     # tagset, spans, char_features, contrastive, timing
├── scripts/
│   ├── train_ner.py                  # End-to-end BERT + CVA pipeline (single run)
│   ├── run_cross_validation.py       # 4-fold CV for BERT
│   ├── run_crf_baseline.py           # 4-fold CV for CRF baseline
│   ├── run_embedding_analysis.py     # Embedding extraction, CVA, PCA/t-SNE/UMAP
│   ├── compare_context_cva.py        # Context-only vs CVA vs Combined experiment
│   ├── oov_experiment.py             # OOV entity top-3 retrieval experiment
│   ├── generate_prototype_visuals.py # Prototype PCA overlay and nearest-neighbor listing
│   ├── generate_thesis_summary.py    # Thesis-ready Markdown + LaTeX report generator
│   ├── run_analysis.py               # Corpus analysis CLI
│   ├── run_char_ner.py               # Character-level CNN NER experiment
│   ├── run_contrastive_ner.py        # Contrastive learning NER experiment
│   └── run_multi_seed.py             # Multi-seed evaluation with statistical tests
├── results/                    # Generated experiment outputs
├── outputs/                    # Model checkpoints
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.8 or newer
- Git

### Setup

#### Windows PowerShell

```powershell
git clone https://github.com/MoussaBane/turkish-extended-ner.git
cd turkish-extended-ner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux or macOS

```bash
git clone https://github.com/MoussaBane/turkish-extended-ner.git
cd turkish-extended-ner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Corpus analysis

Run the analysis pipeline to compute corpus statistics, validate the tagset, and generate plots:

```bash
python scripts/run_analysis.py --data-root data/annotation --results-dir results
```

Expected outputs: `results/stats.json`, `results/label_counts.csv`, `results/type_counts.csv`, `results/plots/*.png`.

### 2. End-to-end BERT + CVA experiment (single run)

```bash
python scripts/train_ner.py \
    --train-file data/full_train.conll \
    --eval-file data/full_eval.conll \
    --output-dir outputs/bert-ner-full \
    --results-dir results/model_comparison_full
```

Expected outputs: `metrics_bert_with_o.json`, `metrics_cva_with_o.json`, `comparison_summary.json`, confusion matrices.

Quick smoke test:

```bash
python scripts/train_ner.py \
    --train-file data/_smoke_train.conll \
    --eval-file data/_smoke_eval.conll \
    --output-dir outputs/bert-ner-smoke \
    --results-dir results/model_comparison_smoke
```

### 3. 4-fold cross-validation (BERT)

```bash
python scripts/run_cross_validation.py \
    --data-file data/full_train.conll \
    --model-name dbmdz/bert-base-turkish-cased \
    --output-dir results/cv_full
```

Outputs per-fold metrics under `results/cv_full/fold_N/` and aggregated `cv_summary.json`.

### 4. CRF baseline (4-fold)

```bash
python scripts/run_crf_baseline.py \
    --data-file data/full_train.conll \
    --output-dir results/crf_full
```

Outputs the same per-fold structure and `crf_cv_summary.json`.

### 5. Embedding extraction, CVA, and visualization

```bash
python scripts/run_embedding_analysis.py \
    --train data/full_train.conll \
    --eval data/full_eval.conll \
    --model-name dbmdz/bert-base-turkish-cased \
    --output-dir results/embedding_full \
    --use-cva
```

Outputs: `train_embeddings.{csv,npy}`, `eval_embeddings.{csv,npy}`, `class_vectors.json`, PCA/t-SNE/UMAP plots, `pca_explained_variance.txt`, `classification_report.json`.

### 6. Context-only vs CVA vs Combined comparison

```bash
python scripts/compare_context_cva.py \
    --train-file data/full_train.conll \
    --eval-file data/full_eval.conll \
    --output-dir results/compare_full
```

Outputs sub-directories `context_only/`, `cva_only/`, `combined/` each with a `classification_report.json`, plus `comparison_summary.json`.

### 7. OOV entity experiment (top-3 retrieval)

```bash
python scripts/oov_experiment.py \
    --train-file data/full_train.conll \
    --eval-file data/full_eval.conll \
    --output-dir results/oov_full
```

Outputs `oov_results.csv` (token, gold label, top-1, top-3 predictions with scores) and `oov_summary.json`.

### 8. Prototype visualization and nearest-neighbor listing

```bash
python scripts/generate_prototype_visuals.py --out-dir results/embedding_full
```

Outputs `prototypes_pca.png` and `prototype_neighbors.json` (top-10 nearest tokens per class prototype).

### 9. Thesis-ready report (Markdown + LaTeX)

```bash
python scripts/generate_thesis_summary.py \
    --cv-dir results/cv_full \
    --crf-dir results/crf_full \
    --compare-dir results/compare_full \
    --bert-cva-dir results/model_comparison_full_cleaned \
    --oov-dir results/oov_full \
    --embed-dir results/embedding_full \
    --out-dir results/thesis_report
```

Outputs `thesis_report.md` and `thesis_report.tex` with all result tables.

## Pipeline Explanation

1. Read CoNLL BIO annotations from the train and evaluation files.
2. Validate BIO labels and normalize label sequences.
3. Build the label map shared by BERT, CRF, and CVA.
4. Tokenize sentences and align labels to the first subtoken of each word.
5. Fine-tune a transformer token-classification model, or load an existing checkpoint when requested.
6. Extract token embeddings from the same transformer backbone.
7. Build CVA class vectors (mean or common vectors) from aligned training embeddings.
8. Predict evaluation labels with BERT, CRF, and CVA.
9. Compute precision, recall, F1, accuracy, and confusion matrices (full and top-N frequent labels).
10. Save metrics, warnings, timing summaries, and CSV/JSON outputs for later reporting.

## Methods

### BERT-based NER

The BERT pipeline fine-tunes a transformer model for token classification using first-subtoken alignment. Default training settings:

- Model: `dbmdz/bert-base-turkish-cased`
- Learning rate: `2e-5`
- Batch size: `8`
- Epochs: `3`
- Seed: `42`

### CRF Baseline

A linear-chain CRF with handcrafted lexical features (word form, capitalization, digit, context window). Trained on the same 4-fold splits as BERT for a fair comparison.

### Common Vector Approach (CVA)

CVA uses the transformer backbone as a shared embedding extractor. Training embeddings are grouped by BIO label. For each class, the SVD of the centered embedding matrix is used to project the class mean onto the orthogonal complement of the within-class variance subspace, producing a stable class prototype. Evaluation embeddings are classified by cosine similarity to these prototypes.

## Evaluation

All experiments report precision, recall, and F1-score at the macro level, together with per-class breakdowns. Confusion matrices are saved for the full label set and for the top-N most frequent labels separately. 4-fold cross-validation results include per-fold values and mean±std aggregates.

## Results Summary

Full-data experiment outcomes:

| Experiment | Metric | Value |
| --- | --- | --- |
| BERT 4-fold CV | Accuracy (mean ± std) | 0.7824 ± 0.0074 |
| BERT 4-fold CV | Macro F1 (mean ± std) | 0.0340 ± 0.0121 |
| CRF 4-fold CV | Macro F1 (mean ± std) | 0.3138 ± 0.0211 |
| OOV experiment | OOV entity tokens | 235 |
| Context-only | Accuracy | 0.3294 |
| CVA-only | Accuracy | 0.0167 |
| Combined | Accuracy | 0.0302 |

Corpus statistics (from `results_example/stats.json`): 130 document folders, 34 annotated, 1,142 sentences, 29,195 tokens, entity-sentence ratio 0.8581, 161 BIO labels, 97 entity types.

## Scientific Contribution

This repository contributes a reusable experimental pipeline for Turkish ENER research: corpus analysis, BERT and CRF training, CVA similarity classification, OOV/top-k retrieval, and a thesis-ready report generator — all operating on the same BIO-aligned evaluation protocol.

## Future Work

- Document dataset provenance, split policy, and annotation guidelines.
- Add a license file before public release.
- Expand character-level and contrastive learning experiments with systematic hyperparameter search.

## License

No license file is currently included in the repository.
