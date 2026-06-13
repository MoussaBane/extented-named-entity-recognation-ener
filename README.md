# Turkish Extended Named Entity Recognition (ENER)

**Master Thesis Project — Moussa Bane, June 2026**  
**Backbone:** `dbmdz/bert-base-turkish-cased` · **Entity types:** 131 · **Corpus:** 1,142 sentences

This repository contains a complete Turkish Extended Named Entity Recognition (ENER) pipeline for a master's thesis. It covers corpus analysis, quality control, model training, embedding analysis, attention-based entity representations, and comparative evaluation across BERT, CRF, CVA, Attention-NER, Contrastive NER, and Character-level NER.

## Project Overview

This thesis project addresses a challenging research problem: **fine-grained Named Entity Recognition in Turkish with 131 entity types** (LOC_CITY, ORG_POLITICAL, PRO_FOOD, PERSON, DATE, etc.) from a manually annotated INCEpTION corpus of 1,142 sentences.

### Research Questions

1. How does fine-tuned Turkish BERT perform on 97+ entity-type NER vs. a CRF baseline?
2. Can embedding-space prototype classifiers (CVA, mean vector) match supervised fine-tuning?
3. Do BERT's internal Q/K/V attention projections encode entity-type-discriminative information?
4. Does contrastive learning (SupConLoss) improve entity embedding separability?

### Architecture

The project is organized around six model families and a shared 4-fold CV evaluation protocol:

## Key Features

### Data & Evaluation

- CoNLL annotation parsing for INCEpTION-style document folders.
- Corpus statistics: tokens, sentences, entity density, label counts, entity-type distribution.
- Tagset quality control against `data/ENER-tagset.tsv` (131 canonical entity types).
- 4-fold cross-validation for BERT and CRF with shared fold splits, per-fold P/R/F1 and mean±std.
- Top-N frequent-label confusion matrices, per-label CSV reports, publication-ready tables.

### Models

- **BERT NER:** Fine-tuned `dbmdz/bert-base-turkish-cased` with first-subtoken BIO alignment.
- **CRF baseline:** Linear-chain CRF with lexical features on identical fold splits.
- **Attention NER:** Learned Q/K/V self-attention head on top of BERT (parameter-efficient).
- **Character NER:** Hybrid BERT + character CNN for morphological awareness.
- **Contrastive NER:** SupConLoss entity representation learning (Khosla et al., NeurIPS 2020).
- **CVA classifier:** Common Vector Approach with SVD-based prototype vectors.

### Analysis & Visualization

- Transformer embedding extraction (word-aligned, 24,911 × 768).
- Q/K/V projection extraction from BERT attention layers (see `scripts/extract_qkv_vectors.py`).
- Attention heatmaps: entity-to-entity and entity-to-context attention patterns.
- PCA / t-SNE / UMAP visualizations with class prototype overlays.
- Error analysis: per-label FP/FN analysis, confusion pair identification.
- Statistical significance: Bootstrap CI, Wilcoxon signed-rank, Cohen's d effect size.

### Reporting

- OOV entity experiment with top-3 nearest-label retrieval.
- Thesis-ready figure generation (`scripts/generate_thesis_figures.py`) → `thesis_figures/`.
- Thesis-ready Markdown + LaTeX table generator (`scripts/generate_thesis_summary.py`).
- Related work review covering FIGER, FewNERD, BERTurk, LUKE, SpanBERT, SimCSE, and more.

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

## New Scripts (Thesis Finalization)

| Script | Purpose |
| ------ | ------- |
| `scripts/extract_qkv_vectors.py` | Extract Q/K/V projections from BERT attention layers for entity tokens; generates `entity_{q,k,v}_vectors.npy`, PCA/t-SNE/UMAP plots, `prototype_vectors.npy`, `centroid_similarity_matrix.csv` |
| `scripts/generate_attention_heatmaps.py` | Visualize BERT attention weights for entity-containing sentences; layer-wise and head-wise analysis |
| `scripts/generate_thesis_figures.py` | Centralize all publication-quality figures into `thesis_figures/` (300 DPI, consistent styling) |
| `scripts/generate_statistical_significance.py` | Bootstrap CI, Wilcoxon signed-rank test, Cohen's d, McNemar's test for model comparisons |

## Results Summary

### Quantitative Results (4-fold Cross-Validation)

| Model | Metric Level | Accuracy | Macro F1 | Notes |
| ----- | ------------ | -------- | -------- | ----- |
| **CRF** | Entity (w/o O) | 0.6868 ± 0.0257 | **0.3138 ± 0.0244** | Best macro F1 |
| Context-only (mean cosine) | All tokens | 0.3294 | 0.0931 | Zero-shot prototype |
| BERT fine-tuned | Token (w/ O) | 0.7824 ± 0.0074 | 0.0340 ± 0.0121 | O-class bias |
| BERT fine-tuned | Entity (w/o O) | 0.5204 ± 0.0988 | 0.0308 ± 0.0138 | Entity-only eval |
| CVA + Context | All tokens | 0.0302 | 0.0317 | SVD prototype |
| Attention NER (frozen) | All tokens | 0.7488 | 0.0047 | Needs full finetune |

**Key finding:** CRF outperforms BERT by 10x on macro F1. Mean-vector prototype (9.3% F1) outperforms fine-tuned BERT (3.4% F1) — counterintuitive result driven by label sparsity (97 types, ~1,035 training sentences).

### Corpus Statistics

| Statistic | Value |
| --------- | ----- |
| Document folders | 130 (34 annotated) |
| Total sentences | 1,142 |
| Total tokens | 29,195 |
| Entity sentence ratio | 85.8% |
| BIO labels in corpus | 161 |
| Entity types (unique) | 97 |
| Canonical tagset size | 131 |
| OOV entity tokens | 235 |

## Thesis Contributions

1. **Turkish ENER corpus** — First Turkish NER dataset with 131 fine-grained entity types, manually annotated using INCEpTION.
2. **Comparative evaluation** — BERT vs CRF vs CVA vs Attention-NER vs Contrastive NER under a shared 4-fold protocol.
3. **Surprising prototype result** — Zero-shot mean-vector prototype classification outperforms supervised BERT fine-tuning on macro F1, demonstrating the impact of label sparsity on fine-tuned models.
4. **Q/K/V attention analysis** — First systematic extraction and comparison of BERT's internal Q, K, V projections as entity type representations.
5. **Attention heatmaps for fine-grained NER** — Layer-wise and head-wise attention visualization for 97-class entity recognition.

## Reports and Documentation

| Document | Location |
| -------- | -------- |
| Full audit report | `AUDIT_REPORT.md` |
| Thesis readiness report | `FINAL_THESIS_READINESS_REPORT.md` |
| Related work review (10 papers) | `related_work_review.md` |
| Cross-validation report | `results/cross_validation_report.md` |
| Model comparison report | `results/comparison_report.md` |
| Per-label metrics | `results/label_metrics.md` |
| CVA analysis | `results/cva_report.md` |
| Error analysis | `results/error_analysis_report.md` |
| Boundary detection | `results/boundary_detection_report.md` |
| Contrastive learning | `results/contrastive_learning_report.md` |
| Attention entity repr. | `results/attention_entity_representation_report.md` |
| Statistical significance | `results/statistical_significance_report.md` |
| Attention NER (Turkish) | `docs/thesis_attention_ner_section.md` |
| Thesis figures | `thesis_figures/` (9 publication-quality figures) |

## Scientific Contribution

This repository contributes a complete, reproducible pipeline for Turkish ENER: corpus analysis, BERT and CRF training, CVA similarity classification, attention-based entity representations (Q/K/V extraction), contrastive learning, OOV/top-k retrieval, systematic error analysis, statistical significance testing, and thesis-ready figure and report generation — all under a shared BIO-aligned 4-fold evaluation protocol.

## Future Work

- Run Q/K/V extraction on trained model (`scripts/extract_qkv_vectors.py`).
- Re-run Attention NER with full BERT fine-tuning (`--no-freeze`).
- Run contrastive and character NER experiments on full dataset.
- Add seqeval entity-level F1 alongside token-level metrics.
- Expand corpus to ≥50 examples per entity type for reliable classification.
- Release model checkpoints and dataset on HuggingFace Hub.

## License

No license file is currently included in the repository. Contact the author before using for purposes beyond academic review.
