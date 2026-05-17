# Turkish Extended Named Entity Recognition (ENER)

This repository contains a Turkish Extended Named Entity Recognition (ENER) workflow for corpus analysis, quality control, model training, and comparative evaluation. It supports CoNLL-style annotation parsing, corpus statistics, visualization, BERT-based token classification, transformer embedding extraction, and a Common Vector Approach (CVA) classifier for sentence-level inference.

## Project Overview

The project is organized around two complementary goals:

1. Characterize the annotated corpus through statistics, tagset checks, and visual summaries.
2. Compare a fine-tuned BERT NER model with a CVA-based similarity classifier using the same BIO-aligned evaluation setup.

The repository is designed to support reproducible analysis of Turkish ENER data while keeping the pipeline simple enough to run from the command line.

## Key Features

- CoNLL annotation parsing for INCEpTION-style document folders.
- Corpus statistics for tokens, sentences, entity density, label counts, and entity-type distribution.
- Tagset quality control against `data/ENER-tagset.tsv`.
- Visualizations for entity frequency, entity-to-sentence ratio, and sentence length distributions.
- BERT-based token classification with first-subtoken alignment.
- Transformer embedding extraction for CVA class-vector construction.
- CVA similarity classification using cosine similarity over class vectors.
- Shared evaluation for precision, recall, F1, confusion matrices, and inference timing.

## Project Structure

```text
extented-named-entity-recognation-ener/
├── data/
│   ├── annotation/             # CoNLL annotation folders
│   ├── ENER-tagset.tsv         # Canonical tagset used for QC
│   ├── _smoke_train.conll      # Small training sample for quick runs
│   ├── _smoke_eval.conll       # Small evaluation sample for quick runs
│   ├── full_train.conll        # Full training split
│   └── full_eval.conll         # Full evaluation split
├── ner_stats/                  # Corpus analysis, tagset, span, and evaluation utilities
├── scripts/
│   ├── run_analysis.py         # Corpus analysis CLI
│   └── train_ner.py            # End-to-end BERT + CVA pipeline
├── results/                    # Generated analysis and comparison outputs
├── results_example/            # Example summary outputs
├── outputs/                    # Model checkpoints and intermediate artifacts
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
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux or macOS

```bash
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
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

Expected outputs:

- `results/stats.json`
- `results/label_counts.csv`
- `results/type_counts.csv`
- `results/plots/*.png`
- `results/unused_tags_in_corpus.txt`
- `results/unknown_types_in_tagset_comparison.txt`

### 2. End-to-end BERT + CVA experiment

Run the NER comparison pipeline on the provided smoke split:

```bash
python scripts/train_ner.py --train-file data/_smoke_train.conll --eval-file data/_smoke_eval.conll --output-dir outputs/bert-ner-smoke --results-dir results/model_comparison_smoke
```

Expected outputs:

- `metrics_bert_with_o.json`
- `metrics_bert_without_o.json`
- `metrics_cva_with_o.json`
- `metrics_cva_without_o.json`
- `confusion_matrix_bert_with_o.csv`
- `confusion_matrix_cva_with_o.csv`
- `comparison_summary.json`
- `bio_validation_warnings.txt`

### 3. Reuse a saved model checkpoint

If `--skip-bert-train` is set and `--output-dir` already contains a Hugging Face checkpoint, the script reuses that checkpoint for evaluation:

```bash
python scripts/train_ner.py --train-file data/full_train.conll --eval-file data/full_eval.conll --output-dir outputs/bert-ner-full --results-dir results/model_comparison_full --skip-bert-train
```

## Embedding analysis and visualization

Use the provided embedding analysis pipeline to extract contextual embeddings, compute class vectors (mean or CVA), and produce PCA/t-SNE/UMAP visualizations suitable for thesis figures.

Run the pipeline:

```bash
python scripts/run_embedding_analysis.py --model-name dbmdz/bert-base-turkish-cased --output-dir results/embedding_analysis --use-cva
```

Outputs (saved under the `--output-dir`):

- `train_embeddings.csv`, `train_embeddings.npy` — token-level extracted embeddings for the training split
- `eval_embeddings.csv`, `eval_embeddings.npy` — token-level extracted embeddings for the evaluation split
- `class_vectors.json` — class mean or CVA common vectors
- `pca_embeddings.png`, `pca_embeddings_3d.png`, `tsne_embeddings.png`, `umap_embeddings.png` — visualizations
- `classification_report.json`, `confusion_matrix.png`, `metrics_summary.csv` — evaluation artifacts

See `notebooks/embedding_visualization.ipynb` for a notebook-based reproduction of the PCA steps and quick inspection.

If no loadable checkpoint exists in `--output-dir`, the script falls back to `--model-name`.

## Pipeline Explanation

1. Read CoNLL BIO annotations from the train and evaluation files.
2. Validate BIO labels and normalize label sequences.
3. Build the label map shared by BERT and CVA.
4. Tokenize sentences and align labels to the first subtoken of each word.
5. Fine-tune a transformer token-classification model, or load an existing checkpoint when requested.
6. Extract token embeddings from the same transformer backbone.
7. Build CVA class vectors from aligned training embeddings.
8. Predict evaluation labels with both BERT and CVA.
9. Compute precision, recall, F1, accuracy, and confusion matrices.
10. Measure sentence-level inference speed for BERT and CVA.
11. Save metrics, warnings, timing summaries, and CSV/JSON outputs for later reporting.

## Methods

### BERT-based NER

The BERT pipeline fine-tunes a transformer model for token classification using first-subtoken alignment. Only the first subtoken of each word contributes to the training labels, which keeps the token-level BIO supervision consistent with the underlying word segmentation.

Default training settings are exposed through command-line arguments, including model name, learning rate, batch size, number of epochs, maximum sequence length, and seed.

Example defaults used by the script:

- Model: `dbmdz/bert-base-turkish-cased`
- Learning rate: `2e-5`
- Train batch size: `8`
- Eval batch size: `8`
- Epochs: `3`
- Seed: `42`

### Common Vector Approach (CVA)

CVA uses transformer embeddings as a shared representation space. The pipeline:

1. Extracts embeddings for aligned words.
2. Aggregates training embeddings by BIO label.
3. Computes class vectors from those label groups.
4. Classifies each evaluation embedding by cosine similarity to the class vectors.

This provides a lightweight similarity-based alternative to the fine-tuned classifier and makes inference speed comparable on the same data split.

## Evaluation

The project reports the following evaluation artifacts:

- Precision, recall, and F1-score.
- Macro and per-class metrics, with and without the `O` label.
- Confusion matrices for BERT and CVA predictions.
- Sentence-level inference timing for both methods.

The main evaluation outputs are written to `results/model_comparison_*` as JSON and CSV files so that they can be reused in tables or figures for a paper.

### Confusion Matrix

Confusion matrices are exported as CSV files and preserve the label order used during evaluation. They are intended for error analysis, per-entity failure inspection, and publication figures.

### Speed Comparison

The pipeline measures average and total sentence inference time for both BERT and CVA. This is useful for comparing the practical cost of a fine-tuned classifier versus a similarity-based method.

## Results Summary

The repository includes example corpus statistics in `results_example/stats.json`.

Observed summary from the example results:

- Total document folders: 130
- Annotated documents: 34
- Unannotated documents: 96
- Total sentences: 1142
- Sentences with entity: 980
- Entity sentence ratio: 0.8581
- Total tokens: 29195
- BIO labels: 161
- Entity types: 97
- Average sentence length: 25.56 tokens

The repository also includes example CSV outputs for label and type counts in `results_example/`.

No paper-level model scores are reported in this README because the repository does not currently ship a finalized results table. Use the generated files under `results/model_comparison_*` to populate the final manuscript.

## Scientific Contribution

This repository contributes a reusable experimental pipeline for Turkish ENER research by combining corpus analysis, model training, and similarity-based classification in one codebase. It is useful for:

- documenting the structure of a custom extended NER tagset,
- validating annotation quality before modeling,
- benchmarking BERT against a CVA baseline under the same BIO evaluation protocol,
- and producing analysis artifacts that can be directly reused in a research paper.

## Future Work

- Add a finalized paper-ready results table with full-dataset scores.
- Document the dataset provenance, split policy, and annotation guidelines.
- Add multi-seed evaluation and significance testing.
- Expand the repository with character-level boundary detection experiments.
- Add contrastive learning and augmentation experiments mentioned in the project goals.
- Add a license file before public release.

## Manuscript-Style Methods and Results

### Methods

We developed an end-to-end Turkish Extended Named Entity Recognition (ENER) pipeline that combines corpus analysis, sequence labeling, and similarity-based classification. The input data are stored in CoNLL format and parsed from INCEpTION-style annotation folders. Before modeling, BIO labels are validated and normalized, and a shared label map is constructed so that both the supervised and similarity-based approaches operate on the same tag space.

For the supervised baseline, we fine-tune a transformer-based token classification model with first-subtoken alignment. In this setup, each word contributes only its first subtoken to the training objective, which preserves word-level BIO supervision while remaining compatible with subword tokenization. The model is trained with configurable hyperparameters, including learning rate, batch size, sequence length, epoch count, and random seed. Evaluation is performed on a held-out split and reported using precision, recall, F1-score, and a confusion matrix.

To provide a lightweight alternative, we implement a Common Vector Approach (CVA) classifier. The same transformer backbone is used as an embedding extractor, and aligned word embeddings are grouped by BIO label to compute class vectors. At inference time, each embedding is assigned to the nearest class vector by cosine similarity. This produces a direct comparison between fine-tuned classification and a similarity-based baseline under a shared evaluation protocol.

In addition to modeling, the pipeline computes corpus-level statistics and quality-control reports. These include counts of documents, sentences, tokens, entity-bearing sentences, BIO labels, and entity types, together with plots showing entity frequency, entity-to-sentence ratio, and sentence-length distributions. The tagset is checked against the canonical ENER tagset to identify unused labels and types that are not covered by the reference schema.

### Results

The repository includes example corpus statistics in `results_example/stats.json`. In that summary, the corpus contains 130 document folders, of which 34 are annotated and 96 are unannotated. The annotated portion comprises 1,142 sentences and 29,195 tokens. Entities appear in 980 sentences, corresponding to an entity-sentence ratio of 0.8581. The corpus contains 161 BIO labels, 97 distinct entity types, and an average sentence length of 25.56 tokens.

The analysis pipeline also generates label-frequency and type-frequency tables, together with plots for entity distribution and sentence-length variation. These outputs provide a descriptive overview of the dataset and support corpus inspection before model training.

The BERT-versus-CVA evaluation pipeline is fully implemented and exports precision, recall, F1-score, confusion matrices, and inference-time summaries for both methods. However, the repository does not currently include a finalized paper table with full-dataset model scores, so no numeric model-performance claims are stated here beyond the available corpus statistics.

## License

No license file is currently included in the repository.
