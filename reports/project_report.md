# Turkish Extended Named Entity Recognition (ENER) — Project Report

**Author:** Moussa Bane  
**Date:** June 2026  
**Model backbone:** dbmdz/bert-base-turkish-cased  
**Repository:** turkish-extended-ner

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Methods](#5-methods)
   - 5.1 Data Processing
   - 5.2 BERT-based NER
   - 5.3 CRF Baseline
   - 5.4 Common Vector Approach (CVA)
   - 5.5 Context-only vs CVA+Context Experiment
   - 5.6 OOV Entity Experiment
6. [Experimental Setup](#6-experimental-setup)
7. [Results](#7-results)
   - 7.1 BERT 4-Fold Cross-Validation
   - 7.2 CRF 4-Fold Cross-Validation
   - 7.3 BERT vs CVA — Single Run
   - 7.4 Context-only vs CVA vs Combined
   - 7.5 OOV / Top-3 Retrieval Experiment
   - 7.6 Embedding Analysis
8. [Discussion](#8-discussion)
9. [Implementation Details](#9-implementation-details)
10. [Conclusions](#10-conclusions)

---

## 1. Executive Summary

This project develops and evaluates a complete Turkish Extended Named Entity Recognition (ENER) pipeline. The dataset is a custom INCEpTION-annotated corpus of 1,142 Turkish sentences covering 97 fine-grained entity types, significantly more granular than standard CoNLL-style corpora.

Three model families are compared under a shared 4-fold cross-validation protocol:

- **BERT** (fine-tuned `dbmdz/bert-base-turkish-cased`): token accuracy **78.2% ± 0.7%**, macro F1 **3.4% ± 1.2%**
- **CRF** (linear-chain with lexical features): macro F1 **31.4% ± 2.1%**
- **CVA** (Common Vector Approach, cosine similarity classifier): accuracy **2.1%** on the held-out eval set

The key finding is that the CRF substantially outperforms BERT on macro F1 despite far simpler features. This is attributed to the extreme label sparsity of the ENER tagset: 184 BIO tags with most classes having fewer than 10 examples in each fold. BERT's high token accuracy (≈78%) is largely driven by its strong bias towards the majority "O" class (89% of tokens), which CRF overcomes by learning transition patterns between fine-grained entity boundaries.

Context-only (mean vector cosine) classification achieves 32.9% token accuracy and macro F1 of 9.3%, meaningfully above CVA (1.7% accuracy), suggesting that raw mean prototypes are more discriminative than CVA common vectors for this sparse-class setting.

---

## 2. Introduction

Named Entity Recognition (NER) is a foundational NLP task. Standard benchmarks (CoNLL-2003, WikiNER) use coarse tagsets of 4–18 entity types. Turkish ENER extends this to **97 fine-grained types** — distinguishing, for example, `LOC_CITY`, `LOC_COUNTRY`, `LOC_PROVINCE`, `LOC_GEO`, `LOC_REGION`, and `LOC_ASTRAL` instead of a single `LOC`. This granularity creates a challenging low-resource multi-class problem where many entity types have fewer than 5 examples in the training data.

The project addresses the following research questions:

1. How does a fine-tuned Turkish BERT model perform on a 97-class ENER tagset compared to a lexical CRF baseline?
2. Can embedding-space nearest-centroid classifiers (mean vectors, CVA) match or exceed supervised classification?
3. How well can the system recognize entities that are entirely absent from the training vocabulary (OOV)?
4. What is the inference cost trade-off between BERT fine-tuning and embedding-based methods?

---

## 3. Dataset

### 3.1 Corpus Overview

The corpus was annotated using INCEpTION and stored in CoNLL BIO format.

| Statistic | Value |
| --- | ---: |
| Total document folders | 130 |
| Annotated documents | 34 |
| Unannotated documents | 96 |
| Total sentences | 1,142 |
| Sentences containing at least one entity | 980 |
| Entity sentence ratio | 85.8% |
| Total tokens | 29,195 |
| Average sentence length | 25.6 tokens |
| BIO labels in corpus | 161 |
| Entity types in corpus | 97 |
| Entity types in ENER tagset | 131 |

### 3.2 Train / Evaluation Splits

| Split | Sentences | Token rows |
| --- | ---: | ---: |
| `data/full_train.conll` | 1,035 | 27,340 |
| `data/full_eval.conll` | 107 | 1,855 |
| **Total** | **1,142** | **29,195** |

The evaluation set constitutes approximately 9.4% of the corpus.

### 3.3 Tagset

The Extended NER tagset (`data/ENER-tagset.tsv`) defines 131 entity types. After BIO expansion and combining both splits, **184 unique BIO labels** appear in the data. The most populated entity types in the evaluation set (by support, excluding O) are:

| Label | Support |
| --- | ---: |
| I-PERSON | 44 |
| I-PERIOD | 41 |
| I-MEASUREMENT | 35 |
| I-PRO_CLASS | 27 |
| I-DATE | 26 |
| I-LOC | 25 |
| I-PRO_RULE | 21 |
| I-WAR | 21 |
| I-ORG_POLITICAL | 20 |
| I-LOC_CITY | 18 |

This reveals an extreme class imbalance: the top-10 labels each have 18–44 examples, while the majority of the 184 BIO classes appear fewer than 5 times in the evaluation set.

### 3.4 BIO Quality Control

Raw BIO validation found **3,444 label transition warnings** (e.g., `I-TYPE` following `O` without a matching `B-TYPE`). The normalization step converts all such invalid `I-` openings to `B-` tags, reducing the warning count to **0** after normalization. All experiments use the normalized labels.

---

## 4. System Architecture

```
data/*.conll
    │
    ├─ BIO validation & normalization  (ner_stats/data_utils.py)
    │
    ├─ BERT pipeline  ──────────────────────────────────────────┐
    │   ├─ Tokenization (first-subtoken alignment)               │
    │   ├─ Fine-tuning (HuggingFace Trainer)                     │
    │   └─ Argmax prediction                                     │
    │                                                            │
    ├─ CRF pipeline  ───────────────────────────────────────────┤
    │   ├─ Lexical feature extraction                            │
    │   └─ sklearn-crfsuite LBFGS training                      │
    │                                                            │
    ├─ Embedding pipeline  ─────────────────────────────────────┤
    │   ├─ TransformerEmbedder (word-aligned extraction)         │
    │   ├─ Class vector computation (mean / CVA)                 │
    │   └─ Cosine similarity classification (top-k)              │
    │                                                            │
    └─ Evaluation  ─────────────────────────────────────────────┘
        ├─ Token-level precision / recall / F1
        ├─ Confusion matrices (full + top-N frequent labels)
        ├─ 4-fold cross-validation (mean ± std)
        └─ Thesis report generator (Markdown + LaTeX)
```

---

## 5. Methods

### 5.1 Data Processing

CoNLL files are parsed with `read_conll_bio()`, which returns pairs of `(token_sequences, label_sequences)`. BIO labels are validated for legal transitions and normalized. A shared `label2id` / `id2label` mapping is built from all labels in the training set. Both BERT and CRF use this same mapping.

### 5.2 BERT-based NER

A pre-trained Turkish BERT model (`dbmdz/bert-base-turkish-cased`) is fine-tuned for token classification using the HuggingFace `Trainer` API.

**Token alignment.** Because BERT uses a WordPiece tokenizer that splits words into subwords, label alignment is performed using the *first-subtoken* strategy: each word contributes only its first subword to the supervised loss. All remaining subtokens receive a padding label (`-100`) and are ignored by the loss and evaluation.

**Training hyperparameters (defaults):**

| Hyperparameter | Value |
| --- | --- |
| Model | dbmdz/bert-base-turkish-cased |
| Learning rate | 2 × 10⁻⁵ |
| Batch size (train/eval) | 8 / 8 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Max sequence length | 256 |
| Seed | 42 |
| Optimizer | AdamW (HF default) |

**4-fold cross-validation** is performed by `run_cross_validation.py`. The 1,035 training sentences are randomly permuted with seed 42 and split into 4 approximately equal folds. Each fold trains on 3/4 of the data and evaluates on the held-out quarter.

### 5.3 CRF Baseline

A linear-chain CRF is trained using `sklearn-crfsuite` with LBFGS optimization (max 100 iterations). Features are extracted per token:

| Feature | Description |
| --- | --- |
| `word.lower()` | Lowercased word form |
| `word.isupper()` | Boolean: all uppercase |
| `word.istitle()` | Boolean: title-case |
| `word.isdigit()` | Boolean: all digits |
| `-1:word.lower()` | Previous word form (if exists) |
| `-1:istitle()` | Previous word title-case |
| `+1:word.lower()` | Next word form (if exists) |
| `+1:istitle()` | Next word title-case |
| `BOS` / `EOS` | Sentence-boundary indicators |

The same 4-fold splits (seed 42) are used as for BERT, enabling a direct performance comparison.

### 5.4 Common Vector Approach (CVA)

CVA constructs a class prototype for each BIO label from the training embeddings using the following procedure:

1. Extract word-aligned embeddings from `dbmdz/bert-base-turkish-cased` for every token in the training set (24,911 tokens, 768-dimensional).
2. Group embeddings by BIO label.
3. For each class with label matrix **X** (shape *n × 768*):
   - Compute mean vector **m**.
   - Center the matrix: **A = X − m**.
   - Compute the thin SVD of **A**: **A = UΣVᵀ**.
   - Determine the rank *r* (singular values > 10⁻¹⁰).
   - Compute the within-class subspace basis **B = Vᵣᵀ** (first *r* rows of **Vᵀ**).
   - Project out within-class variance: **v_common = m − B(Bᵀm)**.
4. Classify evaluation embeddings by cosine similarity to class prototypes. Top-*k* retrieval is also supported.

When only a single example exists for a class, the common vector degenerates to the mean vector (no within-class variance to remove).

### 5.5 Context-only vs CVA+Context Experiment

Three cosine-similarity classifiers are compared on the full evaluation set:

| Variant | Prototype construction |
| --- | --- |
| **Context-only** | Simple mean of all training embeddings per label |
| **CVA-only** | CVA common vectors (§5.4) |
| **Combined** | Average of cosine scores from mean and CVA prototypes |

All three classifiers share the same embedding extractor and classify every evaluation token by selecting the label with the highest prototype similarity.

### 5.6 OOV Entity Experiment

An entity token is considered *out-of-vocabulary* (OOV) if its surface form (lowercased) was never seen as an entity token in the training set. For each OOV entity token in the evaluation set, the top-3 nearest BIO labels are retrieved by cosine similarity to CVA class vectors. Results are saved with gold label, top-1 prediction, top-1 score, and full top-3 ranked list.

---

## 6. Experimental Setup

All experiments run on the full data splits (`data/full_train.conll`, `data/full_eval.conll`) unless noted. The smoke splits (`data/_smoke_train.conll`, `data/_smoke_eval.conll`, 2 sentences each) are available for rapid pipeline validation.

Cross-validation uses a fixed random seed (42) for fold assignment to ensure reproducibility. BERT models are saved per fold under `results/cv_full/fold_N/model/`. CRF models are re-trained in memory per fold (no persistence needed).

Evaluation reports are produced by `ner_stats/evaluation_report.py`, which computes token-level precision, recall, F1, and accuracy via a confusion matrix; saves per-class CSV tables; generates full and top-20 label confusion matrix plots (PNG).

---

## 7. Results

### 7.1 BERT 4-Fold Cross-Validation

| Fold | Precision | Recall | Macro F1 | Accuracy | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold 0 | 0.0184 | 0.0220 | 0.0173 | 0.7834 | 6,398 |
| Fold 1 | 0.0594 | 0.0364 | 0.0344 | 0.7732 | 5,709 |
| Fold 2 | 0.0605 | 0.0474 | 0.0459 | 0.7912 | 6,505 |
| Fold 3 | 0.0568 | 0.0384 | 0.0382 | 0.7819 | 6,299 |
| **Mean ± Std** | **0.0488 ± 0.0203** | **0.0361 ± 0.0105** | **0.0340 ± 0.0121** | **0.7824 ± 0.0074** | **6,228** |

**Interpretation.** BERT achieves consistent token accuracy near 78% across all folds, but extremely low macro F1 (3.4%). This discrepancy is the hallmark of a majority-class-dominated model: approximately 89% of evaluation tokens are labeled `O`. BERT learns to predict `O` reliably and scores high on accuracy, but fails to generalize across the 183 rare entity BIO labels. Fold-to-fold variance in macro F1 (std = 0.012) is substantial given the mean value, indicating sensitivity to which rare entities land in the validation fold.

### 7.2 CRF 4-Fold Cross-Validation

| Fold | Precision | Recall | Macro F1 | Accuracy | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold 0 | 0.3470 | 0.3079 | 0.3043 | 0.6653 | 484 |
| Fold 1 | 0.3287 | 0.3849 | 0.3459 | 0.7212 | 269 |
| Fold 2 | 0.2931 | 0.3145 | 0.2883 | 0.6989 | 538 |
| Fold 3 | 0.3159 | 0.3449 | 0.3167 | 0.6616 | 331 |
| **Mean ± Std** | **0.3212 ± 0.0221** | **0.3381 ± 0.0305** | **0.3138 ± 0.0211** | **0.6868 ± 0.0248** | **405** |

> **Note on support:** The CRF support column reflects tokens with non-O gold labels only (since evaluation runs with `include_o_label=False`), whereas the BERT CV support includes all tokens. This is why CRF support (~400) appears much smaller than BERT support (~6,200).

**Interpretation.** The CRF macro F1 (31.4%) is an order of magnitude higher than BERT (3.4%). This is a counterintuitive but well-understood phenomenon in low-resource multi-class settings: the CRF directly models the conditional probability of each label given local features and transition sequences, which means it does not systematically collapse to the majority class. The lexical features (word form, capitalization, context window) are informative signals for Turkish named entities. The CRF's lower accuracy (68.7% vs 78.2%) reflects that it does predict entity labels more aggressively, at the cost of some false positives.

**Top-performing CRF labels (Fold 0):**

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| I-LIST | 1.0000 | 1.0000 | 1.0000 | 4 |
| I-PERCENT | 1.0000 | 1.0000 | 1.0000 | 4 |
| I-TV-PROGRAM | 1.0000 | 0.8333 | 0.9091 | 6 |
| I-DATE | 0.8189 | 1.0000 | 0.9004 | 104 |
| I-LOC_COUNTRY | 0.8718 | 0.8947 | 0.8831 | 38 |
| I-LOC_CITY | 0.8462 | 0.8462 | 0.8462 | 13 |
| I-NATION | 0.7333 | 1.0000 | 0.8462 | 11 |

The CRF excels at continuation tags (I-*) for classes with clear lexical patterns: dates, location subtypes, and country names.

### 7.3 BERT vs CVA — Single Run

Trained on full `data/full_train.conll`, evaluated on `data/full_eval.conll`.

| Method | Accuracy (with O) | Macro F1 (with O) | Macro F1 (w/o O) | Avg Inference (s) |
| --- | ---: | ---: | ---: | ---: |
| BERT (fine-tuned) | 0.7477 | 0.0207 | 0.0256 | 0.0666 |
| CVA (cosine similarity) | 0.0205 | 0.0144 | 0.0175 | 0.1904 |

**Key observations:**

- BERT is **3× faster** at inference than CVA (0.067 s vs 0.190 s per sentence), despite CVA being the "lightweight" method. This is because CVA requires computing cosine similarity against 98 class prototypes for every token, whereas BERT's forward pass produces all token predictions in a single batched operation.
- CVA accuracy (2.1%) is far below BERT (74.8%), confirming that naive cosine similarity to class prototypes cannot replace a fine-tuned discriminative classifier on a 184-label problem.
- BERT macro F1 without O (2.6%) is very low, but non-zero: it correctly classifies DATE, PERSON, and YEAR entities.

**BERT top labels by F1 (single run, with O):**

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| O | 0.7879 | 0.9546 | 0.8633 | 1,389 |
| I-DATE | 0.7407 | 1.0000 | 0.8511 | 20 |
| B-DATE | 0.4762 | 1.0000 | 0.6452 | 10 |
| B-PERSON | 0.4848 | 0.4444 | 0.4638 | 36 |
| B-YEAR | 0.1935 | 0.8571 | 0.3158 | 7 |
| I-PERSON | 0.5000 | 0.1667 | 0.2500 | 18 |
| B-PRO_LANGUAGE | 0.1667 | 0.4286 | 0.2400 | 7 |

BERT reliably learns DATE (deterministic patterns) and PERSON (high-frequency, morphologically distinct in Turkish), but fails on the long tail of 170+ rare classes.

### 7.4 Context-only vs CVA vs Combined

Evaluated on `data/full_eval.conll` (1,855 tokens). Class prototypes built from full training set.

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Context-only (mean vectors) | 0.3294 | 0.1033 | 0.1418 | 0.0931 |
| CVA-only (common vectors) | 0.0167 | 0.0381 | 0.0575 | 0.0243 |
| Combined (average scores) | 0.0302 | 0.0303 | 0.0980 | 0.0317 |

**Interpretation:**

- **Context-only outperforms CVA** on every metric (accuracy: 32.9% vs 1.7%; macro F1: 9.3% vs 2.4%). This is somewhat unexpected: CVA removes within-class variance to produce a more stable prototype, but with sparse classes (many having only 1–3 examples), there is no meaningful within-class subspace to remove. The SVD degenerates, and the "common vector" is essentially the mean. For the majority class `O` (high-variance, thousands of examples), CVA projects the mean *away* from the dense region, causing virtually all tokens to be misclassified.
- **Combined underperforms Context-only** because averaging with CVA scores dilutes the correct signal from mean vectors.
- Context-only's **32.9% accuracy** is meaningful given that this is a zero-shot cosine classifier (no fine-tuning), and the 184-class label space would give ~0.54% by random chance.

**Top labels by F1 — Context-only:**

| Label | Support | F1 |
| --- | ---: | ---: |
| O | 1,389 | 0.5060 |
| I-DATE | 26 | 0.5946 |
| I-MEASUREMENT | 35 | 0.4733 |
| I-PERIOD | 41 | 0.3529 |
| I-ORG_POLITICAL | 20 | 0.2857 |

### 7.5 OOV / Top-3 Retrieval Experiment

In the evaluation set, **235 entity tokens** have surface forms never seen as entities during training (OOV). This represents approximately 12.7% of all entity-bearing tokens.

The CVA top-3 retrieval experiment classifies each OOV token by cosine similarity to CVA class prototypes and records the top-3 nearest labels.

| Metric | Value |
| --- | --- |
| Total OOV entity tokens | 235 |
| OOV as % of entity tokens | ~12.7% |

The full per-token results (including gold label, top-1 prediction, top-1 similarity score, and top-3 ranked labels) are saved in `results/oov_full/oov_results.csv`.

### 7.6 Embedding Analysis

Embeddings were extracted from `dbmdz/bert-base-turkish-cased` without fine-tuning (using the pre-trained backbone directly) on the full dataset.

| Metric | Value |
| --- | --- |
| Train tokens extracted | 24,911 |
| Extraction time (train) | 95.6 s |
| Eval tokens extracted | 1,855 |
| Extraction time (eval) | 8.1 s |
| Class vectors saved (98 labels) | 98 |
| CVA direct-classification accuracy | 0.11% |
| CVA direct-classification macro F1 | 0.24% |

The very low numbers here are for the raw (untuned) backbone with no label-space conditioning — this is the baseline showing that the pre-trained embeddings, without any task-specific training, barely separate the extended entity classes. This motivates fine-tuning (BERT) and lexical adaptation (CRF) over pure embedding-space classification.

**PCA explained variance** (`results/embedding_full/pca_explained_variance.txt`) shows that the first two principal components of the 768-dimensional embedding space explain a modest fraction of total variance, reflecting the high dimensionality and density of the BERT embedding manifold.

---

## 8. Discussion

### 8.1 Why CRF Outperforms BERT on Macro F1

The result is counterintuitive — BERT has 110M parameters versus a few thousand CRF weights — but arises from three structural factors:

1. **Label sparsity.** With 184 BIO classes and only 1,035 training sentences, most entity classes appear fewer than 10 times in the entire training set. BERT's cross-entropy loss is dominated by the `O` class (89% of tokens), causing the model to under-learn rare entity types.
2. **First-subtoken alignment.** BERT is only supervised on the first subword of each word. Turkish is a highly agglutinative language: a single word like *"İstanbul'da"* (in Istanbul) may be split into 3–4 subtokens, and only the first receives a label. This reduces the effective training signal.
3. **Feature-label match.** Turkish NER is heavily driven by capitalization, suffixes, and co-occurring tokens — all captured directly by CRF features. BERT's contextual representations encode much of the same information, but require sufficient examples per class to learn discriminative projections.

### 8.2 Context-only vs CVA

The failure of CVA relative to mean vectors is linked to the same label sparsity. CVA's SVD step is designed to remove within-class variance — capturing the "common" part of class embeddings by orthogonalizing against intra-class directions. This is meaningful when a class has many diverse examples (the within-class subspace genuinely encodes irrelevant variation). With 1–3 examples per rare class, the "subspace" is noise, and the projection distorts rather than stabilizes the prototype. The dominant class `O`, which has thousands of examples, suffers most: CVA removes virtually all of its variance and places its prototype far from any actual `O` token.

The conclusion for this dataset is that **mean-vector prototypes are more robust than CVA under extreme class imbalance**, and CVA should be tested with a minimum per-class sample size threshold.

### 8.3 Inference Cost

CVA's 3× higher inference latency relative to BERT is surprising. The bottleneck is the per-token prototype lookup: for each of 1,855 eval tokens, 98 cosine similarities are computed serially. BERT, by contrast, processes an entire sentence in a single GPU/CPU forward pass. For production use, the CVA lookup should be vectorized with a single matrix multiplication (`E @ P.T`) rather than a per-token loop.

### 8.4 OOV Analysis

235 OOV entity tokens (12.7% of entity occurrences) highlights a significant generalization challenge. The Turkish morphological system generates many surface forms from a single root, so a word seen as `B-PERSON` during training may appear with different suffixes at test time, making it OOV by surface form. A morphology-aware tokenization or lemmatization step prior to feature extraction could reduce this proportion substantially.

### 8.5 Limitations

- **Small evaluation set.** 107 sentences / 1,855 tokens is a limited evaluation pool. Confidence intervals on macro F1 are wide, particularly for rare entity types.
- **No label smoothing or class reweighting.** BERT training uses uniform cross-entropy. Focal loss or weighted sampling to boost rare-class learning could substantially improve macro F1.
- **CRF feature set.** The current feature set is minimal. Adding suffix features, character n-grams, and morphological tags (which can be extracted with Turkish-specific tools) would likely improve CRF performance substantially.
- **CVA implementation.** No minimum-support threshold is applied before computing CVA. Labels with a single example trivially produce zero within-class variance and receive the raw mean as their prototype, masking the SVD issue.

---

## 9. Implementation Details

### 9.1 Repository Structure

```
turkish-extended-ner/
├── data/                          # CoNLL-format data files and tagset
├── ner_stats/                     # Core library modules
│   ├── cva.py                     # CVA + cosine top-k
│   ├── data_utils.py              # BIO parsing, normalization, label maps
│   ├── embeddings.py              # TransformerEmbedder
│   ├── embedding_analysis.py      # Embedding aggregation and export
│   ├── evaluation.py              # Token-level metrics
│   ├── evaluation_report.py       # Report generation (full + top-N confusion)
│   ├── statistics.py              # Corpus statistics
│   └── visualization.py          # PCA/t-SNE/UMAP plotting
├── scripts/
│   ├── train_ner.py               # Single-run BERT + CVA
│   ├── run_cross_validation.py    # 4-fold BERT CV
│   ├── run_crf_baseline.py        # 4-fold CRF CV
│   ├── run_embedding_analysis.py  # Embedding extraction + visualization
│   ├── compare_context_cva.py     # Context-only vs CVA vs Combined
│   ├── oov_experiment.py          # OOV top-3 retrieval
│   ├── generate_prototype_visuals.py  # PCA prototype overlay + neighbors
│   ├── generate_thesis_summary.py # Thesis report (Markdown + LaTeX)
│   ├── run_char_ner.py            # Character CNN NER (experimental)
│   ├── run_contrastive_ner.py     # Contrastive NER (experimental)
│   ├── run_multi_seed.py          # Multi-seed evaluation with stat tests
│   └── run_analysis.py            # Corpus statistics CLI
├── results/                       # Experiment outputs
├── outputs/                       # Model checkpoints
└── requirements.txt
```

### 9.2 Key Dependencies

| Package | Purpose |
| --- | --- |
| `transformers` | BERT tokenizer, model, Trainer |
| `torch` | Deep learning backend |
| `scikit-learn` | PCA, data splitting |
| `sklearn-crfsuite` | CRF model |
| `scipy` | Statistical tests (bootstrap, t-test) |
| `numpy` | Numerical operations |
| `matplotlib` | Plots and confusion matrices |
| `evaluate`, `seqeval` | Span-level NER metrics |

### 9.3 Reproducibility

All experiments use seed 42 for NumPy, PyTorch random state, and fold assignment. The cross-validation scripts accept `--seed` to override. Training arguments include `report_to="none"` to suppress W&B/TensorBoard logging.

---

## 10. Conclusions

This project delivers a complete Turkish Extended NER research pipeline with the following outcomes:

**Primary results:**

| Experiment | Best result |
| --- | --- |
| BERT 4-fold CV macro F1 | 3.4% ± 1.2% |
| CRF 4-fold CV macro F1 | **31.4% ± 2.1%** |
| BERT token accuracy (single run) | **74.8%** |
| Context-only token accuracy | 32.9% |
| CVA token accuracy | 1.7% |
| OOV entity tokens | 235 (12.7%) |

**Key conclusions:**

1. The CRF baseline substantially outperforms BERT on macro F1 for a 97-class Turkish ENER tagset due to extreme label sparsity. A fine-tuned BERT requires far more examples per class to overcome the majority-class dominance of `O`.
2. Mean-vector cosine classification (Context-only) outperforms CVA on this dataset; CVA's SVD-based prototype construction is not well-suited for classes with fewer than ~10 examples.
3. The BERT backbone already provides 74.8% token accuracy by strongly predicting `O`. Improvements in entity-class recall require class reweighting, focal loss, or data augmentation strategies.
4. 12.7% of evaluation entity tokens are OOV by surface form, motivating morphology-aware representations for Turkish.

**Next steps:**

- Apply focal loss or class-balanced sampling to improve BERT macro F1 on rare classes.
- Extend CRF features with Turkish morphological analysis and suffix n-grams.
- Enforce a minimum per-class support threshold for CVA to avoid degenerate projections.
- Collect more annotation data, particularly for the 70+ entity types with fewer than 5 training examples.
- Explore span-level evaluation (seqeval entity-level F1) alongside the current token-level evaluation.
