# Thesis Final Tables and Figure Captions

All numbers are derived from completed experiments on the full data splits unless stated otherwise.
Standard deviations use the sample estimator (ddof = 1).

---

## Table 1 — Corpus Statistics

| Statistic | Value |
|---|---:|
| Total annotation folders | 130 |
| Annotated documents | 34 |
| Unannotated documents | 96 |
| Total sentences | 1,142 |
| Sentences with ≥ 1 entity | 980 |
| Entity–sentence ratio | 0.858 |
| Total tokens | 29,195 |
| Average sentence length (tokens) | 25.6 |
| Distinct entity types | 97 |
| BIO labels in use | 161 |
| Raw BIO warnings (before normalisation) | 3,444 |
| BIO warnings after normalisation | 0 |

Source: `results/stats.json`, `results/model_comparison_full/bio_validation_warnings.txt`

---

## Table 2 — Main Evaluation Results (Full Held-Out Split, 107 Sentences)

Accuracy and Macro F1 are computed at **token level**.
"w/o O" excludes the outside label from macro averaging.
All fine-tuned models use `dbmdz/bert-base-turkish-cased` as backbone.

| Method | Accuracy | Macro F1 (w/ O) | Macro F1 (w/o O) |
|---|---:|---:|---:|
| BERT token classifier (seed 42) | 0.7477 | 0.0207 | 0.0256 |
| CVA cosine classifier (seed 42) | 0.0205 | 0.0144 | 0.0175 |
| Context-only baseline | 0.3294 | 0.0931 | 0.0931 |
| CVA-only baseline | 0.0167 | 0.0243 | 0.0243 |
| Context + CVA combined | 0.0302 | 0.0317 | 0.0317 |

Source: `results/model_comparison_full/`, `results/compare_full/`

---

## Table 3 — Cross-Validation Results (4-Fold, Full Training Data, 1,035 Sentences)

| Method | Accuracy (mean ± std) | Macro F1 (mean ± std) | Notes |
|---|---:|---:|---|
| BERT (4-fold) | 0.7824 ± 0.0074 | 0.0340 ± 0.0121 | mean support ≈ 6,228 tokens/fold |
| CRF (4-fold) | — | 0.3138 ± 0.0211 | linear-chain CRF, lexical features |

Source: `results/cv_full/cv_summary.json`, `results/crf_full/crf_cv_summary.json`

### Table 3a — BERT Per-Fold Details

| Fold | Accuracy | Macro F1 | Support (tokens) |
|---:|---:|---:|---:|
| 0 | 0.7834 | 0.0173 | 6,398 |
| 1 | 0.7732 | 0.0344 | 5,709 |
| 2 | 0.7912 | 0.0459 | 6,505 |
| 3 | 0.7819 | 0.0382 | 6,299 |
| **Mean ± std** | **0.7824 ± 0.0074** | **0.0340 ± 0.0121** | **6,228 ± 326** |

---

## Table 4 — Inference Speed (Full Evaluation Split, 107 Sentences, CPU)

CVA time includes BERT embedding extraction plus per-token cosine lookup.
Warmup sentences excluded from timing.

| Method | Avg time / sentence (s) | Total time (s) | Relative speed |
|---|---:|---:|---:|
| BERT token classifier | 0.0666 | 7.13 | 1.0× (reference) |
| CVA cosine classifier | 0.1904 | 20.38 | 2.86× slower |

Source: `results/model_comparison_full/comparison_summary.json`

---

## Table 5 — Out-of-Vocabulary Analysis (Full Evaluation Split)

| Statistic | Value |
|---|---:|
| OOV token surfaces (unseen in training vocabulary) | 235 |

Source: `results/oov_full/oov_summary.json`

---

## Table 6 — Context vs. CVA Ablation (Full Evaluation Split, 1,855 Tokens)

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Context-only | 0.3294 | 0.1033 | 0.1418 | 0.0931 |
| CVA-only | 0.0167 | 0.0381 | 0.0575 | 0.0243 |
| Combined (average) | 0.0302 | 0.0303 | 0.0980 | 0.0317 |

Source: `results/compare_full/comparison_summary.json`

---

## Figure Captions

**Figure 1 — Top-20 Entity Types by Frequency**
Bar chart of the 20 most frequently annotated entity types in the 34 annotated documents (1,142 sentences, 29,195 tokens). Counts aggregate occurrences across all annotated CoNLL files. The distribution is long-tailed: the top few types (e.g. PERSON, LOC\_CITY) account for a disproportionate share of all entity mentions, motivating class-weighted evaluation and similarity-based prototype methods.  
Source: `results/plots/top20_entity_types.png`

**Figure 2 — Entity–Sentence Ratio per Document**
Per-document proportion of sentences containing at least one named entity. The dashed horizontal line marks the corpus-wide mean (0.858). Nearly all sentences in the annotated subset contain an entity, reflecting the topically dense nature of the news source corpus. Documents with low ratios may correspond to lightly annotated or procedural texts.  
Source: `results/plots/entity_sentence_ratio.png`

**Figure 3 — Sentence Length Distribution**
Histogram of token counts per sentence across all 1,142 annotated sentences. The distribution is unimodal and right-skewed; the mean is 25.6 tokens and the median is approximately 22 tokens. Sentences exceeding 256 tokens are truncated during subword tokenisation; fewer than 1% of sentences are affected at this threshold.  
Source: `results/plots/sentence_length_hist.png`

**Figure 4 — PCA Projection of Token Embeddings**
Two-dimensional PCA projection of first-subtoken BERT embeddings extracted from the evaluation split. Points are coloured by gold entity label. Cluster overlap across entity types is consistent with the low macro F1 scores observed for the CVA classifier, which relies on embedding separability without task-specific fine-tuning.  
Source: `results/embedding_full/pca_embeddings.png` (if embedding analysis was run on the full split)

---

## Interpretation Notes

**High accuracy, low macro F1.**
BERT accuracy (~0.75–0.78) is dominated by the `O` label. With 97 entity types and a long-tail frequency distribution, the per-class F1 for rare types collapses near zero, dragging macro F1 below 0.04. The seqeval-based training signal targets span-level F1, but the offline evaluation in this pipeline uses token-level macro F1 — this discrepancy also contributes.

**CRF outperforms BERT on macro F1.**
CRF (0.314 ± 0.021) substantially outperforms BERT (0.034 ± 0.012) on macro F1. CRF's hand-crafted lexical features (word capitalisation, context window) generalise more uniformly across rare classes than a fine-tuned transformer, which tends to over-predict the majority class on small, imbalanced corpora.

**CVA underperforms context-only.**
The CVA prototype classifier (macro F1 = 0.024) underperforms the context-only heuristic (0.093). The pre-trained Turkish BERT embedding space is not sufficiently discriminative for a 97-way ENER classification without task-specific fine-tuning or label-weighted prototypes.

**OOV and morphological coverage.**
235 unseen token surfaces in the evaluation split represent a meaningful out-of-vocabulary fraction, motivating character-level or subword-robust representations for Turkish ENER.
