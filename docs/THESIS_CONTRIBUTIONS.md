# Thesis Contributions

## Extracting Extended Named Entity Embeddings from Text via Deep Neural Networks

This document explains the project's components in the order a thesis methodology chapter would
present them: dataset, annotation, taxonomy, architecture, attention mechanism, entity embeddings,
prototype vectors, cross-validation, results, limitations, and future work.

---

## 1. Dataset

- **Source format:** CoNLL-style, tab/space-separated `TOKEN LEMMA POS BIO_LABEL` per line, blank line
  between sentences. Files: `data/full_train.conll` (28,375 lines), `data/full_eval.conll` (1,962 lines).
- **Size:** 1,142 annotated sentences, 29,195 tokens, 85.8% of sentences contain at least one entity.
- **Smoke-test subset:** `data/_smoke_train.conll`, `data/_smoke_eval.conll` for fast pipeline checks
  without GPU time.
- **Original annotation source:** `data/annotation/` — 130 INCEpTION document folders, of which 34 are
  fully annotated and converted into the final CoNLL files; the remaining folders are tracked as a
  documented limitation (see Section 10).

## 2. Annotation Process

- Annotation was performed in **INCEpTION**, a web-based annotation tool that exports per-document
  span annotations later converted to token-level BIO tags.
- `ner_stats/conll_reader.py` parses the INCEpTION export format; `ner_stats/spans.py` converts between
  character-offset spans (as INCEpTION stores them) and BIO token tags.
- `ner_stats/data_utils.py::validate_bio_sequence()` enforces BIO well-formedness (no `I-X` without a
  preceding `B-X` or `I-X` of the same type `X`), and `normalize_bio_labels()` repairs minor annotation
  inconsistencies (e.g., a sentence-initial `I-` tag is promoted to `B-`).
- Quality control: `data/ENER-tagset.tsv` defines the 131 canonical entity types; `ner_stats/tagset.py`
  cross-checks every label observed in the CoNLL files against this canonical list, flagging any
  out-of-tagset labels.

## 3. Extended Entity Taxonomy

- **131 canonical types** in `data/ENER-tagset.tsv`; **97 unique types actually observed** in the
  annotated corpus (the remainder are defined but unused, reflecting the open-domain design of the
  tagset rather than annotation error).
- Taxonomy spans person/organization (`PERSON`, `ORG`, `ORG_POLITICAL`, `ORG_ETHNIC`, `ORG_FAMILY`),
  location (`LOC`, `LOC_CITY`, `LOC_COUNTRY`, `LOC_REGION`, `LOC_GEO`, `LOC_PROVINCE`, `LOC_ADDRESS`,
  `LOC_ASTRAL`), temporal (`DATE`, `TIME`, `DURATION`, `PERIOD`, `YEAR`, `ERA`), event/disaster
  (`EVENT`, `DISASTER`, `WAR`, `DISEASE`, `FESTIVAL`, `CONFERENCE`), facility (`FAC_AIRPORT`,
  `FAC_MUSEUM`, `FAC_PARK`, `FAC_RELIGIOUS`, `FAC_SCHOOL`, `FAC_STATION`), and abstract/product
  (`PRO_AWARD`, `PRO_CLASS`, `PRO_CULTURE`, `PRO_LANGUAGE`, `PRO_LAW`, `PRO_RULE`, `PRO_SERVICE`,
  `PRO_STYLE`, `PRO_THEORY`, and 10+ more) categories, among others.
- This is 19 more fine-grained types than FIGER (112) and a comparable order of magnitude to FewNERD's
  66 fine-grained types — see `docs/LITERATURE_REVIEW.md` Part I and the cross-lingual synthesis table
  for how this scheme compares internationally.

## 4. Model Architecture

Six model families share the same data pipeline and 4-fold CV protocol:

| Model | File | Idea |
|---|---|---|
| BERT NER | `scripts/train_ner.py`, `scripts/run_cross_validation.py` | Fine-tune `dbmdz/bert-base-turkish-cased` with a token-classification head, first-subtoken label alignment |
| CRF baseline | `scripts/run_crf_baseline.py` | Linear-chain CRF (`sklearn-crfsuite`) over lexical/orthographic features |
| Attention NER | `ner_stats/attention_ner.py`, `scripts/run_attention_ner.py` | Learned Q/K/V self-attention head on top of BERT hidden states (frozen or fully fine-tuned) |
| Character NER | `ner_stats/char_features.py`, `scripts/run_char_ner.py` | Hybrid BERT + character-CNN encoder for morphological/OOV robustness |
| Contrastive NER | `ner_stats/contrastive.py`, `scripts/run_contrastive_ner.py` | SupConLoss (Khosla et al., 2020) entity representation learning |
| CVA classifier | `ner_stats/cva.py` | Common Vector Approach: SVD/mean-based class prototype classification, no gradient training |

The full architecture matches the thesis's target pipeline:

```
Text → Tokenizer → BERT Encoder → Hidden States → Attention Layer (learned Q/K/V)
     → Entity Embeddings → Classification Head → Extended Entity Labels
```

## 5. Attention Mechanism

- Implemented per Vaswani et al. (2017): for hidden states `X` of an entity span,
  `Q = XW_Q`, `K = XW_K`, `V = XW_V`, `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`.
- Two complementary attention analyses exist:
  1. **`ner_stats/attention_ner.py`** — a *learned, trainable* Q/K/V head (`nn.Linear(d_model, d_head,
     bias=False)` for each of `W_Q`, `W_K`, `W_V`) stacked on top of BERT, trained end-to-end as part of
     the AttentionNER model family (`results/attention_ner_full_finetune/`, macro-F1 6.38%, the best
     neural model in the comparison).
  2. **`scripts/extract_qkv_vectors.py`** — extracts BERT's *own internal* Q/K/V projections (layer 11)
     for every entity token, without any additional training, to test whether off-the-shelf BERT
     attention already encodes entity-discriminative structure (`results/qkv_analysis/`).
- Attention heatmaps (`scripts/generate_attention_heatmaps.py`, `results/attention_heatmaps/`) visualize
  entity-to-entity vs. entity-to-context attention mass per layer.

## 6. Entity Embeddings

- `ner_stats/embeddings.py::TransformerEmbedder` extracts word-aligned BERT hidden states (first
  subtoken per word) for every token in train/eval, producing a 24,911 × 768 embedding matrix
  (`results/embedding_full/{train,eval}_embeddings.{csv,npy}`).
- Entity-only subsets (excluding `O`) are extracted separately for representation analysis:
  `results/qkv_analysis/entity_embeddings.npy` (hidden states), `entity_q_vectors.npy`,
  `entity_k_vectors.npy`, `entity_v_vectors.npy` (Q/K/V projections), 466 entity tokens × 768 dims each.
- Visualizations colored by entity type: PCA, t-SNE, and UMAP, for both the full embedding space
  (`results/embedding_full/`, `thesis_figures/05_pca_embeddings.png`,
  `thesis_figures/06_tsne_embeddings.png`) and the entity-only Q/K/V/hidden-state spaces
  (`results/qkv_analysis/pca_*.png`, `tsne_*.png`, and the newly added `results/plots/{pca,tsne,umap}_entities.png`).

## 7. Prototype Vectors

- `prototype(label) = mean(entity_embeddings)` computed over all training-set entity tokens of a given
  base label (BIO prefix stripped), for all 96 observed labels (excluding `O`/`null`).
- Stored as `prototype_vectors.pkl` (dict: label → 768-dim `float32` numpy array) and, by Q/K/V channel,
  as `results/qkv_analysis/prototype_vectors.npy` + `prototype_labels.json` (51 classes with sufficient
  eval-set support).
- Cosine-similarity evaluation across all prototype pairs, with a focused comparison of the
  supervisor-requested labels (PERSON, ORG, DATE, EVENT, DISEASE, LOC_CITY, LOC_COUNTRY), is in
  `prototype_analysis.md`. Key finding: many prototype pairs exceed 0.85 cosine similarity, indicating
  that mean-embedding prototypes under-discriminate several label pairs — a likely contributor to the
  CVA mean-vector classifier's relatively low (32.9%) but still BERT-fine-tuning-beating accuracy.
- A pairwise prototype similarity heatmap is in `thesis_figures/10_prototype_similarity.png` (full
  label set) and `thesis_figures/17_prototype_similarity_heatmap.png` (Q/K/V-derived, 51 classes).

## 8. Cross-Validation

- True 4-fold cross-validation (not a single train/test split) with **shared fold indices** between
  BERT and CRF, generated by `make_fold_indices()` in `scripts/run_cross_validation.py` /
  `scripts/run_crf_baseline.py` (`seed=42`).
- Per-fold outputs: `results/{cv_full,crf_full}/fold_{0,1,2,3}/{metrics_summary.csv,
  per_class_metrics.csv, confusion_matrix.csv, confusion_matrix.png, classification_report.json}`.
- Aggregated mean ± std across folds: `results/fold_results.csv`, `results/fold_summary.csv`,
  narrated in `results/cross_validation_report.md`.

## 9. Results

| Model | Accuracy | Macro F1 | Evaluation level |
|---|---|---|---|
| CRF (4-fold) | 0.6868 ± 0.0257 | **0.3138 ± 0.0244** | Entity-level (excl. `O`) — best overall |
| Context-only (mean cosine prototype) | 0.3294 | 0.0931 | All tokens |
| Attention NER (full fine-tune) | 0.7444 | 0.0638 | Entity-level (excl. `O`) — best neural model |
| BERT fine-tuned | 0.7824 ± 0.0074 | 0.0340 ± 0.0121 | Token-level (incl. `O`) |
| CharBERT (BERT + CharCNN) | 0.5200 | 0.0319 | All tokens |
| Contrastive NER (SupConLoss) | 0.5085 | 0.0316 | All tokens |
| CVA-only (mean-vector prototype) | 0.0167–0.329 | 0.0243–0.0931 | All tokens (see `results/cva_report.md`) |
| Attention NER (frozen BERT) | 0.7488 | 0.0047 | All tokens |

Full per-label breakdown: `results/label_metrics.md`, `results/publication_ready_label_table.csv`.
Statistical significance (Bootstrap CI, Wilcoxon signed-rank p=0.034, Cohen's d=14.54 for CRF vs.
BERT): `results/statistical_significance_report.md`.

**Counter-intuitive finding:** the CRF baseline and even the unsupervised mean-vector prototype
classifier outperform fully fine-tuned BERT by a wide margin on macro-F1, despite BERT's higher raw
token accuracy. This is explained by class imbalance (the majority `O` class is ~89% of tokens) and
label sparsity (many of the 97 types have <10 training examples) — see Section 10.

## 10. Limitations

- **Corpus size vs. taxonomy size:** 1,142 sentences across 97 observed entity types means most types
  have fewer than 10 training examples — insufficient for stable gradient-based fine-tuning of rare
  classes, and numerically unstable for SVD-based CVA prototypes (see `results/cva_report.md`).
- **Annotation coverage:** only 34 of 130 INCEpTION document folders are fully annotated; the remaining
  96 folders are not converted to CoNLL and are excluded from all experiments. Completing this would
  require additional manual annotation effort beyond the scope of this thesis; the exact folders and
  conversion script (`ner_stats/conll_reader.py`) are documented so this can be resumed.
- **Q/K/V representation not used for direct classification:** BERT's internal Q/K/V projections are
  extracted and analyzed for class separability (`results/qkv_analysis/`, `prototype_analysis.md`), but
  classifying tokens by nearest Q/K/V prototype (rather than by the trained classification head) was not
  run as a standalone experiment. To complete: build a cosine-similarity classifier over
  `entity_q_vectors.npy`/`entity_k_vectors.npy`/`entity_v_vectors.npy` analogous to the existing
  CVA mean-vector classifier, and report its accuracy/F1 alongside the table in Section 9.
- **Entity-level (span) vs. token-level evaluation:** most reported metrics are token-level; a stricter
  span-level (seqeval entity-level) evaluation would likely report lower numbers still, and is noted as
  a methodological caveat rather than completed in full for every model family.

## 11. Future Work

- Extend annotation to the remaining 96 INCEpTION document folders to reduce label sparsity.
- Train a direct Q/K/V-prototype nearest-centroid classifier and report it alongside CVA/BERT/CRF.
- Apply class-weighted loss or focal loss to BERT/Attention-NER fine-tuning to counter the 89% `O`-class imbalance.
- Restrict SVD-based CVA to classes with ≥10 examples, falling back to mean vectors otherwise (already
  recommended in `results/cva_report.md`).
- Explore cross-lingual transfer from the German/French/Arabic/Chinese resources surveyed in
  `docs/LITERATURE_REVIEW.md` Part V (e.g., multilingual BERT fine-tuned jointly on WikiNER + this
  corpus) to partially offset Turkish-specific data scarcity.
