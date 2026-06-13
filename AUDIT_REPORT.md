# Repository Audit Report — Turkish Extended NER (ENER)

**Author:** Moussa Bane  
**Date:** June 2026  
**Auditor:** Senior NLP Research Engineer  
**Scope:** Full repository audit for thesis submission readiness

---

## 1. Existing Components

The following components are implemented, tested, and producing results.

### 1.1 Data Pipeline
| Component | File | Status |
|-----------|------|--------|
| CoNLL reader | `ner_stats/conll_reader.py` | ✅ Complete |
| BIO validation | `ner_stats/data_utils.py` | ✅ Complete |
| Label map builder | `ner_stats/data_utils.py` | ✅ Complete |
| Tagset loader (131 types) | `ner_stats/tagset.py` | ✅ Complete |
| Corpus statistics | `ner_stats/statistics.py` | ✅ Complete |
| INCEpTION annotation reader | `ner_stats/conll_reader.py` | ✅ Complete |
| Character span converter | `ner_stats/spans.py` | ✅ Complete |

### 1.2 Models
| Component | File | Status |
|-----------|------|--------|
| BERT fine-tuning (HF Trainer) | `scripts/train_ner.py` | ✅ Complete |
| CRF baseline (sklearn-crfsuite) | `scripts/run_crf_baseline.py` | ✅ Complete |
| Attention NER (learned Q/K/V head) | `ner_stats/attention_ner.py` | ✅ Complete |
| Character CNN encoder | `ner_stats/char_features.py` | ✅ Complete |
| Hybrid BERT+CharCNN | `scripts/run_char_ner.py` | ✅ Complete |
| Contrastive NER (SupConLoss) | `ner_stats/contrastive.py` | ✅ Complete |
| Common Vector Approach (CVA) | `ner_stats/cva.py` | ✅ Complete |

### 1.3 Evaluation
| Component | File | Status |
|-----------|------|--------|
| Token-level metrics (P/R/F1) | `ner_stats/evaluation.py` | ✅ Complete |
| Confusion matrix (full + top-N) | `ner_stats/evaluation_report.py` | ✅ Complete |
| Per-label classification report | `scripts/run_attention_ner.py` | ✅ Complete |
| 4-fold cross-validation (BERT) | `scripts/run_cross_validation.py` | ✅ Complete |
| 4-fold cross-validation (CRF) | `scripts/run_crf_baseline.py` | ✅ Complete |
| Multi-seed + bootstrap significance | `scripts/run_multi_seed.py` | ✅ Complete |

### 1.4 Embedding & Representation
| Component | File | Status |
|-----------|------|--------|
| TransformerEmbedder (word-aligned) | `ner_stats/embeddings.py` | ✅ Complete |
| Embedding extraction pipeline | `scripts/run_embedding_analysis.py` | ✅ Complete |
| CVA class vectors (mean + SVD) | `ner_stats/cva.py` | ✅ Complete |
| OOV entity retrieval (top-K) | `scripts/oov_experiment.py` | ✅ Complete |
| PCA / t-SNE / UMAP visualization | `ner_stats/visualization.py` | ✅ Complete |
| Prototype overlay plots | `scripts/generate_prototype_visuals.py` | ✅ Complete |

### 1.5 Documentation & Reports
| Component | File | Status |
|-----------|------|--------|
| Main README (EN) | `README.md` | ✅ Complete |
| Turkish README | `README.tr.md` | ✅ Complete |
| Project report | `reports/project_report.md` | ✅ Complete |
| Thesis attention NER section | `docs/thesis_attention_ner_section.md` | ✅ Complete |
| Thesis summary generator | `scripts/generate_thesis_summary.py` | ✅ Complete |

### 1.6 Experimental Results
| Experiment | Output Location | Status |
|------------|-----------------|--------|
| BERT 4-fold CV | `results/cv_full/` | ✅ 4 folds complete |
| CRF 4-fold CV | `results/crf_full/` | ✅ 4 folds complete |
| Context vs CVA vs Combined | `results/compare_full/` | ✅ Complete |
| Attention NER (frozen BERT) | `results/attention_ner/` | ✅ Complete |
| Attention NER (weighted loss) | `results/attention_ner_weighted/` | ✅ Complete |
| Embedding extraction (full) | `results/embedding_full/` | ✅ Complete |
| OOV retrieval (full) | `results/oov_full/` | ✅ Complete |

---

## 2. Missing Components

The following items are required by supervisor requirements but are absent from the repository.

### 2.1 Structured Report Files
| Required File | Status | Notes |
|---------------|--------|-------|
| `fold_results.csv` | ❌ Missing | Must be generated from `cv_full/` JSON data |
| `fold_summary.csv` | ❌ Missing | Must be generated from `cv_full/` + `crf_full/` JSON |
| `cross_validation_report.md` | ❌ Missing | Narrative report on 4-fold CV |
| `comparison_results.csv` | ❌ Missing | Unified BERT vs CRF vs Attention comparison |
| `comparison_report.md` | ❌ Missing | Narrative comparison report |
| `label_metrics.csv` | ❌ Missing | Per-label P/R/F1 for all 97+ entity types |
| `label_metrics.md` | ❌ Missing | Markdown table of per-label metrics |
| `publication_ready_label_table.csv` | ❌ Missing | Formatted for LaTeX/paper submission |
| `cva_report.md` | ❌ Missing | CVA analysis narrative |
| `entity_embedding_analysis.md` | ❌ Missing | Embedding analysis narrative |
| `attention_analysis.md` | ❌ Missing | Attention mechanism analysis |
| `attention_entity_representation_report.md` | ❌ Missing | Thesis-ready attention report |
| `boundary_detection_report.md` | ❌ Missing | Character-level boundary report |
| `contrastive_learning_report.md` | ❌ Missing | Contrastive learning experiment report |
| `error_analysis_report.md` | ❌ Missing | False positive/negative analysis |
| `false_positive_analysis.md` | ❌ Missing | FP breakdown per label |
| `false_negative_analysis.md` | ❌ Missing | FN breakdown per label |
| `statistical_significance_report.md` | ❌ Missing | Bootstrap/McNemar tests |
| `related_work_review.md` | ❌ Missing | Academic literature review |
| `FINAL_THESIS_READINESS_REPORT.md` | ❌ Missing | Completion matrix and readiness score |
| `AUDIT_REPORT.md` | ✅ This file | Being generated now |

### 2.2 Required Output Files
| Required File | Status | Notes |
|---------------|--------|-------|
| `entity_embeddings.npy` | ❌ Missing | Entity-only subset of embeddings |
| `prototype_vectors.npy` | ❌ Missing | Class prototype array |
| `centroid_similarity_matrix.csv` | ❌ Missing | Pairwise cosine similarity between centroids |
| `prototype_similarity_heatmap.png` | ❌ Missing | Heatmap of centroid similarities |
| `entity_q_vectors.npy` | ❌ Missing | Q-projections for entity tokens |
| `entity_k_vectors.npy` | ❌ Missing | K-projections for entity tokens |
| `entity_v_vectors.npy` | ❌ Missing | V-projections for entity tokens |
| `attention_heatmaps/` | ❌ Missing | Attention visualization PNGs |
| `attention_statistics.csv` | ❌ Missing | Attention weight statistics |
| `bert_results.csv` | ❌ Missing | BERT evaluation result table |
| `bert_attention_results.csv` | ❌ Missing | Attention NER result table |
| `attention_comparison.md` | ❌ Missing | BERT vs Attention-BERT comparison |
| `pca_q_vectors.png` | ❌ Missing | PCA of Q-projections |
| `pca_k_vectors.png` | ❌ Missing | PCA of K-projections |
| `pca_v_vectors.png` | ❌ Missing | PCA of V-projections |
| `tsne_q_vectors.png` | ❌ Missing | t-SNE of Q-projections |
| `tsne_k_vectors.png` | ❌ Missing | t-SNE of K-projections |
| `tsne_v_vectors.png` | ❌ Missing | t-SNE of V-projections |
| `umap_q_vectors.png` | ❌ Missing | UMAP of Q-projections |
| `umap_k_vectors.png` | ❌ Missing | UMAP of K-projections |
| `umap_v_vectors.png` | ❌ Missing | UMAP of V-projections |
| `thesis_figures/` | ❌ Missing | Centralized publication-quality figure directory |

### 2.3 Missing Scripts
| Script | Purpose |
|--------|---------|
| `scripts/extract_qkv_vectors.py` | Extract Q/K/V projections from BERT attention layers |
| `scripts/generate_attention_heatmaps.py` | Visualize attention patterns per entity |
| `scripts/generate_thesis_figures.py` | Centralized publication-quality figure generator |
| `scripts/generate_error_analysis.py` | FP/FN analysis from predictions |
| `scripts/generate_statistical_significance.py` | Bootstrap/McNemar test runner |
| `scripts/generate_cva_analysis.py` | CVA embedding similarity analysis |

---

## 3. Weak Components

### 3.1 Attention NER Performance
**Issue:** The AttentionNER (frozen BERT + learned Q/K/V head) achieves 0% macro F1 without O-label.  
**Root Cause:** Training with `freeze_bert=True` leaves BERT weights frozen — the task-specific Q/K/V head cannot learn with only ~5,000 positive entity examples across 97 classes.  
**Recommendation:** Re-run with `freeze_bert=False` for full fine-tuning, or implement a two-stage training strategy (warm-up frozen → unfreeze).

### 3.2 BERT vs CRF Performance Gap
**Issue:** CRF (macro F1 31.4%) vastly outperforms BERT (macro F1 3.4%) on a 97-class ENER task.  
**Root Cause:** Label sparsity. The majority class (O-token) represents 89% of tokens, causing BERT to overfit to O predictions. CRF's transition model explicitly captures BIO structure.  
**Recommendation:** Apply class-weighting in BERT training loss; experiment with entity-only fine-tuning; evaluate seqeval entity-level (not token-level) F1.

### 3.3 CVA Classification
**Issue:** CVA common vectors (SVD-based) achieve only 1.7% accuracy, below mean-vector approach (32.9%).  
**Root Cause:** SVD-based common vectors require sufficient within-class variance to extract meaningful subspaces. With <10 examples per class, the SVD is numerically unstable.  
**Recommendation:** Restrict CVA to classes with ≥10 training examples; fall back to mean vectors for sparse classes.

### 3.4 OOV Experiment
**Issue:** OOV results only report total_oov count; no recall@3 or accuracy metric.  
**Recommendation:** Add recall@1, recall@3, and mean reciprocal rank (MRR) to OOV results.

### 3.5 Contrastive Learning
**Issue:** Contrastive NER script exists but no dedicated results directory or final experiment outputs.  
**Recommendation:** Run `scripts/run_contrastive_ner.py` and generate `results/contrastive_full/`.

### 3.6 Character-Level NER
**Issue:** CharCNN script exists but no results directory, suggesting it has not been run.  
**Recommendation:** Run `scripts/run_char_ner.py` and generate `results/char_ner_full/`.

---

## 4. Scientific Gaps

### 4.1 Entity-Level vs Token-Level Evaluation
The current evaluation uses token-level accuracy and macro F1. A strong thesis requires **entity-level (span-level) evaluation** using seqeval:
- Entity-level Precision, Recall, F1
- Partial match metrics (boundary vs. type accuracy)

### 4.2 Q/K/V Representation Study
The AttentionNER module implements learned Q/K/V projections but never extracts or analyzes the underlying Q/K/V representations from BERT's internal attention layers. The thesis requires:
- Extraction of BERT's native Q, K, V matrices for entity tokens
- Comparison of entity representations: hidden state vs. Q vs. K vs. V
- Class separability analysis (silhouette score, Davies-Bouldin index)

### 4.3 Attention Visualization
No attention heatmaps have been generated. Entity-to-entity and entity-to-context attention patterns are a core contribution for the thesis.

### 4.4 Error Analysis
No systematic false positive / false negative analysis exists. A thesis-quality error analysis should:
- Identify the most common confusion pairs (e.g., LOC_CITY → LOC_PROVINCE)
- Categorize errors (boundary errors vs. type errors)
- Analyze per-label error rates

### 4.5 Statistical Significance
Multi-seed results exist but formal statistical testing (McNemar test, bootstrap) comparing BERT vs. CRF vs. Attention-BERT is not documented.

### 4.6 Related Work Survey
No related_work_review.md exists. A publishable thesis must situate the contribution relative to:
- Standard NER benchmarks (CoNLL-2003, OntoNotes)
- Fine-grained / extended NER (FIGER, FewNERD, Ultra-Fine NER)
- Turkish NER (BOUN NER, WikiANN-TR)
- Attention-based entity learning (LUKE, SpanBERT, EntityBERT)
- Contrastive learning for NER (SimCSE, Contrastive NER)

### 4.7 Thesis Figure Centralization
Publication-quality figures are scattered across `results/` subdirectories. A dedicated `thesis_figures/` directory with consistent styling (font size ≥14pt, 300 DPI, consistent color palette) is required.

---

## 5. Immediate Action Plan

| Priority | Action | Effort |
|----------|--------|--------|
| P1 | Generate structured CV reports from existing JSON | 1h |
| P1 | Generate per-label metrics CSV/MD from existing data | 1h |
| P1 | Generate comparison_results.csv from existing results | 30m |
| P1 | Generate error_analysis_report.md from confusion matrices | 2h |
| P1 | Generate FINAL_THESIS_READINESS_REPORT.md | 1h |
| P2 | Create Q/K/V extraction script | 2h |
| P2 | Create attention heatmap script | 2h |
| P2 | Create thesis_figures/ generator script | 2h |
| P2 | Generate CVA analysis report | 1h |
| P2 | Generate related_work_review.md | 3h |
| P3 | Re-run Attention NER with full fine-tuning | 4h (GPU) |
| P3 | Run contrastive NER experiment | 4h (GPU) |
| P3 | Run char NER experiment | 4h (GPU) |
| P3 | Statistical significance testing | 2h |
