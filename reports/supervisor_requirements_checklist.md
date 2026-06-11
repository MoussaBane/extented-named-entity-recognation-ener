# Supervisor Requirements Checklist

This checklist verifies the supervisor's requested features against the current codebase and gives recommended actions.

Requirement A: 4-fold cross validation

- Implemented?: YES
- Location: `scripts/run_cross_validation.py` — full 4-fold BERT CV with deterministic seeds, per-fold model saving, and aggregated summary.
- Recommended actions: None.

Requirement B: Precision / Recall / F1 reporting per fold

- Implemented?: YES
- Location: `scripts/run_cross_validation.py` saves `fold_N/metrics.json` (per_class precision/recall/F1) and `cv_summary.json` with macro mean±std across folds. `scripts/run_crf_baseline.py` does the same for CRF.
- Recommended actions: None.

Requirement C: Confusion matrices for most frequent labels

- Implemented?: YES
- Location: `ner_stats/evaluation_report.py` — `generate_reports()` now calls `save_top_n_confusion_matrix(metrics, top_n=20, out_dir)` to save `confusion_matrix_top20.csv` and `confusion_matrix_top20.png` (top-20 labels by support) alongside the full confusion matrix.
- Recommended actions: None. Adjust `top_n` argument in `generate_reports()` calls if a different count is needed.

Requirement D: BERT baseline

- Implemented?: YES
- Location: `scripts/train_ner.py` (training and evaluation), `requirements.txt` references transformers.
- Recommended actions: None.

Requirement E: Alternative model comparison (CRF preferred)

- Implemented?: YES
- Location: `scripts/run_crf_baseline.py` — CRF with sklearn-crfsuite, same 4-fold split as BERT, saves per-fold metrics and `crf_cv_summary.json`.
- Recommended actions: None.

Requirement F: Context vector extraction

- Implemented?: YES
- Location: `ner_stats/embeddings.py` (TransformerEmbedder), `ner_stats/embedding_analysis.py`, `scripts/run_embedding_analysis.py` (exports `.npy`, `.csv`, `class_vectors.json`).
- Recommended actions: None.

Requirement G: Common Vector Analysis (CVA)

- Implemented?: YES
- Location: `ner_stats/cva.py` (`compute_cva_common_vectors`), used in `scripts/run_embedding_analysis.py` and `scripts/train_ner.py`.
- Recommended actions: None.

Requirement H: Average label vectors from contextual embeddings

- Implemented?: YES
- Location: `ner_stats/cva.py` (`compute_class_mean_vectors`) and `ner_stats/embedding_analysis.py` (`build_class_vectors_from_records` uses mean/CVA selection). `class_vectors.json` produced by `run_embedding_analysis.py`.
- Recommended actions: None.

Requirement I: Unknown entity classification

- Implemented?: YES
- Location: `scripts/oov_experiment.py` — finds entity tokens unseen during training, runs top-3 retrieval against CVA class vectors, saves `oov_results.csv` (sent_idx, token, gold, top1, top1_score, top3) and `oov_summary.json`.
- Recommended actions: None.

Requirement J: Top-3 nearest label prediction

- Implemented?: YES
- Location: `ner_stats/cva.top_k_cosine_similarities(embedding, class_vectors, k=3)` — returns sorted `(label, score)` list. Used in `scripts/oov_experiment.py`.
- Recommended actions: None.

Requirement K: Context-only classification experiment

- Implemented?: YES
- Location: `scripts/compare_context_cva.py` — "context_only" method uses mean class vectors (no CVA projection); produces `compare_dir/context_only/classification_report.json` and per-class metrics.
- Recommended actions: None.

Requirement L: CVA + Context classification experiment

- Implemented?: YES
- Location: `scripts/compare_context_cva.py` — "combined" method averages cosine scores from mean vectors (context) and CVA common vectors; produces `compare_dir/combined/classification_report.json`.
- Recommended actions: None.

Requirement M: Performance comparison between Context-only and CVA+Context

- Implemented?: YES
- Location: `scripts/compare_context_cva.py` produces `comparison_summary.json` with macro P/R/F1/accuracy for context_only, cva_only, and combined. `scripts/generate_thesis_summary.py` formats this as both Markdown and LaTeX tables.
- Recommended actions: None.

Requirement N: PCA visualization of entity embeddings

- Implemented?: YES
- Location: `scripts/run_embedding_analysis.py` — produces PCA 2D/3D, t-SNE, and UMAP plots; `pca_embeddings_with_prototypes.png` overlays CVA/mean prototypes; `pca_explained_variance.txt` now saves explained variance ratios for all components.
- Recommended actions: None.

Requirement O: Thesis-ready tables and figures

- Implemented?: YES
- Location: `scripts/generate_thesis_summary.py` — comprehensive report generator. Accepts `--cv-dir`, `--crf-dir`, `--compare-dir`, `--bert-cva-dir`, `--oov-dir`, `--embed-dir`, `--out-dir`. Produces:
  - `thesis_report.md` — Markdown tables for all experiments (BERT CV, CRF CV, BERT vs CVA, Context/CVA/Combined, top-10 per-class F1, OOV, embedding stats).
  - `thesis_report.tex` — LaTeX `\begin{table}` blocks with `\toprule`/`\midrule`/`\bottomrule` for direct inclusion in a thesis.
- Recommended actions: Run with full-data result dirs:

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

---

Summary: All supervisor requirements (A–O) are now fully implemented. The pipeline covers: 4-fold BERT CV (A, B), top-N frequent-label confusion matrices (C), BERT baseline (D), CRF baseline with 4-fold CV (E), context vector extraction (F), CVA (G, H), OOV/top-3 retrieval experiment (I, J), context-only and CVA+context classifiers with comparison (K, L, M), PCA/t-SNE visualizations with prototype overlay and explained variance (N), and a comprehensive thesis-ready Markdown + LaTeX report generator (O).
