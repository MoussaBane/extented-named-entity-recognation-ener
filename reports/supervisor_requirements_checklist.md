# Supervisor Requirements Checklist

This checklist verifies the supervisor's requested features against the current codebase and gives recommended actions.

Requirement A: 4-fold cross validation
- Implemented?: NO
- Location: N/A
- Recommended actions: Implement a k-fold orchestration module that reuses `scripts/train_ner.py` components (tokenization, dataset, Trainer) to train/evaluate on 4 folds. Save per-fold metrics and aggregated mean/std in `results/cross_validation/`.

Requirement B: Precision / Recall / F1 reporting per fold
- Implemented?: NO
- Location: Existing single-run reporting in `scripts/train_ner.py`, `ner_stats/evaluation.py`, `ner_stats/evaluation_report.py`
- Recommended actions: Extend k-fold runner to call `evaluate_token_classification` per fold and save per-fold CSV/JSON plus aggregated stats (mean/std).

Requirement C: Confusion matrices for most frequent labels
- Implemented?: PARTIAL
- Location: Confusion matrices saved for full label set by `scripts/train_ner.py` and `ner_stats/evaluation_report.py` (e.g., `results/model_comparison_*/confusion_matrix_bert_with_o.csv`).
- Recommended actions: Add utility to compute label frequency (`ner_stats/statistics.py` already has counts) and script to compute confusion restricted to top-N frequent labels (save PNG & CSV).

Requirement D: BERT baseline
- Implemented?: YES
- Location: `scripts/train_ner.py` (training and evaluation), `requirements.txt` references transformers.
- Recommended actions: None — already present. For multi-fold ensure Trainer reuse and deterministic seeds.

Requirement E: Alternative model comparison (CRF preferred)
- Implemented?: NO
- Location: N/A
- Recommended actions: Add CRF baseline using `sklearn-crfsuite` or `sklearn_crfsuite` with same first-subtoken alignment as BERT. Ensure CRF uses same folds and save metrics and confusion matrices in `results/model_comparison_crf/`.

Requirement F: Context vector extraction
- Implemented?: YES
- Location: `ner_stats/embeddings.py`, `ner_stats/embedding_analysis.py`, `scripts/run_embedding_analysis.py` (exports `.npy` and `.csv`).
- Recommended actions: Ensure storage format includes item → (sent_idx, token_idx) mapping for OOV experiments (embedding CSV already contains metadata). Consider adding an HDF5 or memory-mapped `.npy` index for large corpora.

Requirement G: Common Vector Analysis (CVA)
- Implemented?: YES
- Location: `ner_stats/cva.py`, used in `scripts/run_embedding_analysis.py` and `scripts/train_ner.py`.
- Recommended actions: Add top-K retrieval utilities and a documented API for producing common vectors per label (already available as `compute_cva_common_vectors`).

Requirement H: Average label vectors from contextual embeddings
- Implemented?: YES
- Location: `ner_stats/cva.py` (`compute_class_mean_vectors`) and `ner_stats/embedding_analysis.py` (`build_class_vectors_from_records` uses mean/CVA selection).
- Recommended actions: None — ensure these are exported (e.g., `class_vectors.json` is produced by `run_embedding_analysis.py`).

Requirement I: Unknown entity classification
- Implemented?: NO (PARTIAL support)
- Location: Building blocks exist: embedding extraction + CVA classification in `scripts/run_embedding_analysis.py` and `ner_stats/cva.py`.
- Recommended actions: Implement a routine that constructs an evaluation set of entities unseen during training and runs top-3 retrieval; save predictions and evaluation summary under `results/oov_experiment/`.

Requirement J: Top-3 nearest label prediction
- Implemented?: NO (PARTIAL)
- Location: `ner_stats/cva.classify_embedding_by_cosine_similarity` currently returns top-1 only.
- Recommended actions: Add a `top_k_cosine_similarities` utility returning sorted (label, score) pairs; use it in OOV experiments and reporting.

Requirement K: Context-only classification experiment
- Implemented?: PARTIAL
- Location: BERT inference and CVA already implemented individually (`scripts/train_ner.py` compares BERT vs CVA). A dedicated "context-only" (embedding nearest-label without CVA prototypes) experiment is not explicitly implemented.
- Recommended actions: Implement a context-only classifier that compares evaluation embeddings directly to label-average vectors or uses kNN on training embeddings (per-token) and save metrics.

Requirement L: CVA + Context classification experiment
- Implemented?: NO
- Location: N/A
- Recommended actions: Implement a combined decision rule (e.g., weighted score between CVA prototype similarity and context-only classifier or concatenated features into a lightweight classifier). Run on same folds and report comparison.

Requirement M: Performance comparison between Context-only and CVA+Context
- Implemented?: NO
- Location: N/A
- Recommended actions: After implementing K and L, run both experiments on same evaluation splits (or cross-validation folds) and produce comparative tables with accuracy/precision/recall/F1 and statistical summaries.

Requirement N: PCA visualization of entity embeddings
- Implemented?: PARTIAL
- Location: `scripts/run_embedding_analysis.py` produces PCA 2D/3D and t-SNE; `ner_stats/visualization.py` provides plotting functions including `plot_prototypes_2d` to overlay prototypes.
- Recommended actions: Add an automated step to include CVA/mean prototypes in PCA plots and save explained variance text. Ensure figures are labeled and sized for thesis use.

Requirement O: Thesis-ready tables and figures
- Implemented?: PARTIAL
- Location: The repository generates CSV/JSON and PNG outputs (`results/` and `results_example/`) but does not assemble formatted tables or figure captions.
- Recommended actions: Create a `scripts/generate_thesis_reports.py` (or extend `scripts/generate_thesis_summary.py`) to ingest results and produce publication-ready markdown/LaTeX tables and figure files with captions and explained variance text.

---

Summary: The repository contains a solid core: annotation parsing, BERT training, CVA, embedding extraction, evaluation and visualization. Missing items are primarily experimental orchestration (4-fold CV, per-fold reporting), a CRF baseline, explicit OOV/top-K experiments, and a few convenience/reporting utilities to produce thesis-ready artefacts.
