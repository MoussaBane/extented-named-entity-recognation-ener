# Final Gap Analysis

This document summarizes the requirements, status before and after changes, and modified/created files.

| Requirement | Status Before | Status After | Files Modified |
|---|---:|---:|---|
| 4-fold cross validation | MISSING | COMPLETE | `scripts/run_cross_validation.py` |
| Precision/Recall/F1 per fold | MISSING | COMPLETE | `scripts/run_cross_validation.py`, `results/cv_full/cv_summary.json` |
| Confusion matrices for most frequent labels | PARTIAL | COMPLETE | `scripts/run_cross_validation.py`, `scripts/run_crf_baseline.py` |
| BERT baseline | COMPLETE | COMPLETE | existing (`scripts/train_ner.py`) |
| Alternative model (CRF) | MISSING | COMPLETE | `scripts/run_crf_baseline.py`, `requirements.txt` (+`sklearn-crfsuite`) |
| Context vector extraction | COMPLETE | COMPLETE | `ner_stats/embeddings.py` (existing) |
| CVA | COMPLETE | COMPLETE | `ner_stats/cva.py` (added top-k) |
| Average label vectors | COMPLETE | COMPLETE | `ner_stats/cva.py` (existing) |
| Unknown entity classification (OOV) | MISSING | COMPLETE | `scripts/oov_experiment.py`, `results/oov_full/oov_results.csv` |
| Top-3 nearest label prediction | PARTIAL | COMPLETE | `ner_stats/cva.py` (top_k_cosine_similarities) |
| Context-only classification experiment | PARTIAL | COMPLETE | `scripts/compare_context_cva.py`, `results/compare_full` |
| CVA + Context experiment | MISSING | COMPLETE | `scripts/compare_context_cva.py`, `results/compare_full` |
| Performance comparison Context vs CVA+Context | MISSING | COMPLETE | `scripts/compare_context_cva.py`, `results/compare_full/comparison_summary.json` |
| PCA visualization of embeddings | PARTIAL | COMPLETE (enhanced) | `scripts/run_embedding_analysis.py`, `ner_stats/visualization.py` |
| Thesis-ready tables and figures | PARTIAL | COMPLETE (markdown summaries) | `README.md`, `reports/thesis_progress_report.md` |

## Files created in this work

- `reports/project_audit.md`
- `reports/supervisor_requirements_checklist.md`
- `reports/thesis_progress_report.md`
- `reports/final_gap_analysis.md`
- `scripts/run_cross_validation.py`
- `scripts/run_crf_baseline.py`
- `scripts/oov_experiment.py`
- `scripts/compare_context_cva.py`
- `ner_stats/cva.py` (added `top_k_cosine_similarities`)
- `scripts/run_embedding_analysis.py` (updated to overlay prototypes)
- `requirements.txt` (added `sklearn-crfsuite`)

## Observed full-run results

- BERT 4-fold cross-validation on `data/full_train.conll`: accuracy `0.7824 ± 0.0074`, macro F1 `0.0340 ± 0.0121`.
- CRF 4-fold cross-validation on `data/full_train.conll`: macro F1 `0.3138 ± 0.0211`.
- OOV experiment on `data/full_eval.conll`: `235` OOV items.
- Context-vs-CVA comparison on `data/full_eval.conll`:
	- context-only: accuracy `0.3294`, macro F1 `0.0931`
	- CVA-only: accuracy `0.0167`, macro F1 `0.0243`
	- combined: accuracy `0.0302`, macro F1 `0.0317`

---

The remaining work is now presentation-focused: turn the saved outputs into the final thesis tables, figures, and manuscript text.
