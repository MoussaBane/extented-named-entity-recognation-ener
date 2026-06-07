# Thesis Result Summary

This table summarizes the main evaluation metrics produced by the completed experiments (full-data runs).

| Model / Method | Accuracy (mean ± std) | Macro F1 (mean ± std) | Notes |
|---|---:|---:|---|
| BERT (4-fold CV) | 0.7824 ± 0.0074 | 0.0340 ± 0.0121 | mean support ≈ 6227.75 tokens/fold |
| CRF (4-fold CV) | - | 0.3138 ± 0.0211 | CRF baseline (k-fold) |
| Context-only (eval) | 0.3294 | 0.0931 | evaluation split (support 1855) |
| CVA-only (eval) | 0.0167 | 0.0243 | similarity-based CVA |
| Combined (context + CVA) | 0.0302 | 0.0317 | average scoring ensemble |

**OOV (full eval)**: 235 unseen token surfaces identified in `data/full_eval.conll`.

Files referenced for numbers: `results/cv_full/cv_summary.json`, `results/crf_full/crf_cv_summary.json`, `results/oov_full/oov_summary.json`, `results/compare_full/*/metrics_summary.csv`.

Use this table as a quick insertion into thesis drafts; a LaTeX version is available in `reports/thesis_results_table.tex`.
