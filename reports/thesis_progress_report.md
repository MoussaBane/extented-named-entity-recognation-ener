# Thesis Progress Report

Date: 2026-06-07

1. Current project status

- Core pipeline implemented: CoNLL parsing, corpus statistics, BERT token classification, embedding extraction, CVA, visualization, and evaluation.

2. Completed work (this update)

- Full repository audit and supervisor checklist (`reports/project_audit.md`, `reports/supervisor_requirements_checklist.md`).
- Implemented 4-fold cross-validation runner: `scripts/run_cross_validation.py`.
- Implemented CRF baseline with k-fold evaluation: `scripts/run_crf_baseline.py`.
- Implemented OOV top-3 retrieval experiment: `scripts/oov_experiment.py`.
- Implemented context vs CVA comparison script: `scripts/compare_context_cva.py`.
- Added top-k cosine utility: `ner_stats/cva.py::top_k_cosine_similarities`.
- Added prototype overlay to PCA visualization in `scripts/run_embedding_analysis.py`.
- Added dependency `sklearn-crfsuite` to `requirements.txt`.

3. Newly added features

- Cross-validation orchestration and aggregated metrics.
- CRF baseline for alternative model comparison.
- OOV experiment and top-3 nearest label retrieval.
- Context-only, CVA-only, and combined classifier comparison.

4. Experimental results

- Results will be written under `results/` (see individual script `--output-dir` defaults). Run scripts to generate current numeric outputs.

5. Remaining work

- Integrate CRF confusion matrices side-by-side with BERT and CVA in a single comparison report.
- Add PCA figures with captions and explained variance text tailored for the thesis.
- Create thesis-ready tables (LaTeX/markdown) from aggregated results.

6. Risks and limitations

- The CRF baseline requires `sklearn-crfsuite` and may be slower on large datasets.
- OOV selection uses a conservative surface-based definition; for multi-word entities additional grouping logic may be needed.

7. Suggested thesis contributions

- A reproducible comparison of BERT vs CVA vs CRF on Turkish extended NER, including OOV analysis and prototype-based interpretability.
