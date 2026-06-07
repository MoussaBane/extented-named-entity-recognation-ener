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

- Full 4-fold BERT cross-validation is complete on `data/full_train.conll` with mean accuracy `0.7824 ± 0.0074` and mean macro F1 `0.0340 ± 0.0121`.
- Full 4-fold CRF cross-validation is complete with mean macro F1 `0.3138 ± 0.0211`.
- The full OOV experiment completed on `data/full_eval.conll` and identified `235` unseen token surfaces.
- The full context-vs-CVA comparison completed on `data/full_eval.conll`:
	- context-only: accuracy `0.3294`, macro F1 `0.0931`
	- CVA-only: accuracy `0.0167`, macro F1 `0.0243`
	- combined: accuracy `0.0302`, macro F1 `0.0317`

5. Remaining work

- Assemble the final thesis tables and figures from the completed outputs in `results/`.
- Decide whether the manuscript should emphasize the CRF baseline or the context-only similarity model as the main non-BERT comparator.
- Add short explanatory captions for the final figures and metric tables.

6. Risks and limitations

- The CRF baseline requires `sklearn-crfsuite` and may be slower on large datasets.
- OOV selection uses a conservative surface-based definition; for multi-word entities additional grouping logic may be needed.
- The full comparison scripts run on the same transformer backbone as the BERT baseline, so runtime is dominated by embedding extraction rather than the scoring step.

7. Suggested thesis contributions

- A reproducible comparison of BERT vs CVA vs CRF on Turkish extended NER, including OOV analysis and prototype-based interpretability.
