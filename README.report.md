# Turkish Extended Named Entity Recognition (ENER)

---

## Project Title
**Turkish Extended Named Entity Recognition (ENER)**

---

## 1. Overview

This repository implements a research-grade, reproducible, and extensible pipeline for Turkish Extended Named Entity Recognition (ENER). It couples careful corpus-quality tooling with modeling baselines to support academic experimentation, method comparison, and system-level analysis. The system integrates:

- robust CoNLL / INCEpTION ingestion and BIO validation,
- corpus analytics and visualization,
- a transformer-based token-classifier training flow (Hugging Face Transformers),
- a representation-driven Common Vector Approach (CVA) prototype classifier (SVD-based),
- a reusable, device-aware `TransformerEmbedder` to extract word-aligned embeddings efficiently,
- CLI tools with `--device` support, smoke experiments, and reproducible result exports.

This repository is suitable for academic reproducibility, collaborative research, and as a foundation for follow-up work (contrastive learning, character-level detection, hierarchical taxonomies, and publication-ready experiments).

---

## 2. Key Features

- End-to-end pipeline: ingestion → QC → analysis → modeling → evaluation.
- Robust BIO validation and normalization utilities (warnings and audit logs).
- Corpus analytics: sentence/token counts, entity density, type-frequency, and plotting.
- Transformer token classification: first-subtoken alignment, HF `Trainer`, `seqeval`.
- CVA prototype classification: class mean and SVD-based common vectors with cosine similarity inference.
- Reusable, device-aware `TransformerEmbedder` to avoid repeated model loads and support GPU/CPU execution.
- CLI scripts (`run_analysis.py`, `train_ner.py`) with `--device` flag for production-like runs.
- Outputs organized for reproducible research (JSON, CSV, PNG, confusion matrices).
- Research-first modular structure to make extensions and experiments tractable.

---

## 3. Repository Architecture

The repository follows a modular layered architecture:

- `data/` — raw and canonical dataset artifacts (CoNLL splits and `ENER-tagset.tsv`).
- `ner_stats/` — core library:
  - `conll_reader.py` — file ingestion
  - `data_utils.py` — BIO validation, normalization, label maps
  - `tagset.py` — tagset loading & comparison
  - `statistics.py` — corpus aggregation and plotting
  - `embeddings.py` — `TransformerEmbedder` (device-aware)
  - `cva.py` — prototype vectors, CVA, cosine classifier
  - `evaluation.py` — confusion matrices and token-level metrics
  - `spans.py` — token-to-character span mapping
  - `timing.py` — inference-time measurement
- `scripts/` — orchestrating CLIs:
  - `run_analysis.py` — corpus analytics & QC
  - `train_ner.py` — full BERT + CVA experiment
- `results/` & `outputs/` — experiment artifacts and model checkpoints
- `results_example/` — example outputs for quick inspection
- `requirements.txt` — runtime dependencies

Conceptual data flow:

```text
raw files (data/) -> conll_reader -> data_utils (validate/normalize) -> statistics + tagset QC
                                                                \
                                                                 -> tokenization (HF tokenizer)
                                                                    -> model training (HF Trainer)
                                                                    -> embeddings extraction (TransformerEmbedder)
                                                                    -> CVA prototypes (cva.py)
                                                                    -> evaluation (evaluation.py, seqeval)
```

---

## 4. Scientific Motivation

Fine-grained named entity taxonomies provide higher-value semantic annotations for downstream tasks (relation extraction, knowledge base population, domain-specific IE). However, fine-grained ENER increases label sparsity, annotation complexity, and evaluation demands. This repository is designed to:

1. Provide reproducible quality-control steps to ensure that a canonical tagset aligns with corpus usage.
2. Offer baseline models that capture two complementary paradigms: supervised token classification (fine-tuning transformers) and similarity/prototype-based classification using rich contextual embeddings (CVA).
3. Serve as a platform for research into handling morphological complexity (character-level boundaries), improving prototypes and embeddings (contrastive learning), and designing hierarchical and scalable NER systems.

---

## 5. Turkish NLP Challenges

Turkish poses specific and severe challenges for NER:

- Agglutinative morphology: long words formed by many suffixes that alter meaning and entity membership.
- Tokenization sensitivity: subword tokenizers (WordPiece/BPE) may split entity-bearing stems across subtokens; simple label transfer to subtokens is incorrect without alignment.
- Boundary ambiguity: suffixes and clitic forms may include or exclude parts of entity mentions, requiring precise annotation rules.
- Label sparsity: fine-grained types lead to long-tail distributions; many classes have very few examples.

Consequently, robust token-to-character alignment, character-aware models, prototype/stability techniques, and hierarchical modeling are especially important for Turkish ENER.

---

## 6. Supported ENER Taxonomy

The repository ships with a canonical tagset `data/ENER-tagset.tsv` representing a deep taxonomy (100+ entity subtypes). Examples:

- LOC_CITY, LOC_REGION
- FAC_AIRPORT, FAC_STATION
- ORG_POLITICAL, ORG_FINANCIAL
- PRO_LANGUAGE, PRO_PROGRAM
- DISEASE, EVENT, ACT (activity), AGE, and many more.

`ner_stats/tagset.py` provides utilities to load this tagset, group tags by prefix, and compare the canonical schema against observed types in the corpus (identifying unused or unknown types).

---

## 7. Project Workflow

High-level pipeline stages:

1. Ingest INCEpTION-style folders or flat CoNLL files.
2. Validate BIO sequences with `data_utils.validate_bio_labels` — write warnings and normalized outputs.
3. Generate corpus statistics and plots with `statistics.compute_corpus_statistics`.
4. Train/Load a transformer token-classifier (Hugging Face) with first-subtoken alignment.
5. Extract word-aligned embeddings with the device-aware `TransformerEmbedder`.
6. Compute CVA class vectors (mean or SVD-projected common vectors).
7. Predict with BERT and CVA, compute token- and span-level metrics, generate confusion matrices and timing reports.
8. Save reproducible artifacts (JSON/CSV/PNG) in `results/`.

---

## 8. Installation

Prerequisites:

- Python 3.8+
- CUDA toolkit if using GPU
- Git

Install dependencies:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

or (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes (representative):

- transformers
- torch
- evaluate
- seqeval
- numpy
- pandas
- matplotlib
- accelerate
- protobuf
- sentencepiece

For reproducibility, pin versions in a lockfile or use Poetry.

---

## 9. Environment Setup

- Create a Python virtual environment and install dependencies (see Installation).
- If using GPU, ensure `torch` is installed with CUDA support and `--device cuda` is passed to scripts when running embedding/model extraction.

Example (GPU):

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

---

## 10. Dataset Structure

Minimal structure in `data/`:

```
data/
  annotation/
    article-text-n-1000233810/
      INITIAL_CAS.conll
      admin.conll
    article-text-n-1000233811/
      ...
  ENER-tagset.tsv
  _smoke_train.conll
  _smoke_eval.conll
  full_train.conll
  full_eval.conll
```

- Document folders under `annotation/` contain INCEpTION exports (unified text + per-document annotation CAS).
- `*_smoke*` splits are provided for fast local debugging and CI smoke tests.

---

## 11. INCEpTION / CoNLL Integration

- The `conll_reader.py` parser expects CoNLL rows where the token is the first whitespace column and the final column is the BIO tag.
- Blank lines separate sentences.
- The pipeline prefers `admin.conll` as the authoritative annotated file for corpus statistics; `INITIAL_CAS.conll` exists as the unannotated baseline.

---

## 12. Corpus Analysis Pipeline

Run corpus analytics:

```bash
python scripts/run_analysis.py \
  --data-root data/annotation \
  --results-dir results
```

Outputs:

- `results/stats.json`
- `results/label_counts.csv`
- `results/type_counts.csv`
- `results/plots/top20_entity_types.png`
- `results/plots/entity_sentence_ratio.png`
- `results/plots/sentence_length_hist.png`
- `results/unused_tags_in_corpus.txt`
- `results/unknown_types_in_tagset_comparison.txt`

`statistics.py` computes:

- total document folders
- annotated/unannotated document counts
- total sentences and tokens
- sentences with at least one entity and entity-sentence ratio
- average sentence length
- label and type frequency counters

---

## 13. Transformer-Based NER Training

`train_ner.py` implements a full Transformer training + evaluation flow:

- Tokenization uses `AutoTokenizer` with `is_split_into_words=True`.
- Label alignment uses first-subtoken alignment: only the first subtoken receives the label; subsequent subtokens are masked with `-100`.
- Model: `AutoModelForTokenClassification` (configurable `--model-name`), fine-tuned using Hugging Face `Trainer`.
- Sequence metrics: `seqeval` (entity-level precision/recall/F1) is used during training, and token-level metrics/confusion matrices are computed offline.
- Model checkpointing uses `Trainer` conventions and can be reused by setting `--skip-bert-train` and pointing `--output-dir` to a HF-compatible checkpoint.

Example training command (smoke run):

```bash
python scripts/train_ner.py \
  --train-file data/_smoke_train.conll \
  --eval-file data/_smoke_eval.conll \
  --device cuda \
  --output-dir outputs/bert-ner-smoke \
  --results-dir results/model_comparison_smoke
```

Key training details:

- Default model: `dbmdz/bert-base-turkish-cased`
- Default learning rate: `2e-5`
- Batch sizes and epoch count are configurable via CLI arguments
- BIO warnings are written to `results/` before and after normalization

---

## 14. CVA Prototype Classification

The CVA baseline is a representation-based classifier that uses embeddings extracted from the same transformer backbone to construct class prototypes and classify by cosine similarity.

Rationale: CVA provides a lightweight baseline that tests how much the embedding geometry alone (without task-specific fine-tuning) can separate extended entity classes. It is especially relevant for low-resource and long-tail regimes.

Pipeline steps:

1. Extract word-aligned embeddings for all training tokens (first-subtoken embeddings).
2. Group embeddings by class label to create `X_c ∈ ℝ^{n_c × d}` for each class `c`.
3. Compute class prototype:
   - Mean vector: simple prototype
   - CVA common vector: remove the within-class subspace components (SVD projection removal) to isolate the "core" direction for the class
4. Classify evaluation token embeddings by cosine similarity to class prototypes.

Mathematics:

- Cosine similarity:

```math
\mathrm{cosine}(a,b)=\frac{a^\top b}{\|a\|\|b\|}
```

- SVD decomposition for centered class matrix \( A \):

```math
A = U \Sigma V^\top
```

- Given class mean \( \bar{x} \), remove projection onto within-class subspace \( V_r \):

```math
p = V_r V_r^\top \bar{x}
\quad
\text{common_vector} = \bar{x} - p
```

Fallback: if SVD indicates rank 0 or numeric instability, fallback to the mean vector.

Benefits:

- Simple to compute for small-to-medium datasets.
- Interpretable prototypes and fast inference (cosine lookups).
- Provides a baseline complementary to supervised fine-tuning.

Limitations:

- Per-class SVD can be expensive at scale.
- Multi-modal class distributions require multiple prototypes or mixture models.
- Rare classes may produce noisy prototypes.

---

## 15. Device-Aware Embedding Refactor

To address repeated transformer loads and improve GPU utilization, we implemented a reusable, device-aware `TransformerEmbedder` (`ner_stats/embeddings.py`).

Design goals:

- Single point to instantiate `AutoTokenizer` and `AutoModel` and place the model on a specified device (`cpu` or `cuda`).
- Provide `encode_sentence_subwords()` and `encode_words()` that return embeddings on the embedder's device.
- Allow callers to pass an `embedder` instance to CVA and timing functions to avoid reloads.
- Provide backward-compatible helpers that create temporary embedders if none supplied.

Why this matters:

- Reusing a single embedder eliminates repeated I/O and memory overhead of loading the same HF model multiple times.
- Device-aware placement reduces host-device transfers and significantly improves throughput when using GPU.
- Ease of experiment reproducibility: embedding extraction is deterministic given seed and model checkpoint.

Usage (example):

```python
from ner_stats.embeddings import TransformerEmbedder

embedder = TransformerEmbedder(model_name="dbmdz/bert-base-turkish-cased", device="cuda")
tokens, embeddings = embedder.encode_sentence_subwords("Ankara havalimanı açık.")
```

---

## 16. Evaluation Metrics

- Sequence-level: `seqeval` metrics (entity-level precision, recall, F1).
- Token-level: confusion matrix, per-class precision/recall/F1, accuracy (computed by `ner_stats/evaluation.py`).
- Timing: per-sentence average and total inference time measured for BERT and CVA (warmup included).
- Reporting: results exported as JSON (metrics) and CSV (confusion matrices), enabling further table generation and statistical aggregation.

---

## 17. Example Commands

Corpus analysis:

```bash
python scripts/run_analysis.py \
  --data-root data/annotation \
  --results-dir results \
  --device cpu
```

Transformer NER training (smoke):

```bash
python scripts/train_ner.py \
  --train-file data/_smoke_train.conll \
  --eval-file data/_smoke_eval.conll \
  --device cuda \
  --output-dir outputs/bert-ner-smoke
```

Re-use a pretrained checkpoint (skip training):

```bash
python scripts/train_ner.py \
  --train-file data/full_train.conll \
  --eval-file data/full_eval.conll \
  --skip-bert-train \
  --output-dir outputs/bert-ner-full \
  --device cuda \
  --results-dir results/model_comparison_full
```

---

## 18. Repository Structure Tree

```text
extended-named-entity-recognition-ener/
├── data/                       # annotation folders, tagset, train/eval splits
├── ner_stats/                  # core library: parsers, utils, embeddings, CVA, evaluation
│   ├── __init__.py
│   ├── conll_reader.py
│   ├── data_utils.py
│   ├── tagset.py
│   ├── statistics.py
│   ├── embeddings.py          # device-aware TransformerEmbedder
│   ├── cva.py                 # CVA, prototypes, cosine classifier
│   ├── evaluation.py
│   ├── spans.py
│   └── timing.py
├── scripts/
│   ├── run_analysis.py         # corpus analytics + QC
│   └── train_ner.py            # end-to-end BERT + CVA pipeline
├── outputs/                    # model checkpoints / Hugging Face-like folders
├── results/                    # generated stats, plots, metrics, confusion matrices
├── results_example/            # example outputs
├── requirements.txt
└── README.report.md            # this file
```

Each folder’s role:

- `data/` — canonical data and smoke splits for quick iteration.
- `ner_stats/` — the research library; intended for import in scripts and experiments.
- `scripts/` — user-facing entry points to recreate experiments.
- `outputs/` — model artifacts that can be uploaded to HF Hub.
- `results/` — deterministic experiment outputs for reporting and figures.

---

## 19. Mathematical Foundations

### BIO tagging

BIO encodes entity span boundaries and type in token-level labels. For tokens \( x_1, \ldots, x_n \), labels \( y_i \in \{O\} \cup \{B\text{-}t, I\text{-}t : t \in \mathcal{T}\} \) where \( \mathcal{T} \) is the taxonomy. Valid transitions must obey BIO constraints to represent consistent spans.

### Transformer token classification

For subtoken embeddings \( h_i \in \mathbb{R}^d \), classification logits:

```math
z_i = W h_i + b
```

Softmax over \( z_i \) yields per-token distributions; cross-entropy loss is applied at first-subtoken positions only.

### Cosine similarity classification

Given prototype \( v_c \) and query embedding \( q \):

```math
\mathrm{cosine}(q, v_c) = \frac{q^\top v_c}{\|q\| \|v_c\|}
```

Prediction: \( \hat{c} = \arg\max_c \mathrm{cosine}(q, v_c) \).

### SVD and CVA

Given centered class matrix \( A \in \mathbb{R}^{n \times d} \), its compact SVD:

```math
A = U \Sigma V^\top
```

Let \( V_r \) be top r right-singular vectors. The within-class subspace projection of mean \( \bar{x} \) is \( p = V_r V_r^\top \bar{x} \). The CVA common vector is \( c = \bar{x} - p \).

---

## 20. Research Roadmap

The repository is designed to support the following research items (short descriptions):

- **Multi-seed experiments & statistical testing**: run training across multiple seeds, aggregate mean±std and confidence intervals, and perform significance tests (paired bootstrap / t-test) between BERT and CVA methods.
- **Prototype robustness**: evaluate mean vs. CVA prototypes, prototype shrinkage, and mixture-of-prototypes for multi-modal classes.
- **Contrastive learning**: pretrain encoder with supervised or unsupervised contrastive objectives to improve prototype separability, especially for long-tail classes.
- **Hierarchical classification**: exploit tagset prefixes to implement coarse→fine pipelines (e.g., predict FAC vs. LOC first, then subtype).
- **Character-level boundary detection**: implement hybrid token+character models to improve entity boundary detection in agglutinative Turkish.
- **Hugging Face dataset & model publishing**: standardize dataset metadata, produce dataset card, upload models and training scripts to HF Hub for community reuse.

---

## 21. Performance and Scalability Notes

- **Embedding reuse**: Reusing a single `TransformerEmbedder` instance substantially reduces memory footprint and eliminates repeated model initialization latency.
- **Device-aware execution**: Placing model and inputs on the same device avoids host-device synchronization costs and improves throughput for extraction and inference.
- **CVA SVD cost**: Per-class SVD is \( O(\min(n d^2, n^2 d)) \). For large classes or high dimensionality, use randomized/truncated SVD (`sklearn.utils.extmath.randomized_svd`) or incremental PCA.
- **Batching & caching**: Embedder methods can be extended to process batched inputs for greater throughput; cache prototypes to disk (`np.save`) to avoid recomputation across experiments.
- **Recommended hardware**: For full experiments, a GPU with ≥16GB memory is recommended. For large CVA computations, a machine with large RAM (64GB+) or batched/distributed SVD is beneficial.

---

## 22. Future Work

Short-term (engineering):

- Add unit tests (BIO normalization, span alignment, embedding APIs).
- Create CI (GitHub Actions) with smoke-run tests using `_smoke_*` splits.
- Add dependency lockfile for environment reproducibility.

Medium-term (research):

- Contrastive pretraining for entity-aware embeddings.
- Hierarchical classification experiments using tagset ontology.
- Character-level and span-based models to compare token-level BIO performance.

Long-term (publication & deployment):

- Provide HF Dataset and Model Hub artifacts with dataset card and model card.
- Add experiment tracking integration (MLflow / Weights & Biases).
- Develop a lightweight web UI for annotation QA and human-in-the-loop correction.

---

## 23. Citation

If you use this code or dataset in research, please cite the repository and include a reference to the ENER dataset and this pipeline. (Add your preferred citation block here when a DOI or formal reference is available.)

---

## 24. License

No license is included in the current repository snapshot. Before public distribution, add an appropriate license (MIT, Apache-2.0, or similar) consistent with dataset permissions and institutional requirements.

---

## 25. Acknowledgements

This repository was developed as a research platform for Turkish ENER experimentation. Acknowledgements should be added for dataset contributors, annotators, and funding entities when known.

---

## 26. Appendix: Future Roadmap, CI/testing plans, and Research Roadmaps (detailed)

### CI / Testing Plans

- Add `tests/` with unit tests for:
  - `data_utils.normalize_bio_sequence` (edge cases, malformed BIO sequences)
  - `spans.bio_to_character_spans` (alignment tests with repeated tokens and punctuation)
  - `embeddings.TransformerEmbedder` (device placement and batched call behavior)
- GitHub Actions pipeline:
  - `lint` (flake8/black)
  - `unit-tests` (run `pytest` with `_smoke_*` artifacts)
  - `smoke-run` (run `scripts/run_analysis.py` and `scripts/train_ner.py` with smoke arguments)

### Contrastive Learning Roadmap

- Implement supervised contrastive loss on the training set where positive pairs are same-class mentions (augmented) and negatives sampled across classes.
- Investigate augmentations: morphological suffix perturbation, back-translation, random token masking.
- Evaluate improvements in CVA prototype separability (cluster metrics: silhouette, Davies-Bouldin).

### Hierarchical Classification Roadmap

- Build a parent-child ontology from `ENER-tagset.tsv` (prefix → children).
- Implement coarse classifier for prefixes (e.g., `FAC` vs `LOC`) and conditional fine-grained classifier for subtypes.
- Explore multi-task learning to predict coarse and fine labels jointly with shared encoder and separate heads.

### Character-Level Modeling Roadmap

- Build a char-level encoder (CNN/BiLSTM) that supplies features concatenated with transformer token embeddings.
- Implement pure character-span detector as an alternative: predict start/end indices and classify span labels.
- Compare boundary recall and micro/macro F1 across token-level BIO baseline and char+token hybrids.

### Hugging Face Dataset/Model Publishing Plans

- Create a `dataset_card` and `dataset_infos.json` aligning to HF Dataset format.
- Export final training/evaluation splits as standardized JSON/CoNLL.
- Publish best-performing models (checkpoint + tokenizer) to HF Hub with a `model_card` describing training regime, dataset provenance, and evaluation metrics.

---

## Contact / Contribution

Contributions are welcome. Please open issues or pull requests to propose improvements, bug fixes, or research experiments. For questions about annotation policy, dataset provenance, or experimental design, please open an issue in the repository so we can track discussion and reproducibility.

---

This README.report.md is crafted to serve both as a user manual for reproducing experiments and a research-oriented description suitable for academic portfolios and collaboration.
