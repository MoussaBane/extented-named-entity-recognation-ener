# Turkish Extended NER — Annotation Statistics & Analysis

This repository provides tools to compute corpus-level statistics, run quality control, and generate visualizations for the Turkish Extended Named Entity Recognition (NER) project. It analyzes INCEpTION/CoNLL annotations to prepare data for downstream tasks such as character-level boundary detection and contrastive learning experiments.

## Key Features

- CoNLL parsing: Read INCEpTION/CoNLL-style annotations from document folders.  
- Corpus statistics: Compute token/sentence counts, entity density, label and type distributions.  
- Tagset QC: Validate annotations against `data/ENER-tagset.tsv` and report unused or undefined tags.  
- Visualization: Produce plots for entity frequency, sentence ratios, and length distributions.  
- Export: Save summaries as JSON and CSV for reporting and further analysis.

## Repository Layout

```
extented-named-entity-recognation-ener/
│
├── data/
│   ├── annotation/           # Input: document subfolders with .conll files
│   └── ENER-tagset.tsv       # Canonical tagset for QC
│
├── ner_stats/                # Library code
│   ├── conll_reader.py       # CoNLL parsing
│   ├── statistics.py         # Metric computation
│   └── tagset.py             # Tagset validation
│
├── scripts/
│   └── run_analysis.py       # CLI entrypoint
│
├── results/                  # Auto-created output directory
│   ├── plots/                # Generated visualizations
│   ├── stats.json            # Global statistics
│   └── *.csv                 # Detailed reports
│
└── requirements.txt
```

## Getting Started

Prerequisites

- Python 3.8+ (3.10 recommended)
- Git

Install

1. Clone the repo and create a virtual environment:

Linux / macOS

```bash
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell)

```powershell
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Basic

```bash
python scripts/run_analysis.py --data-root data/annotation --results-dir results
```

Run as a module (avoids PYTHONPATH issues)

Linux / macOS

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_analysis --data-root data/annotation --results-dir results
```

Windows (PowerShell)

```powershell
$env:PYTHONPATH = "C:\path\to\extented-named-entity-recognation-ener"
python -m scripts.run_analysis --data-root data/annotation --results-dir results
```

## Outputs

After a successful run, the `results/` directory contains:

- stats.json — corpus-level summary (tokens, docs, entity/sentence ratios, distinct types)  
- label_counts.csv — BIO label counts  
- type_counts.csv — aggregated counts by entity type (e.g., PERSON, DATE)  
- plots/ — visualizations such as top20_entity_types.png, entity_sentence_ratio.png, sentence_length_hist.png  
- QC reports: `unused_tags_in_corpus.txt`, `unknown_types_in_tagset_comparison.txt`

## ENER Tagset

The project uses a custom extended tagset defined in `data/ENER-tagset.tsv`, covering 130+ entity types grouped into families such as:

- Locations: LOC_CITY, LOC_COUNTRY, LOC_GEO, LOC_ASTRAL, LOC_ADDRESS  
- Facilities: FAC_AIRPORT, FAC_MUSEUM, FAC_SCHOOL, FAC_RELIGIOUS, FAC_ZOO  
- Organizations: ORG_CORPORATION, ORG_POLITICAL, ORG_SPORTS, ORG_TERROR  
- Professional/Conceptual: PRO_LANGUAGE, PRO_THEORY, PRO_MONEY, PRO_ART, PRO_LAW  
- Temporal/Numeric: DATE, TIME, DURATION, CALORIE, TEMPERATURE, DISTANCE  
- Semantic categories: ANIMAL, PLANT, DISEASE, EVENT, MOVIE, BOOK, SPACE

## Scientific Goals

Primary research goals enabled by this toolkit:

- Character-level boundary detection: convert token-level BIO annotations to exact character offsets and create a binary character-level entity classification task.  
- Contrastive learning & augmentation: mine positive/negative sentence or span pairs and leverage entity-rich sentences and label-rich types to build robust dense representations.

## Contributing

Contributions are welcome:

1. Fork the repository.  
2. Create a feature branch: git checkout -b feature/YourFeature  
3. Commit changes and open a Pull Request.

## License & Contact

- License: add an appropriate license file (e.g., MIT, Apache 2.0).  
- Contact: open an issue for questions about the BAP project or dataset access.
