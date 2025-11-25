# Turkish Extended NER – Annotation Statistics

This repository provides corpus-level statistics and visualizations for the project:

> **“Türkçe İçin Derin Öğrenme Tabanlı Genişletilmiş Adlandırılmış Varlık Tanıma (Extended NER) Sistemi Geliştirilmesi”**

The main goal is to analyze INCEpTION-based CoNLL annotations for Turkish Extended Named Entities and to prepare the data for **character-level boundary detection** and **contrastive learning** experiments within a BAP project.

---

## 1. Folder Structure

```text
turkish-extended-ner-stats/
│
├── data/
│   └── annotation/
│       ├── article-text-n-XXXXXXXXXX/
│       │   ├── admin.conll
│       │   └── INITIAL_CAS.conll
│       └── ...
│
├── ner_stats/
│   ├── __init__.py
│   ├── conll_reader.py
│   └── statistics.py
│
├── scripts/
│   └── run_analysis.py
│
├── results/
│   ├── stats.json
│   ├── label_counts.csv
│   ├── type_counts.csv
│   └── plots/
│       ├── top20_entity_types.png
│       ├── entity_sentence_ratio.png
│       └── sentence_length_hist.png
│
├── README.md
└── requirements.txt
```

## 2. Installation

git clone https://github.com/<your-username>/turkish-extended-ner-stats.git
cd turkish-extended-ner-stats

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt

## 3. Usage

Run the analysis script:

python scripts/run_analysis.py \
    --data-root data/annotation \
    --results-dir results

This will:

    Scan all document folders under data/annotation

    Use only admin.conll files for NER statistics

    Compute corpus-level statistics

    Save numeric results as:

        results/stats.json

        results/label_counts.csv

        results/type_counts.csv

    Generate plots into results/plots/:

        top20_entity_types.png

        entity_sentence_ratio.png

        sentence_length_hist.png

## 4. Current Corpus Snapshot (Example)

    These numbers correspond to the current snapshot of the annotated subset and will automatically update as more documents are annotated.

Total document folders: 130

Annotated documents (with admin.conll): 34

Unannotated documents: 96

Total sentences (annotated docs): 1,142

Sentences with at least one entity: 980

Entity sentence ratio: ≈ 0.858 (≈ 85.8%)

Total tokens: 29,195

# BIO labels: 161

# distinct entity types: 97

Top entity types (by frequency):

PERSON, DATE, PRO_THEORY, PRO_LANGUAGE, PRO_CLASS, COUNT, PERIOD, NATION, YEAR, ELEMENT, …

## 5. Extended Tagset (ENER)

The project uses a custom extended named entity tagset for Turkish.

- Tagset file: `data/ENER-tagset.tsv`
- Number of entity types: **130** (current version)
- Tags are grouped into several families:

**Location-related entities**

- `LOC_CITY`, `LOC_COUNTRY`, `LOC_REGION`, `LOC_PROVINCE`,  
  `LOC_ADDRESS`, `LOC_GEO`, `LOC_ASTRAL`, `LOC_DIV`

**Facilities and infrastructures**

- `FAC_AIRPORT`, `FAC_PARK`, `FAC_MUSEUM`, `FAC_PORT`,  
  `FAC_SCHOOL`, `FAC_RELIGIOUS`, `FAC_THEATER`, `FAC_STATION`, `FAC_ZOO`, …

**Organizations**

- `PER`, `ORG`, `ORG_CORPORATION`, `ORG_POLITICAL`,  
  `ORG_SPORTS`, `ORG_TERROR`, `ORG_ETHNIC`, `ORG_FAMILY`

**Professional / domain-specific entities (PRO_*)**

- `PRO_LANGUAGE`, `PRO_THEORY`, `PRO_CLASS`, `PRO_FOOD`, `PRO_DRUG`,  
  `PRO_MONEY`, `PRO_STOCK`, `PRO_ART`, `PRO_AWARD`, `PRO_RELIGION`,  
  `PRO_VEHICLE`, `PRO_WEAPON`, `PRO_LAW`, `PRO_PLAN`, …

**Temporal and numeric entities**

- `DATE`, `YEAR`, `TIME`, `PERIOD`, `DURATION`,  
  `COUNT`, `PERCENT`, `CALORIE`, `TEMPERATURE`, `DISTANCE`, `AREA`, `VOLUME`, `WEIGHT`, `SPEED`, …

**Other semantic categories** (examples)

- Living things and biology: `ANIMAL`, `BIRD`, `INSECT`, `PLANT`, `AMPHIBIA`, `DISEASE`, `DRUG`, `ELEMENT`, …
- Events and media: `EVENT`, `WAR`, `CONFERENCE`, `TV-PROGRAM`, `MOVIE`, `BOOK`, `TITLE`, …
- Geo/physical: `COORDINATE`, `SPACE`, `LOCATION`, `PHYSICAL_EXTEND`, …

The full tagset is stored as a machine-readable file and can be inspected or extended via `ner_stats/tagset.py`.
A simple quality-check pipeline compares the tagset with the corpus and reports:

- tags that are **defined but never used** in the current annotations
- entity types that **appear in the corpus but are not defined** in the tagset

These statistics are saved to:

- `results/unused_tags_in_corpus.txt`
- `results/unknown_types_in_tagset_comparison.txt`


## 6. Contrastive Learning & Character-Level Boundary Detection

These statistics are intended as a first step towards:

1. Character-level boundary detection

    Using token-level BIO tags to derive character-level spans

    Defining a boundary detection task over entity vs. non-entity characters

2. Contrastive learning for data augmentation

    Mining positive/negative pairs at sentence or span level

    Leveraging high-coverage entity sentences (≈ 86%)

    Exploring label-rich entity types (e.g., PERSON, DATE, PRO_THEORY, PRO_LANGUAGE) for more robust representations

Future work in this repository may include:

    Scripts to convert token-level BIO tags to character-level boundary labels

    Contrastive learning-ready datasets (anchor / positive / negative triples)

    Baseline models implemented in PyTorch (e.g., BiLSTM-CRF, BERT-based encoders)
