# Entity Prototype Vector Analysis

Prototype vectors are computed as `prototype(label) = mean(entity_embeddings)` over all BERT (layer 11) token embeddings belonging to entities of that label, across the training set. Stored in `prototype_vectors.pkl` (96 labels, 768-dim, BIO prefix stripped). Source data: `results/embedding_full/class_vectors.json`.

**Total prototypes:** 96

## Cosine Similarity — Supervisor-Requested Focus Labels

PERSON, ORG, DATE, EVENT, DISEASE, LOC_CITY, LOC_COUNTRY

| Label | PERSON | ORG | DATE | EVENT | DISEASE | LOC_CITY | LOC_COUNTRY |
|---|---|---|---|---|---|---|---|
| PERSON | 1.000 | 0.434 | 0.442 | 0.411 | 0.333 | 0.448 | 0.448 |
| ORG | 0.434 | 1.000 | 0.503 | 0.660 | 0.622 | 0.640 | 0.634 |
| DATE | 0.442 | 0.503 | 1.000 | 0.526 | 0.456 | 0.537 | 0.558 |
| EVENT | 0.411 | 0.660 | 0.526 | 1.000 | 0.648 | 0.605 | 0.611 |
| DISEASE | 0.333 | 0.622 | 0.456 | 0.648 | 1.000 | 0.542 | 0.570 |
| LOC_CITY | 0.448 | 0.640 | 0.537 | 0.605 | 0.542 | 1.000 | 0.638 |
| LOC_COUNTRY | 0.448 | 0.634 | 0.558 | 0.611 | 0.570 | 0.638 | 1.000 |

## Nearest-Neighbor Prototype per Label (all 96 labels)

Each prototype's most cosine-similar *other* prototype. High similarity (>0.85) between semantically distinct labels indicates that mean BERT-layer-11 embeddings under-discriminate those classes — a likely driver of the low macro-F1 reported for fine-tuned BERT.

| Label | Support (tokens) | Nearest Neighbor | Cosine Sim |
|---|---|---|---|
| AREA | 28 | HEIGHT | 0.823 |
| HEIGHT | 12 | AREA | 0.823 |
| DISTANCE | 55 | HEIGHT | 0.804 |
| PRO_VEHICLE | 24 | PRO_WEAPON | 0.785 |
| PRO_WEAPON | 8 | PRO_VEHICLE | 0.785 |
| PERCENT | 51 | AREA | 0.757 |
| POPULATION | 42 | AREA | 0.744 |
| LOC_PROVINCE | 36 | LOC_REGION | 0.742 |
| LOC_REGION | 80 | LOC_PROVINCE | 0.742 |
| TEMPERATURE | 15 | AREA | 0.741 |
| ORG_ETHNIC | 31 | LOC_PROVINCE | 0.734 |
| GOD | 7 | PRO_RELIGION | 0.728 |
| PRO_RELIGION | 29 | GOD | 0.728 |
| FREQUENCY | 21 | AREA | 0.728 |
| AGE | 7 | DURATION | 0.725 |
| DURATION | 18 | AGE | 0.725 |
| PRO_AWARD | 29 | TEXT | 0.724 |
| TEXT | 39 | PRO_AWARD | 0.724 |
| ACT | 36 | PRO_AWARD | 0.721 |
| VOLUME | 2 | AREA | 0.720 |
| TITLE | 80 | LOC_PROVINCE | 0.718 |
| FESTIVAL | 11 | PRO_AWARD | 0.718 |
| DISEASE | 17 | FREQUENCY | 0.718 |
| POSITION_ROLE | 98 | TEXT | 0.718 |
| ORG_CORPORATION | 57 | POSITION_ROLE | 0.717 |
| ORDINAL | 10 | FREQUENCY | 0.716 |
| MEASUREMENT | 130 | AREA | 0.715 |
| FAC_PARK | 2 | FAC_STATION | 0.714 |
| FAC_STATION | 2 | FAC_PARK | 0.714 |
| ARCHITECTURE | 65 | LOC_PROVINCE | 0.710 |
| PRO_OFFENCE | 51 | DISEASE | 0.710 |
| ORG | 89 | ORG_CORPORATION | 0.710 |
| PRO_MOVEMENT | 28 | PRO_AWARD | 0.709 |
| PRO_SPORT | 8 | PRO_VEHICLE | 0.706 |
| PHYSICAL_EXTEND | 24 | AREA | 0.706 |
| LOC_GEO | 70 | LOC_REGION | 0.705 |
| EVENT | 77 | POSITION_ROLE | 0.704 |
| AGREEMENT | 3 | DISEASE | 0.703 |
| PRO_MATERIAL | 111 | TEXT | 0.700 |
| LOC_ADDRESS | 44 | LOC_PROVINCE | 0.699 |
| TIME | 37 | POPULATION | 0.699 |
| FAC_MUSEUM | 4 | LINE_ROAD | 0.699 |
| LINE_ROAD | 2 | FAC_MUSEUM | 0.699 |
| COMPOUND | 15 | DISEASE | 0.698 |
| WAR | 43 | AGREEMENT | 0.697 |
| FAC_SCHOOL | 85 | LOC_PROVINCE | 0.696 |
| PRO_CULTURE | 65 | LOC_PROVINCE | 0.694 |
| COLOR | 2 | COMPOUND | 0.694 |
| PRO_RULE | 150 | POSITION_ROLE | 0.693 |
| TV-PROGRAM | 58 | PRO_CULTURE | 0.693 |
| LINE_BRIDGE | 1 | FAC_MUSEUM | 0.690 |
| MONUMENT | 20 | LINE_ROAD | 0.689 |
| DISASTER | 14 | DISEASE | 0.688 |
| MONEY | 17 | POPULATION | 0.686 |
| PRO_SERVICE | 107 | PRO_CULTURE | 0.685 |
| PRO_STYLE | 6 | PRO_VEHICLE | 0.684 |
| FAC_RELIGIOUS | 7 | LOC_PROVINCE | 0.684 |
| KINGDOM | 8 | LOC_PROVINCE | 0.676 |
| ERA | 16 | TIME | 0.676 |
| INTENSITY | 2 | TEMPERATURE | 0.675 |
| FAC_AIRPORT | 3 | FAC_MUSEUM | 0.671 |
| LOC_COUNTRY | 189 | LOC_REGION | 0.668 |
| LOC_CITY | 170 | LOC_PROVINCE | 0.666 |
| LOC | 192 | EVENT | 0.663 |
| PERIOD | 254 | YEAR | 0.663 |
| YEAR | 223 | PERIOD | 0.663 |
| ORG_FAMILY | 6 | ORG_ETHNIC | 0.662 |
| ELEMENT | 221 | PRO_FOOD | 0.660 |
| PRO_FOOD | 134 | ELEMENT | 0.660 |
| COUNT | 259 | POPULATION | 0.658 |
| PRO_ART | 57 | TV-PROGRAM | 0.654 |
| COORDINATE | 51 | DISTANCE | 0.653 |
| ORG_POLITICAL | 192 | ORG_CORPORATION | 0.653 |
| PER | 118 | TITLE | 0.651 |
| LOC_ASTRAL | 27 | ORG_ETHNIC | 0.650 |
| PRO_CLASS | 297 | POSITION_ROLE | 0.648 |
| LIST | 36 | PRO_AWARD | 0.642 |
| OTH | 107 | PERCENT | 0.640 |
| SEISMIC | 7 | TEMPERATURE | 0.638 |
| PRO_PLAN | 10 | ORG | 0.637 |
| CONFERENCE | 2 | WAR | 0.632 |
| PRO_MAGAZINE | 1 | CONFERENCE | 0.619 |
| DATE | 433 | YEAR | 0.615 |
| NATION | 244 | ORG_POLITICAL | 0.614 |
| PRO_THEORY | 402 | PRO_CLASS | 0.609 |
| SPACE | 1 | DISASTER | 0.606 |
| PRO_CLOTHING | 1 | PRO_WEAPON | 0.602 |
| MINERAL | 2 | COMPOUND | 0.593 |
| PRO_LANGUAGE | 311 | LOC_CITY | 0.591 |
| PRO_LAW | 1 | DISEASE | 0.577 |
| SCHOOL_AGE | 2 | AGE | 0.569 |
| ORG_TERROR | 21 | ORG_FAMILY | 0.561 |
| FLORA | 2 | SPACE | 0.558 |
| PRO_ID | 4 | TEMPERATURE | 0.537 |
| PERSON | 717 | PRO_LANGUAGE | 0.491 |
| LOC_DIV | 4 | LOC_REGION | 0.480 |

## Observations

- 0 / 96 labels have a nearest-neighbor cosine similarity above 0.85, indicating substantial overlap in mean-embedding space.
- 29 labels have fewer than 10 supporting tokens, making their prototypes statistically unstable (consistent with the SVD instability noted in `results/cva_report.md`).

## Focus-Label Pairwise Similarity (sorted)

| Pair | Cosine Sim |
|---|---|
| ORG <-> EVENT | 0.660 |
| EVENT <-> DISEASE | 0.648 |
| ORG <-> LOC_CITY | 0.640 |
| LOC_CITY <-> LOC_COUNTRY | 0.638 |
| ORG <-> LOC_COUNTRY | 0.634 |
| ORG <-> DISEASE | 0.622 |
| EVENT <-> LOC_COUNTRY | 0.611 |
| EVENT <-> LOC_CITY | 0.605 |
| DISEASE <-> LOC_COUNTRY | 0.570 |
| DATE <-> LOC_COUNTRY | 0.558 |
| DISEASE <-> LOC_CITY | 0.542 |
| DATE <-> LOC_CITY | 0.537 |
| DATE <-> EVENT | 0.526 |
| ORG <-> DATE | 0.503 |
| DATE <-> DISEASE | 0.456 |
| PERSON <-> LOC_CITY | 0.448 |
| PERSON <-> LOC_COUNTRY | 0.448 |
| PERSON <-> DATE | 0.442 |
| PERSON <-> ORG | 0.434 |
| PERSON <-> EVENT | 0.411 |
| PERSON <-> DISEASE | 0.333 |
