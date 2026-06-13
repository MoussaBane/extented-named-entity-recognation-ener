# Related Work Review
## Turkish Extended Named Entity Recognition: Literature Survey

**Author:** Moussa Bane  
**Date:** June 2026  
**Scope:** Extended NER · Turkish NLP · Attention-based Entity Learning · Contrastive Learning

---

## Overview

This review covers four research areas directly relevant to the Turkish Extended NER (ENER) thesis:

1. Extended and Fine-Grained NER systems
2. Turkish NER methods and benchmarks
3. Attention-based entity representation learning
4. Contrastive learning for NER

For each paper, we provide: citation details, methodology, dataset, results, strengths, weaknesses, and comparison to this thesis.

---

## Part I: Extended and Fine-Grained NER

### 1.1 FIGER — Fine-Grained Entity Recognition

**Citation:** Ling & Weld (2012). "Fine-Grained Entity Recognition." *Proceedings of AAAI 2012.*

**Methodology:**
- Multi-label entity typing with 112 entity types organized in a 2-level hierarchy
- Used distant supervision with Freebase knowledge base
- Feature-rich linear classifier (SVM) over hand-engineered features
- Type propagation through entity hierarchy

**Dataset:**
- Wikipedia + news articles (English)
- 200 news sentences manually annotated
- 434 Wikipedia sentences with Freebase alignment

**Results:**
- Strict accuracy: 52.3%
- Loose micro F1: 69.9%

**Strengths:**
- First system with 100+ entity types
- Hierarchical type propagation reduces annotation burden
- Distant supervision scales to large corpora

**Weaknesses:**
- Noisy labels from distant supervision
- English only
- Requires external knowledge base (Freebase)
- Type hierarchy is English-centric

**Comparison to This Thesis:**
- Our work targets 131 entity types (19 more than FIGER) without a hierarchical constraint
- We use direct human annotation (INCEpTION) rather than distant supervision, producing cleaner labels
- We address Turkish (morphologically complex) vs. English (analytic), making our task harder
- We do not require a knowledge base; entity types are corpus-defined

---

### 1.2 FewNERD — Few-Shot Named Entity Recognition

**Citation:** Ding et al. (2021). "Few-NERD: A Few-Shot Named Entity Recognition Dataset." *Proceedings of ACL 2021.*

**Methodology:**
- 8 coarse entity types, 66 fine-grained subtypes
- Episode-based few-shot learning: N-way K-shot
- Prototype-based classification with BERT encoder
- MNRE (Multiple-class NER Re-labeling Episodes) evaluation protocol

**Dataset:**
- 188,200 sentences (English Wikipedia)
- 491,711 entities
- 8 coarse, 66 fine-grained types
- Manually annotated by crowd workers

**Results:**
- BERT-PROTO: micro F1 = 62.1% (5-way 5-shot)
- SpanProto: micro F1 = 68.4% (5-way 5-shot)

**Strengths:**
- First large-scale few-shot NER benchmark
- 66 fine-grained types cover many real-world scenarios
- Episode-based evaluation directly tests generalization
- Rich annotation guidelines

**Weaknesses:**
- English only (does not address morphological complexity)
- Episode-based evaluation is not directly comparable to standard NER evaluation
- 5-shot assumption may be unrealistic for specialized domains

**Comparison to This Thesis:**
- FewNERD uses 66 types; our ENER uses 131 types (2× more fine-grained)
- FewNERD has 188K sentences; our corpus has 1,142 sentences (100× smaller)
- This makes our task significantly harder: fewer examples per class under a larger tagset
- Both works use BERT-based prototype classifiers; our CVA approach is complementary
- Our attention-based approach goes beyond prototypes by extracting Q/K/V representations

---

### 1.3 Ultra-Fine Entity Typing

**Citation:** Choi et al. (2018). "Ultra-Fine Entity Typing." *Proceedings of ACL 2018.*

**Methodology:**
- 10,331 entity types (ultra-fine-grained)
- Multi-label entity typing (a mention can have multiple types)
- BERT encoder + sigmoid output for multi-label classification
- Label embedding with type hierarchy

**Dataset:**
- 6 billion tokens of web text (English)
- 25M automatically labeled mentions
- 2,000 manually annotated sentences for evaluation

**Results:**
- Macro F1: 46.8% (on 10K types)
- Manually annotated test: 48.1%

**Strengths:**
- Most fine-grained NER system published (10K types)
- Large-scale automatic labeling
- Multi-label capability

**Weaknesses:**
- 10K types requires massive data; few-shot performance is poor
- Label noise from automatic annotation
- Evaluated only in English

**Comparison to This Thesis:**
- Ultra-Fine uses 10K types but requires billions of training tokens
- Our 131-type corpus uses 1,142 manually annotated sentences
- This places our work in a different regime: supervised fine-grained NER on small, high-quality data
- Our research question is different: can attention-based representations improve classification in the low-resource ENER setting?

---

## Part II: Turkish NLP and Turkish NER

### 2.1 Turkish BERT NER (BERTurk)

**Citation:** Schweter & Çano (2020). "BERTurk — BERT models for Turkish." *arXiv:2003.10195.*

**Methodology:**
- Pre-trained BERT on Turkish Wikipedia and OSCAR corpus
- Fine-tuned on Turkish NER datasets
- Two models: `dbmdz/bert-base-turkish-cased` and `dbmdz/bert-base-turkish-uncased`
- Standard sequence labeling with CRF on top (optional)

**Dataset (NER):**
- Turkish NER benchmark (Tür et al., 2003)
- WikiNER Turkish subset
- 4 entity types: PERSON, LOCATION, ORGANIZATION, MISC

**Results:**
- F1 (Turkish NER): 93.6% (BERTurk + CRF)
- F1 (WikiNER-TR): 89.2%

**Strengths:**
- Turkish-specific pre-training corpus (better tokenization for agglutinative morphology)
- Open-source release of model weights (used in this thesis)
- Strong baseline for Turkish NLP tasks

**Weaknesses:**
- Trained for 4-class NER; not designed for fine-grained (97+ class) NER
- Does not address class imbalance or few-shot learning
- No morphological analysis integration

**Comparison to This Thesis:**
- We use `dbmdz/bert-base-turkish-cased` as our backbone (the same model evaluated here)
- We fine-tune it for 97 entity types (vs. 4 types) — a 24× increase in label space
- The drop from 93.6% F1 (4-class) to 3.4% macro F1 (97-class) illustrates the ENER difficulty
- Our work provides the first Turkish ENER evaluation using the BERTurk model family

---

### 2.2 BOUN Turkish NER

**Citation:** Kucuk & Can (2020). "Named Entity Recognition for Turkish: A Survey." *Digital Scholarship in the Humanities, 35(4).*

**Methodology:**
- Survey of 20+ Turkish NER systems
- Rule-based, statistical (CRF, HMM), and neural approaches
- Evaluation on standard Turkish NER benchmarks

**Key Systems Reviewed:**
- TurkishNER (CRF + morphological features)
- BilSTM-CRF for Turkish
- BERT-based Turkish NER

**Dataset:**
- Tür et al. (2003) Turkish NER corpus
- NKJP Polish NER (cross-lingual comparison)

**Results (CRF baseline):**
- Turkish NER (4 types): 70–85% F1 (depending on features)

**Strengths:**
- Comprehensive survey of Turkish NER literature
- Identifies key challenges: agglutination, data scarcity, entity boundary ambiguity
- Provides baseline numbers for comparison

**Weaknesses:**
- Only covers standard 4-type NER
- Does not address fine-grained or domain-specific NER
- Pre-BERT-era systems dominate the survey

**Comparison to This Thesis:**
- Our CRF baseline (macro F1 31.4% on 97 types) is not directly comparable to 4-type CRF results
- We adopt the same CRF feature engineering approach (lexical features, context window)
- Our thesis extends the Turkish NER landscape from 4 to 97 entity types
- The agglutination challenges identified in the survey directly motivate our character-level CNN approach

---

### 2.3 WikiANN Turkish

**Citation:** Pan et al. (2017). "Cross-lingual Name Tagging and Linking for 282 Languages." *Proceedings of ACL 2017.*

**Methodology:**
- Automatically constructed NER dataset from Wikipedia for 282 languages
- Distant supervision using Wikipedia anchor texts and Freebase
- PER, LOC, ORG types only (IOB format)

**Dataset (Turkish):**
- ~20,000 sentences (Turkish Wikipedia)
- 3 entity types: PER, LOC, ORG
- Automatic annotation (noisy)

**Results on Turkish:**
- Cross-lingual transfer (best): ~72% F1
- Monolingual Turkish: ~80% F1 (with BERT fine-tuning)

**Strengths:**
- Only cross-lingual Turkish NER dataset at scale
- Enables zero-shot cross-lingual transfer experiments
- Available in HuggingFace Datasets

**Weaknesses:**
- Only 3 entity types (very coarse)
- High label noise from automatic construction
- No fine-grained entity type coverage

**Comparison to This Thesis:**
- WikiANN-TR uses 3 coarse types; we use 131 fine-grained types (43× more granular)
- WikiANN-TR is automatically labeled; our corpus is manually annotated with INCEpTION
- Our work fills the gap in Turkish NER: there is no existing fine-grained Turkish NER dataset
- Our ENER corpus is the first Turkish fine-grained NER resource with 130+ types

---

## Part III: Attention-Based Entity Representation Learning

### 3.1 Attention Is All You Need (Transformer)

**Citation:** Vaswani et al. (2017). "Attention Is All You Need." *Proceedings of NeurIPS 2017.*

**Methodology:**
- Self-attention mechanism with multi-head scaled dot-product attention
- Q, K, V linear projections from input embeddings
- Positional encodings for sequence order
- Encoder-decoder architecture for sequence transduction

**Key Formulas:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
MultiHead(H) = concat(head_1, ..., head_h) · W_O
```

**Results:**
- BLEU score 28.4 on WMT English-German translation
- State-of-the-art at time of publication

**Relevance to This Thesis:**
- Provides the mathematical foundation for Q/K/V extraction
- The Q, K, V roles (query/key/value) directly motivate our hypothesis that K-projections may encode entity "identity" signals
- The multi-head mechanism explains why layer 11 Q/K/V may differ from earlier layers

---

### 3.2 LUKE — Language Understanding with Knowledge-Based Embeddings

**Citation:** Yamada et al. (2020). "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention." *Proceedings of EMNLP 2020.*

**Methodology:**
- BERT-like model with additional entity embedding layer
- Entity-aware self-attention: words attend to entities and vice versa
- Pre-trained on Wikipedia with entity prediction objective
- Token and entity classification heads

**Dataset:**
- English Wikipedia (3.5M entities)
- CoNLL-2003 NER, SQuAD 1.1, Open Entity

**Results:**
- CoNLL-2003 NER: F1 = 94.3% (new SOTA in 2020)
- Open Entity typing: F1 = 78.2%

**Strengths:**
- First model to integrate entity-aware attention at pre-training
- Achieves SOTA on multiple entity understanding tasks
- Q/K/V modifications directly relevant to our approach

**Weaknesses:**
- Requires Wikipedia-scale entity database for pre-training
- English only
- 3.5M entity embeddings make the model extremely large

**Comparison to This Thesis:**
- LUKE uses external entity embeddings as Q/K/V inputs; we extract Q/K/V from a standard BERT without external entities
- LUKE's entity-aware attention is pre-trained; our task-specific attention head is fine-tuned
- Both approaches use Q/K/V projections to improve entity representation, but at different stages
- Our approach is simpler (no external KB required) but limited to in-distribution entities

---

### 3.3 SpanBERT — Better Span-Level Representations

**Citation:** Joshi et al. (2020). "SpanBERT: Improving Pre-training by Representing and Predicting Spans." *TACL 2020.*

**Methodology:**
- BERT extension with span-level pre-training objective
- Span boundary representation: concatenate [start, end, width] tokens
- SpanMLM: predict masked span content from boundary representations
- SBO: Span Boundary Objective for span prediction

**Dataset:**
- Wikipedia + BookCorpus (English)
- CoNLL-2003 NER, SQuAD, coreference resolution

**Results:**
- CoNLL-2003 NER: F1 = 93.4%
- SQuAD 2.0: F1 = 88.7%

**Strengths:**
- Span-level representations capture multi-token entity boundaries better
- Simple modification to BERT pre-training
- Significant improvement on span prediction tasks

**Weaknesses:**
- Span representations are fixed at pre-training time
- Entity type discrimination still relies on fine-tuning
- Not released for non-English languages

**Comparison to This Thesis:**
- SpanBERT uses span boundary representations; we use token-level Q/K/V projections
- For Turkish agglutinative NER, span boundaries are ambiguous (where does the entity end when a suffix is attached?)
- Our work focuses on entity type discrimination (97 classes), not span detection
- SpanBERT's approach is complementary: combining span representations with Q/K/V projections is future work

---

### 3.4 EntityBERT — Entity-centric BERT Pre-training

**Citation:** Liu et al. (2021). "EntityBERT: Entity-centric Masking Strategy for Model Pretraining for the Open-domain Dialogue System." *Proceedings of ACL 2021.*

**Methodology:**
- BERT pre-training with entity-centric masking (mask full entity spans, not random tokens)
- Entity replacement language modeling objective
- Designed for dialogue NER but applicable generally

**Results:**
- NER F1 improvement of 1.2% on CoNLL-2003 vs. standard BERT
- Dialogue entity recognition: significant improvement

**Strengths:**
- Simple modification to pre-training
- Entity-centric masking forces better entity representations
- Applicable to any NER domain

**Weaknesses:**
- Requires full pre-training (computationally expensive)
- Entity spans must be pre-identified for masking
- Only evaluated on English

**Comparison to This Thesis:**
- EntityBERT requires re-pre-training; we use post-hoc analysis of a pre-trained model
- Entity-centric masking is a valid future direction for Turkish ENER
- Our work demonstrates that even without entity-centric pre-training, Q/K/V extraction reveals entity structure in BERT representations

---

## Part IV: Contrastive Learning for NLP and NER

### 4.1 SimCSE — Simple Contrastive Learning for Sentence Embeddings

**Citation:** Gao et al. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." *Proceedings of EMNLP 2021.*

**Methodology:**
- Unsupervised: same sentence → two augmentations (different dropout masks) as positive pair
- Supervised: NLI premise-entailment as positive pair; contradiction as hard negative
- InfoNCE contrastive loss on CLS representations

**Dataset:**
- STS benchmark (sentence similarity)
- SNLI + MultiNLI (supervised version)

**Results:**
- STS-B Spearman: 76.3% (unsupervised), 83.5% (supervised)
- Significant improvement over standard BERT

**Strengths:**
- Simple dropout-based augmentation requires no external data
- Addresses representation degeneration in BERT (anisotropic distribution)
- Works for sentence-level representations

**Weaknesses:**
- Designed for sentence-level, not token-level tasks
- Dropout augmentation may not create meaningful entity-level pairs

**Comparison to This Thesis:**
- SimCSE operates on sentence embeddings; our SupConLoss operates on token-level entity embeddings
- Both address the same problem: BERT representations are not well-distributed for similarity tasks
- Our supervised contrastive approach (SupConLoss) is closer to SimCSE's supervised variant
- We use BIO labels to define positive/negative pairs (natural supervision), avoiding the need for NLI data

---

### 4.2 Supervised Contrastive Learning (SupConLoss)

**Citation:** Khosla et al. (2020). "Supervised Contrastive Learning." *Proceedings of NeurIPS 2020.*

**Methodology:**
- Extension of SimCLR to supervised setting with label information
- Multiple positive pairs per anchor (all samples of same class)
- Temperature-scaled InfoNCE loss
- Two-stage training: contrastive pre-training → linear probing

**Results (ImageNet):**
- Top-1 accuracy: 78.7% (vs. CrossEntropy: 78.3%)
- Improved calibration and out-of-distribution robustness

**Relevance to This Thesis:**
- We directly use the SupConLoss formulation from this paper
- For entity NER, all tokens with the same BIO label form positive pairs
- The temperature (τ=0.07) is adopted from this paper
- The projection head design follows the 2-layer MLP from this paper

**Key adaptation for NER:**
- Original: image crops as augmentation; Our work: same token in different batch contexts as "augmentation"
- Original: batch of images; Our work: batch of entity tokens from multiple sentences

---

### 4.3 Contrastive Learning for NER

**Citation:** Das et al. (2022). "CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning." *Proceedings of ACL 2022.*

**Methodology:**
- Few-shot NER with contrastive learning
- Gaussian embedding for uncertainty modeling
- Episode-based training with support/query sets
- Gaussian distribution distance for classification

**Dataset:**
- I2B2 (clinical NER), Few-NERD, CoNLL-2003

**Results:**
- Few-NERD 5-way 1-shot: F1 = 55.2%
- I2B2 5-way 5-shot: F1 = 72.1%

**Strengths:**
- Addresses few-shot NER directly (1-5 examples per class)
- Gaussian embedding captures uncertainty
- Strong empirical results on clinical NER

**Weaknesses:**
- Episode-based evaluation not standard for full-corpus NER
- Gaussian complexity adds training instability
- Only tested on English

**Comparison to This Thesis:**
- CONTaiNER directly motivates our contrastive approach for low-resource ENER
- Our dataset has 1–10 examples per entity type per fold (matching the few-shot setting)
- We use SupConLoss (simpler) vs. Gaussian contrastive (complex) — a conscious design choice for stability
- Both works demonstrate contrastive learning benefits for sparse entity types
- Our work extends this to Turkish and to 97 entity types vs. CONTaiNER's 6–8 types per episode

---

## Summary Table: Comparison to This Thesis

| Paper | Types | Lang | Training Size | Key Method | F1 | This Thesis Difference |
|-------|-------|------|--------------|-----------|-----|----------------------|
| FIGER | 112 | EN | Distant sup. | Linear CRF | 70% | Turkish, manual annot., Q/K/V attention |
| FewNERD | 66 | EN | 188K sents | BERT Prototype | 68% | 2× more types, 100× less data |
| Ultra-Fine | 10K | EN | 6B tokens | Multi-label BERT | 47% | 100× less data, manual labels |
| BERTurk NER | 4 | TR | 50K sents | BERT+CRF | 94% | Same backbone, 24× more types |
| WikiANN-TR | 3 | TR | 20K sents | BERT | 80% | 43× more types, manual annotation |
| LUKE | 3 | EN | 3.5M entities | Entity-aware attn. | 94% | No external KB required |
| SpanBERT | 4 | EN | Wikipedia | Span repr. | 93% | Token-level Q/K/V, Turkish |
| SimCSE | N/A | EN | STS | Contrastive sent. | 84% (STS) | Token-level, NER task |
| CONTaiNER | 8 | EN | Few-shot | Gaussian contra. | 72% | Turkish, 97 types |
| **This Thesis** | **131** | **TR** | **1,142 sents** | **BERT+CRF+Q/K/V+CVA+SupCon** | **CRF: 31.4%** | — |

---

## Conclusion

The Turkish ENER project occupies a unique position in the literature:

1. **Most fine-grained Turkish NER dataset** — 131 types vs. 3–4 in existing Turkish NER work
2. **First Q/K/V attention analysis for fine-grained NER** — extends LUKE's entity-aware attention to post-hoc analysis
3. **Low-resource ENER benchmark** — 1,142 sentences for 131 types is a new challenge setting
4. **Multi-method comparison under shared protocol** — BERT, CRF, CVA, Attention-NER, Contrastive NER evaluated on the same data with the same folds

This combination of factors makes the work a novel contribution to both Turkish NLP and to the broader field of fine-grained named entity recognition.
