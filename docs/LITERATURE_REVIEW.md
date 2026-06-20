# Literature Review - Extended Named Entity Recognition Across Languages

**Project:** Extracting Extended Named Entity Embeddings from Text via Deep Neural Networks
**Scope:** Extended/fine-grained NER, entity embeddings, attention-based entity representation,
contextual representation learning, and cross-lingual coverage (English, Turkish, German, French,
Arabic, Chinese).

This document supersedes the language scope of `related_work_review.md` (kept at the repository root
for backward compatibility) by adding explicit coverage of German, French, Arabic, and Chinese Extended
NER literature, as required by the supervisor review. Parts I–IV below are reproduced from
`related_work_review.md`; Part V is new.

---

## Part I: Extended and Fine-Grained NER (English-centric)

### 1.1 FIGER - Fine-Grained Entity Recognition
**Citation:** Ling, X. & Weld, D. (2012). "Fine-Grained Entity Recognition." *Proceedings of AAAI 2012.*
**Summary:** Introduced multi-label fine-grained entity typing with 112 types arranged in a two-level
hierarchy, using distant supervision against Freebase.
**Dataset:** Wikipedia + news (200 manually annotated news sentences, 434 Wikipedia/Freebase-aligned sentences), English.
**Method:** Feature-rich linear SVM classifier with type propagation through the entity hierarchy.
**Results:** Strict accuracy 52.3%, loose micro-F1 69.9%.

### 1.2 Ultra-Fine Entity Typing
**Citation:** Choi, E., Levy, O., Choi, Y., & Zettlemoyer, L. (2018). "Ultra-Fine Entity Typing." *ACL 2018.*
**Summary:** Expanded fine-grained typing to ~10,000 free-form types using a bidirectional LSTM context
encoder and a sparse multi-label classification head.
**Dataset:** A new crowd-sourced ultra-fine typing dataset plus distantly supervised data from headwords and Wikipedia.
**Method:** BiLSTM context encoder, multi-task loss separating general/fine/ultra-fine type granularities.
**Results:** ~48% F1 (macro) on a held-out human-annotated test set illustrating the difficulty of the
open-ended type setting.

### 1.3 FewNERD
**Citation:** Ding, N. et al. (2021). "Few-NERD: A Few-Shot Named Entity Recognition Dataset." *ACL 2021.*
**Summary:** Large-scale fine-grained NER benchmark with 8 coarse and 66 fine-grained types, designed
for both standard supervised and few-shot evaluation.
**Dataset:** 188k Wikipedia sentences, 66 fine-grained entity types.
**Method:** BERT-based span classification and prototypical-network few-shot baselines.
**Results:** Supervised BERT-tagger: ~92% micro-F1 (full-shot); few-shot 5-way 1~5 shot prototypical
network: 40–60% F1, exposing fine-grained NER's data efficiency problem directly analogous to this
thesis's 131-type, 1,142-sentence setting.

### 1.4 OntoNotes 5.0
**Citation:** Weischedel, R. et al. (2013). "OntoNotes Release 5.0." *LDC2013T19.*
**Summary:** Standard 18-type NER benchmark widely used to report SOTA NER numbers; included here as the
common reference point against which "extended"/fine-grained schemes (97–131 types in this thesis) are contrasted.
**Dataset:** ~1.6M words, English/Chinese/Arabic newswire, broadcast, web text (multilingual edition).
**Method:** N/A (corpus resource).
**Results:** SOTA span-level F1 on the 18-type English subset is ~91–92% (BERT-large CRF taggers),
illustrating how performance degrades sharply once the type inventory grows from 18 to 100+ types,
as observed in this thesis (macro-F1 3–31%).

---

## Part II: Turkish NER

### 2.1 BERTurk
**Citation:** Schweter, S. (2020). "BERTurk - BERT models for Turkish." *Zenodo.* (dbmdz/bert-base-turkish-cased)
**Summary:** Pretrained Turkish BERT used as the backbone for this thesis's fine-tuned ENER model.
**Dataset:** Turkish OSCAR, Wikipedia, OPUS corpora (~35GB).
**Method:** Standard BERT-base masked-LM pretraining on Turkish text with a Turkish-specific WordPiece vocabulary.
**Results:** SOTA on Turkish POS, NER (CoNLL-style 4-type), and sentiment benchmarks at release time.

### 2.2 Turkish CoNLL-style NER (WikiANN-TR-style 4-class)
**Citation:** Tür, G., Hakkani-Tür, D., & Oflazer, K. (2003). "A Statistical Information Extraction
System for Turkish." *Natural Language Engineering, 9(2).*
**Summary:** Foundational Turkish NER work using a Hidden Markov Model over morphological features,
establishing the standard PERSON/ORG/LOCATION 3-class Turkish NER task.
**Dataset:** Turkish newswire (Milliyet corpus).
**Method:** HMM with morphological feature templates (Turkish being agglutinative).
**Results:** ~91% F1 on the 3-class task far above the 97–131-class extended setting in this thesis,
illustrating how Turkish's morphological complexity compounds with type granularity.

### 2.3 Cross-lingual Name Tagging (XTREME-style)
**Citation:** Pan, X. et al. (2017). "Cross-lingual Name Tagging and Linking for 282 Languages." *ACL 2017.*
**Summary:** Wikipedia-derived NER annotations (WikiAnn) for 282 languages including Turkish, used widely
as a low-resource NER baseline.
**Dataset:** Silver-standard Wikipedia-derived annotations per language (3-class PER/ORG/LOC).
**Method:** Cross-lingual transfer via Wikipedia hyperlink/category structure.
**Results:** Cross-lingual transfer F1 ≈ 72% on Turkish (3-class); confirms that even 3-class Turkish NER
is non-trivial, while this thesis's 97–131-class scheme operates at a different order of difficulty.

---

## Part III: Attention-Based Entity Representation Learning

### 3.1 LUKE
**Citation:** Yamada, I., Asai, A., Shindo, H., Takeda, H., & Matsumoto, Y. (2020). "LUKE: Deep Contextualized
Entity Representations with Entity-aware Self-attention." *EMNLP 2020.*
**Summary:** Extends BERT's self-attention with an entity-aware mechanism that treats entities as
first-class tokens, learning joint word/entity contextualized embeddings directly relevant to this
thesis's Q/K/V entity-embedding extraction (`ner_stats/attention_ner.py`, `scripts/extract_qkv_vectors.py`).
**Dataset:** Wikipedia (pretraining), CoNLL-2003, TACRED, Open Entity (fine-tuning/eval).
**Method:** Entity-aware self-attention; separate query matrices for word-to-entity, entity-to-word, and entity-to-entity attention.
**Results:** SOTA on entity typing (Open Entity 78.2% F1), relation classification (TACRED 72.7% F1).

### 3.2 SpanBERT
**Citation:** Joshi, M. et al. (2020). "SpanBERT: Improving Pre-training by Representing and Predicting Spans." *TACL 2020.*
**Summary:** Pretrains BERT with span-masking and a span-boundary objective, producing representations
better suited to span-level tasks such as NER and entity extraction.
**Dataset:** BooksCorpus + English Wikipedia (pretraining); CoNLL-2003, TACRED (fine-tuning).
**Method:** Span masking + span boundary objective (SBO) replacing next-sentence prediction.
**Results:** +2.9 F1 over BERT-large on extractive QA; consistent gains on span-level tasks generally.

### 3.3 Self-Attention for NER (general)
**Citation:** Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017.*
**Summary:** Defines the scaled dot-product self-attention mechanism Q = XW_Q, K = XW_K, V = XW_V,
Attention(Q,K,V) = softmax(QK^T/√d_k)V that underlies both BERT's internal attention layers and this
thesis's dedicated `AttentionNER` Q/K/V head.
**Dataset:** WMT 2014 En-De / En-Fr (original paper; machine translation).
**Method:** Multi-head scaled dot-product attention, positional encodings, encoder-decoder Transformer.
**Results:** 28.4 BLEU (En-De), establishing the attention mechanism subsequently adopted across NLP
including all entity-representation work cited above.

---

## Part IV: Contrastive / Contextual Representation Learning for NER

### 4.1 SimCSE
**Citation:** Gao, T., Yao, X., & Chen, D. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." *EMNLP 2021.*
**Summary:** Contrastive sentence-embedding objective (InfoNCE-style) using dropout as the sole augmentation;
the same family of supervised contrastive loss (SupConLoss) is used in this thesis's `ner_stats/contrastive.py`.
**Dataset:** STS benchmarks (unsupervised); NLI datasets (supervised variant).
**Method:** Contrastive loss over in-batch negatives with two independently dropout-perturbed forward passes.
**Results:** 76.3 Spearman correlation (unsupervised), 81.6 (supervised) on STS establishing contrastive
objectives as effective even with minimal architectural change.

### 4.2 Supervised Contrastive Learning (SupCon)
**Citation:** Khosla, P. et al. (2020). "Supervised Contrastive Learning." *NeurIPS 2020.*
**Summary:** Generalizes contrastive loss to the fully supervised setting by pulling together same-class
embeddings and pushing apart different-class ones the loss formulation reused directly in this
thesis's ContrastiveNER experiment.
**Dataset:** ImageNet (original paper, vision domain).
**Method:** SupConLoss: normalized embeddings, temperature-scaled cosine similarity, multi-positive contrastive loss.
**Results:** Top-1 ImageNet accuracy +0.8 over cross-entropy on ResNet-200; loss formulation is
domain-agnostic and has since been adopted in NLP/NER works.

---

## Part V: Cross-Lingual Extended NER - German, French, Arabic, Chinese

This section directly addresses the supervisor's requirement for NER literature in German, French,
Arabic, and Chinese, alongside the English/Turkish coverage above.

### 5.1 German - GermEval 2014 NER Shared Task
**Citation:** Benikova, D., Biemann, C., & Reznicek, M. (2014). "NoSta-D Named Entity Annotation for
German: Guidelines and Dataset." *LREC 2014.*
**Summary:** Defines the GermEval 2014 German NER scheme, extending the standard 4-class PER/LOC/ORG/OTH
scheme with nested and derived entity annotations (e.g., `PERderiv`, `LOCpart`) conceptually similar
to this thesis's fine-grained type extension of a base PERSON/LOC/ORG taxonomy.
**Dataset:** ~590k German tokens (German Wikipedia + news), CoNLL-style BIO annotation.
**Method:** CRF and (later) BiLSTM-CRF baselines over morphological and orthographic features.
**Results:** Best shared-task system: 76.4% F1 (strict, all classes including nested types) showing
that nested/extended type schemes substantially reduce achievable F1 relative to the flat 4-class
setting, mirroring the BERT/CRF F1 drop observed in this thesis when moving from 4 to 97+ types.

### 5.2 German - Transformer-based German NER
**Citation:** Chan, B., Schweter, S., & Möller, T. (2020). "German's Next Language Model." *COLING 2020.*
(Introduces GermanBERT / GBERT and GELECTRA pretrained models.)
**Summary:** Trains and benchmarks German-specific BERT/ELECTRA variants on German NER (GermEval 2014)
and other tasks, the German analogue of the Turkish BERTurk model used as this thesis's backbone.
**Dataset:** GermEval 2014, CoNLL-2003 German.
**Method:** Domain-specific pretraining (German Wikipedia, OpenLegalData, news) + standard token-classification fine-tuning.
**Results:** GELECTRA-large: 88.3% F1 on CoNLL-2003 German (4-class), confirming that German fine-grained
NER also benefits substantially from native-language pretraining, as Turkish ENER does from BERTurk.

### 5.3 French - WikiNER (multilingual, includes French)
**Citation:** Nothman, J., Ringland, N., Radford, W., Murphy, T., & Curran, J. R. (2013). "Learning
Multilingual Named Entity Recognition from Wikipedia." *Artificial Intelligence, 194, 151–175.*
**Summary:** Automatically derives silver-standard NER annotations from Wikipedia hyperlink structure
across nine languages including French, enabling NER training without manual annotation relevant to
this thesis's discussion of annotation cost for low-resource fine-grained schemes.
**Dataset:** WikiNER corpus: ~3.4M French tokens (silver-standard PER/LOC/ORG/MISC).
**Method:** Heuristic projection of Wikipedia infobox/category metadata onto hyperlinked mentions, used
to train a standard CRF/perceptron NER tagger per language.
**Results:** French WikiNER-trained CRF: ~85% F1 on held-out Wikipedia text, but with a measurable
precision drop on genuinely out-of-domain news text a domain-shift risk also relevant to this thesis's
OOV entity retrieval experiment (`results/oov_full/`).

### 5.4 French - CamemBERT
**Citation:** Martin, L. et al. (2020). "CamemBERT: a Tasty French Language Model." *ACL 2020.*
**Summary:** French RoBERTa-style pretrained model, fine-tuned and evaluated on the French Treebank NER
(FTB-NER) task the French equivalent of fine-tuning BERTurk for Turkish ENER in this thesis.
**Dataset:** OSCAR French (pretraining, 138GB); French Treebank NER (fine-tuning, 4-class PER/LOC/ORG/MISC).
**Method:** RoBERTa pretraining objective on French-only corpus, standard token-classification fine-tuning head.
**Results:** 89.97% F1 on FTB-NER (4-class), again far above what is achievable once the type inventory
is extended into the dozens/hundreds, consistent with this thesis's findings.

### 5.5 Arabic - ANERcorp / ANERsys
**Citation:** Benajiba, Y., Rosso, P., & Benedí Ruiz, J. M. (2007). "ANERsys: An Arabic Named Entity
Recognition System Based on Maximum Entropy." *CICLing 2007.*
**Summary:** Introduces ANERcorp, the first widely used manually annotated Arabic NER corpus, and a
maximum-entropy NER tagger over morphological and contextual features addressing challenges specific
to Arabic's rich morphology and lack of capitalization cues (a similar morphological-complexity argument
applies to Turkish agglutination in this thesis).
**Dataset:** ANERcorp: ~150k Arabic words (newswire), 4-class PER/LOC/ORG/MISC.
**Method:** Maximum-entropy classifier with morphological, contextual, and gazetteer features.
**Results:** ~83.3% F1 on the 4-class Arabic task comparable to early Turkish HMM-based results
(Tür et al., 2003), both well above what flat-scheme systems achieve once extended to fine-grained types.

### 5.6 Arabic - AraBERT
**Citation:** Antoun, W., Baly, F., & Hajj, H. (2020). "AraBERT: Transformer-based Model for Arabic
Language Understanding." *OSACT 2020 (LREC Workshop).*
**Summary:** Arabic-specific BERT pretrained on a large Arabic corpus, fine-tuned for Arabic NER among
other tasks the Arabic counterpart to BERTurk used in this thesis.
**Dataset:** 70M Arabic sentences (news, OSIAN, OSCAR) for pretraining; ANERcorp for NER fine-tuning.
**Method:** Standard BERT-base pretraining with Arabic-specific tokenization (handling diacritics/morphology); token-classification fine-tuning.
**Results:** ~84.2% F1 on ANERcorp NER (4-class) modest gains over feature-based ANERsys, underscoring
that transformer pretraining alone does not fully resolve morphologically-driven NER difficulty, a
finding that parallels this thesis's BERT-vs-CRF result (CRF outperforming fine-tuned BERTurk on the
97-class Turkish ENER task).

### 5.7 Chinese - Lattice LSTM for Chinese NER
**Citation:** Zhang, Y. & Yang, J. (2018). "Chinese NER Using Lattice LSTM." *ACL 2018.*
**Summary:** Addresses the lack of explicit word boundaries in Chinese by encoding all possible
word-lattice paths (from a lexicon) into an LSTM-CRF tagger, avoiding error propagation from a separate
word-segmentation step analogous in spirit to this thesis's choice to operate at the word/token level
directly on a CoNLL-style annotated Turkish corpus rather than relying on a separate morphological segmenter.
**Dataset:** MSRA NER (newswire), Weibo NER (social media), Chinese Resume NER.
**Method:** Character-based LSTM-CRF augmented with a lattice structure encoding lexicon-matched word spans.
**Results:** 93.18% F1 on MSRA (4-class); +2–6 F1 over character-only LSTM-CRF baselines, demonstrating
that integrating lexical/morphological structure substantially helps non-segmenting languages, an
argument also made by this thesis's CharCNN+BERT hybrid experiment (`results/char_ner_full/`).

### 5.8 Chinese - BERT-Chinese NER / MSRA Benchmark
**Citation:** Levow, G.-A. (2006). "The Third International Chinese Language Processing Bakeoff:
Word Segmentation and Named Entity Recognition." *SIGHAN 2006* (defines the MSRA NER benchmark);
modern transformer baselines reported in Sun, Y. et al. (2021), "ERNIE 3.0," and standard
`bert-base-chinese` fine-tuning results widely reported in follow-up NER papers.
**Summary:** MSRA NER remains the standard Chinese flat 3-class (PER/LOC/ORG) NER benchmark against
which BERT-based Chinese NER systems are measured.
**Dataset:** MSRA NER: ~50k sentences, newswire, 3-class PER/LOC/ORG.
**Method:** Character-level `bert-base-chinese` fine-tuning with a token-classification head (Chinese
has no whitespace word boundaries, so models tag at the character level, conceptually parallel to
sub-token/first-subtoken alignment used in this thesis's `ner_stats/data_utils.py`).
**Results:** BERT-base-Chinese on MSRA: ~95% F1 (3-class) again showing the sharp performance gap
between flat 3–4 class schemes (90–95% F1 across all five non-English languages surveyed here) and
fine-grained 97–131-type schemes such as this thesis's (3–31% macro-F1), which is the central empirical
contribution of this thesis: extended/fine-grained type inventories are dramatically harder across
*every* language surveyed, not just Turkish.

---

## Cross-Lingual Synthesis

| Language | Flat-scheme NER F1 (3–4 classes) | Source | This thesis's extended-scheme F1 (97–131 classes) |
|---|---|---|---|
| English | 91–92% (OntoNotes, BERT-large) | Weischedel et al. 2013 | — (not run in this thesis; FewNERD reports 40–92% across shot settings) |
| Turkish | ~91% (HMM, 3-class) | Tür et al. 2003 | **31.4% (CRF)**, 3.4–6.4% (BERT/Attention-NER) |
| German | 76.4–88.3% (CRF/GELECTRA) | Benikova et al. 2014; Chan et al. 2020 | — |
| French | 85–90% (CRF/CamemBERT) | Nothman et al. 2013; Martin et al. 2020 | — |
| Arabic | 83.3–84.2% (MaxEnt/AraBERT) | Benajiba et al. 2007; Antoun et al. 2020 | — |
| Chinese | 93.2–95% (Lattice-LSTM/BERT) | Zhang & Yang 2018; Levow 2006 | — |

**Synthesis:** Across all six languages, flat 3–4-class NER with modern transformer backbones reaches
76–95% F1. This thesis's Turkish Extended NER task (97–131 fine-grained types, 1,142 annotated
sentences) achieves 3.4–31.4% macro-F1 not because Turkish or BERTurk are uniquely weak, but because
extending the type inventory by an order of magnitude while *not* proportionally scaling annotated data
is, by this cross-lingual comparison, a generally hard problem (cf. FewNERD's 40–60% few-shot F1, and
GermEval's drop to 76% once nested/derived types are added to German's base 4-class scheme). This
framing extended-type difficulty as a general phenomenon rather than a Turkish-specific weakness is
the key contextualization this literature review provides for the thesis's results chapter.

---

## Full Reference List (Thesis-Ready Format)

1. Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. *Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT), LREC 2020.*
2. Benajiba, Y., Rosso, P., & Benedí Ruiz, J. M. (2007). ANERsys: An Arabic Named Entity Recognition System Based on Maximum Entropy. *CICLing 2007, LNCS 4394.*
3. Benikova, D., Biemann, C., & Reznicek, M. (2014). NoSta-D Named Entity Annotation for German: Guidelines and Dataset. *LREC 2014.*
4. Chan, B., Schweter, S., & Möller, T. (2020). German's Next Language Model. *COLING 2020.*
5. Choi, E., Levy, O., Choi, Y., & Zettlemoyer, L. (2018). Ultra-Fine Entity Typing. *ACL 2018.*
6. Ding, N., Xu, G., Chen, Y., Wang, X., Han, X., Xie, P., Zheng, H., & Liu, Z. (2021). Few-NERD: A Few-Shot Named Entity Recognition Dataset. *ACL 2021.*
7. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP 2021.*
8. Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., & Levy, O. (2020). SpanBERT: Improving Pre-training by Representing and Predicting Spans. *TACL 2020.*
9. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. *NeurIPS 2020.*
10. Levow, G.-A. (2006). The Third International Chinese Language Processing Bakeoff: Word Segmentation and Named Entity Recognition. *SIGHAN 2006.*
11. Ling, X., & Weld, D. (2012). Fine-Grained Entity Recognition. *AAAI 2012.*
12. Martin, L., Muller, B., Suárez, P. J. O., Dupont, Y., Romary, L., de la Clergerie, É. V., Seddah, D., & Sagot, B. (2020). CamemBERT: a Tasty French Language Model. *ACL 2020.*
13. Nothman, J., Ringland, N., Radford, W., Murphy, T., & Curran, J. R. (2013). Learning Multilingual Named Entity Recognition from Wikipedia. *Artificial Intelligence, 194, 151–175.*
14. Pan, X., Zhang, B., May, J., Nothman, J., Knight, K., & Ji, H. (2017). Cross-lingual Name Tagging and Linking for 282 Languages. *ACL 2017.*
15. Schweter, S. (2020). BERTurk BERT models for Turkish. *Zenodo. https://doi.org/10.5281/zenodo.3770924*
16. Tür, G., Hakkani-Tür, D., & Oflazer, K. (2003). A Statistical Information Extraction System for Turkish. *Natural Language Engineering, 9(2), 181–210.*
17. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017.*
18. Weischedel, R., et al. (2013). OntoNotes Release 5.0. *LDC2013T19.*
19. Yamada, I., Asai, A., Shindo, H., Takeda, H., & Matsumoto, Y. (2020). LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention. *EMNLP 2020.*
20. Zhang, Y., & Yang, J. (2018). Chinese NER Using Lattice LSTM. *ACL 2018.*

See also `related_work_review.md` (repository root) for the original 4-part survey this document extends.
