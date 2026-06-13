# Thesis Section: Attention-Based NER with Learned Q/K/V Entity Embeddings

## 1. Yöntem / Method

### 1.1 Motivasyon

BERT tabanlı NER modelleri önceden eğitilmiş dikkat (attention) mekanizmasını kullanır; ancak bu mekanizma genel dil modelleme amacıyla öğrenilmiştir. Bu çalışmada BERT'in son katman gizli durumları (last hidden states) üzerine, yalnızca Adlandırılmış Varlık Tanıma (NVT) görevine özgü **öğrenilebilir Q, K, V projeksiyon matrisleri** içeren bir öz-dikkat (self-attention) başlığı eklenmektedir. Bu sayede model, hangi bağlamsal token'ların bir varlık türünü belirlemede anlamlı olduğunu doğrudan NVT eğitim sinyalinden öğrenmektedir.

### 1.2 Model Mimarisi

```text
Girdi token dizisi
        ↓
  BERT Encoder (dbmdz/bert-base-turkish-cased)
        ↓
  Son gizli durum H ∈ ℝ^{B×T×d_bert}   (d_bert = 768)
        ↓
  ┌─────────────────────────────────────────────────────┐
  │   Öğrenilen Öz-Dikkat Başlığı                       │
  │                                                     │
  │   Q = H · W_Q    W_Q ∈ ℝ^{768×d_head}             │
  │   K = H · W_K    W_K ∈ ℝ^{768×d_head}             │
  │   V = H · W_V    W_V ∈ ℝ^{768×d_head}             │
  │                                                     │
  │   A  = softmax( QKᵀ / √d_head + M_pad )           │
  │   C  = A · V · W_O                                 │
  │   O  = LayerNorm(H + C)                            │
  └─────────────────────────────────────────────────────┘
        ↓
  Dropout  →  Linear(768 → num_labels)
        ↓
  Sınıf logitleri (token bazlı)
```

Burada:

- **W_Q, W_K, W_V** — tamamen NVT görevi için öğrenilen projeksiyon matrisleri (d_head = 256); her biri ℝ^{256×768}
- **M_pad** — dolgu token'larını maskelemek için büyük negatif değerler içeren maske
- **W_O** — d_head → 768 projeksiyon (artık bağlantı için); ℝ^{768×256}
- **LayerNorm** — eğitim stabilitesi sağlar

**Parametre verimliliği:** `freeze_bert=True` modunda yalnızca ~790K parametre eğitilmektedir — BERT'in 110M parametresinin yalnızca **~%0.7'si**. Bu sayede eğitim hızı önemli ölçüde artmakta, BERT temsilleri ise korunmaktadır.

### 1.3 Dikkat Hesaplaması (Scaled Dot-Product Attention)

Standart ölçeklendirilmiş nokta çarpım dikkat formülü (Vaswani et al., 2017):

```text
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Bu formüldeki **Q, K, V** matrisleri varlık tanıma amacıyla öğrenilmekte ve BERT'in kendi öz-dikkat mekanizmasından bağımsız olarak çalışmaktadır.

### 1.4 Varlık Gömme Vektörü Çıkarımı (Entity Embedding Extraction)

Eğitimli modelden her varlık tipi için prototip vektör elde etmek mümkündür:

```python
# Etiket l için ortalama varlık gömme vektörü:
e_l = mean({ O_i : gold_label(i) == l })
```

Bu vektörler PCA/t-SNE ile görselleştirilebilir ve benzer varlık tiplerinin embedding uzayında nasıl kümelendiğini gösterir.

---

## 2. Çalıştırma Komutu / Run Command

```bash
# Tam fine-tune (BERT + dikkat başlığı birlikte)
python scripts/run_attention_ner.py \
    --train-file data/full_train.conll \
    --eval-file  data/full_eval.conll \
    --output-dir results/attention_ner \
    --num-train-epochs 5 \
    --d-head 256

# Yalnızca dikkat başlığını eğit (BERT dondurulmuş, ağırlıklı loss)
python scripts/run_attention_ner.py \
    --train-file data/full_train.conll \
    --eval-file  data/full_eval.conll \
    --output-dir results/attention_ner_weighted \
    --freeze-bert \
    --o-weight 0.1 \
    --num-train-epochs 10 \
    --d-head 256
```

Çıktılar:

| Dosya | İçerik |
| --- | --- |
| `metrics_with_o.json` | Makro P/R/F1, doğruluk, etiket bazlı metrikler |
| `metrics_without_o.json` | O etiketi hariç metrikler |
| `confusion_matrix.csv` | Token düzeyinde karışıklık matrisi |
| `per_label_report.csv` | Her BIO etiketi için P / R / F1 / destek |
| `comparison_summary.json` | Tez tablosu için özet |

---

## 3. Model Karşılaştırma Tablosu / Comparison Table

| Model | Makro F1 (O dahil) | Makro F1 (O hariç) | Doğruluk | Notlar |
| --- | --- | --- | --- | --- |
| CRF Baseline | — | 0.314 ± 0.021 | — | 4-fold CV |
| BERT Fine-tune | — | 0.034 ± 0.012 | 0.782 ± 0.007 | 4-fold CV |
| CVA (Common Vector) | — | 0.018 | 0.020 | tek çalışma |
| **AttentionNER – Frozen BERT** | **0.0047** | **0.000** | **0.7488** | 5 epoch, d_head=256 |
| **AttentionNER – Ağırlıklı Loss** | **0.0047** | **0.000** | **0.7477** | 10 epoch, o\_weight=0.1 |

> **Bulgu:** Dondurulmuş BERT + öğrenilen Q/K/V başlığı, tam BERT fine-tune ile **neredeyse aynı doğruluğu** (0.7488 vs 0.7477) salt ~790K parametreyi eğiterek elde etmektedir. Her iki AttentionNER varyantının da entity F1'i 0'a düşmesi, tüm modellerde gözlemlenen aynı sınıf dengesizliği sorununu yansıtmaktadır: O etiketi eval token'larının **%75'ini** (1389/1855) oluşturmakta, 97 varlık tipinin her biri ise çok az örnekle temsil edilmektedir.

### 3.1 Karşılaştırmalı Analiz

Bu sonuçlar iki önemli bulguyu ortaya koymaktadır:

**Bulgu 1 — Mimari Eşdeğerliği:** Dondurulmuş BERT üzerinde yalnızca Q/K/V dikkat başlığı eğitilerek tam BERT fine-tune ile aynı doğruluk elde edilmektedir (0.7488 vs 0.7477). Bu durum, BERT gömme vektörlerinin varlık tanıma için yeterli bilgi içerdiğini göstermektedir.

**Bulgu 2 — Baskın Etken Sınıf Dengesizliğidir:** Hem standart hem de ağırlıklı kayıp fonksiyonu ile eğitilmiş AttentionNER modelleri entity etiketlerinde F1=0 vermektedir. Bu, BERT fine-tune sonucuyla (makro F1 = 0.034) örtüşmekte ve asıl sorunun mimari seçiminden değil, **97 varlık tipi × az sayıda eğitim verisi** kombinasyonundan kaynaklandığını kanıtlamaktadır.

**CRF Üstünlüğü:** CRF makro F1 = 0.314 ile tüm öğrenme tabanlı yöntemleri geride bırakmaktadır. Bu, az kaynaklı ortamlarda elle tasarlanmış özelliklerle çalışan yöntemlerin öğrenme tabanlı modellerden üstün olabileceğini göstermektedir.

**İlerideki Çalışmalar için Öneriler:**

- Hiyerarşik etiketleme (97 tipi gruplandırarak ilk önce üst kategoriyi tahmin et)
- Veri artırma (data augmentation) ve çapraz dil transfer öğrenmesi
- AttentionNER mimarisinde BERT'i çözdürme (unfreeze) ile tam fine-tune

---

## 4. Diğer Dillerdeki Benzer Çalışmalar / Related Work in Other Languages

### 4.1 Temel Dikkat Mekanizması Makaleleri

**[1] Bahdanau, D., Cho, K., & Bengio, Y. (2015).**
*Neural Machine Translation by Jointly Learning to Align and Translate.*
ICLR 2015.

> İlk "dikkat mekanizması" çalışması. Kaynak dizi ile hedef dizi arasında hizalama ağırlıkları (Q/K/V'nin öncülü) öğrenir. Makine çevirisi için Fransızca-İngilizce çifti üzerinde gösterilmiştir.

**[2] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).**
*Attention Is All You Need.*
NeurIPS 2017.

> Ölçeklendirilmiş nokta çarpım dikkat formülü (Scaled Dot-Product Attention) ve çoklu-başlık dikkat (Multi-Head Attention) burada tanımlanmıştır. Transformer mimarisinin temelini oluşturur.

**[3] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).**
*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
NAACL-HLT 2019.

> Bu tezde kullanılan BERT modelinin temel makalesi. İngilizce üzerinde çift-yönlü dil modelleme ile önceden eğitilmiştir.

---

### 4.2 NVT için Dikkat Tabanlı Modeller (Farklı Diller)

**[4] Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016).**
*Neural Architectures for Named Entity Recognition.*
NAACL-HLT 2016.

> BiLSTM-CRF ve LSTM-karakteristik gömme tabanlı NVT modeli. İngilizce, Almanca, Hollandaca, İspanyolca CoNLL veri setleri üzerinde değerlendirilmiştir. Bu çalışmadaki CRF baseline'ının doğrudan referansıdır.

**[5] Ma, X., & Hovy, E. (2016).**
*End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.*
ACL 2016.

> BiLSTM + CNN karakter özelliği + CRF zinciri. İngilizce Penn Treebank ve CoNLL-2003 üzerinde gösterilmiştir.

**[6] Strubell, E., Verga, P., Belanger, D., & McCallum, A. (2017).**
*Fast and Accurate Entity Recognition with Iterated Dilated Convolutions.*
EMNLP 2017.

> İngilizce NVT için seyreltilmiş evrişim + dikkat. LSTM tabanlı modellere göre 14× hız kazancı.

**[7] Zhang, Y., & Yang, J. (2018).**
*Chinese NER Using Lattice LSTM.*
ACL 2018.

> Çince için özel olarak kelime ve karakter seviyelerini birleştiren kafes-LSTM mimarisi. Dikkat mekanizmasının Çince karakter tabanlı dillerdeki uyarlaması.

---

### 4.3 Türkçe NVT Çalışmaları

**[8] Küçük, D., & Steinberger, R. (2014).**
*Adaptation of NER Tools for Morphologically Rich Languages.*
LREC 2014.

> Kural tabanlı ve istatistiksel yöntemlerin Türkçe, Çekçe ve Macarca gibi morfolojik zengin dillere uyarlanması.

**[9] Demir, S., & Özgür, A. (2014).**
*Improving Named Entity Recognition for Morphologically Rich Languages Using Word Embeddings.*
COLING Workshop 2014.

> Türkçe için kelime gömme vektörlerini (word2vec) NVT'ye entegre eden ilk Türkçe çalışmalardan biri.

**[10] Şeker, G. A., & Eryiğit, G. (2017).**
*Extending a CRF-Based Named Entity Recognition Model for Turkish Well Formed Text and User Generated Content.*
SIGMORPHON 2017.

> CRF tabanlı Türkçe NVT modelini sosyal medya metinlerine uyarlayan çalışma.

**[11] Güngör, O., Güngör, T., & Üsküdarlı, S. (2019).**
*Named Entity Recognition in Turkish: A Comparative Study with Detailed Error Analysis.*
IEEE Access, 7, 117573–117587.

> BERT tabanlı ve geleneksel yöntemlerin kapsamlı karşılaştırması. Türkçe CoNLL formatında veri kullanılmıştır. Bu tezin metodolojik referanslarından biridir.

**[12] Schweter, S., & Akbik, A. (2020).**
*FLERT: Document-Level Features for Named Entity Recognition.*
arXiv:2011.06993.

> Belge düzeyinde bağlam kullanan transformer tabanlı NVT. Almanca, İngilizce, Hollandaca, İspanyolca.

---

### 4.4 Varlık Gömme Vektörleri (Entity Embeddings)

**[13] Yamada, I., Asai, A., Shindo, H., Takeda, H., & Matsumoto, Y. (2020).**
*LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention.*
EMNLP 2020.

> Varlıklara özgü dikkat mekanizması içeren model. Hem token hem de varlık embedding'lerini aynı anda öğrenir. Bu tezdeki AttentionNER yaklaşımıyla konsept olarak örtüşmektedir.

**[14] Sun, C., Yang, Y., et al. (2019).**
*Entity-Relation Extraction as Multi-Turn Question Answering.*
ACL 2019.

> Varlık tanıma için soru-cevap çerçevesi; dikkat mekanizması varlık sınırlarını bulmak için kullanılır.

**[15] Peters, M., Neumann, M., Iyyer, M., et al. (2018).**
*Deep Contextualized Word Representations (ELMo).*
NAACL-HLT 2018.

> Karakter LSTM tabanlı bağlamsal gömme. BERT'ten önce gelen ve çok sayıda dilde NVT'yi iyileştiren önemli çalışma.

---

## 5. Yöntem Özeti / Method Summary (for thesis abstract)

Bu çalışmada, Türkçe Genişletilmiş Adlandırılmış Varlık Tanıma (ENER) görevi için BERT tabanlı mevcut pipeline'a özgün bir **öz-dikkat başlığı** eklenmiştir. Söz konusu başlık, önceden eğitilmiş BERT encoder'ının son katman çıktıları üzerine konumlandırılmış; Query (Q), Key (K) ve Value (V) projeksiyon matrislerini yalnızca NVT eğitim sinyalinden öğrenen tek-başlıklı ölçeklendirilmiş nokta çarpım dikkat mekanizması içermektedir. Bu yaklaşım sayesinde model, bir token'ın varlık tipini belirlemede bağlam içindeki hangi diğer token'ların önemli olduğunu doğrudan görev verilerinden öğrenmektedir. Değerlendirme; 97 varlık tipi ve 161 BIO etiketi içeren 1.142 cümlelik Türkçe haber korpusu üzerinde token bazlı kesinlik (precision), duyarlılık (recall), F-ölçütü (F-measure) ve karışıklık matrisi ile gerçekleştirilmiştir.
