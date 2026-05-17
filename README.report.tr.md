# Türkçe Genişletilmiş Varlık Tanıma (ENER)

---

## Proje Başlığı
**Türkçe Genişletilmiş Varlık Tanıma (ENER)**

---

## 1. Genel Bakış

Bu depo, Türkçe Genişletilmiş Varlık Tanıma (ENER) için araştırma düzeyinde, yeniden üretilebilir ve genişletilebilir bir boru hattı uygular. Titiz veri kalite araçları ile modelleme temel karşılaştırmalarını birleştirerek akademik deney, yöntem karşılaştırması ve sistem düzeyinde analizler için bir temel sağlar. Sistem şunları içerir:

- sağlam CoNLL / INCEpTION alımı ve BIO doğrulaması,
- korpus analizleri ve görselleştirme,
- dönüştürücü tabanlı token-sınıflandırma eğitim akışı (Hugging Face Transformers),
- temsile dayalı Ortak Vektör Yaklaşımı (CVA) prototip sınıflandırıcısı (SVD tabanlı),
- kelime-hizalanmış gömme çıkarımı için yeniden kullanılabilir, cihaz-duyarlı `TransformerEmbedder`,
- `--device` desteği, hızlı smoke deneyleri ve yeniden üretilebilir sonuç çıktıları sunan CLI araçları.

Bu depo, akademik yeniden üretilebilirlik, ortak araştırma ve sonraki çalışmalara (kontrastif öğrenme, karakter-düzey tespit, hiyerarşik taksonomiler ve yayın düzeyi deneyler) temel oluşturmak için uygundur.

---

## 2. Öne Çıkan Özellikler

- Uçtan uca boru hattı: alım → kalite kontrol → analiz → modelleme → değerlendirme.
- Uyarılar ve denetim kayıtlarıyla birlikte güvenilir BIO doğrulama ve normalizasyon yardımcıları.
- Korpus analizleri: cümle/token sayıları, varlık yoğunluğu, tür-frekansı ve grafikler.
- Dönüştürücü token sınıflandırma: ilk-alttoken hizalaması, HF `Trainer`, `seqeval`.
- CVA prototip sınıflandırma: sınıf ortalaması ve SVD-tabanlı ortak vektörler ile kosinüs benzerliği çıkarımı.
- Tekrar model yüklemelerini önleyen ve GPU/CPU çalışmasını destekleyen yeniden kullanılabilir, cihaz-duyarlı `TransformerEmbedder`.
- Üretim benzeri çalıştırmalar için `run_analysis.py`, `train_ner.py` gibi CLI betikleri ve `--device` bayrağı.
- Çıktılar JSON, CSV, PNG, karışıklık matrisleri vb. biçiminde deneyin yeniden üretilebilir olması için düzenlenir.
- Araştırma-öncelikli modüler yapı, uzantılar ve deneylerin uygulanmasını kolaylaştırır.

---

## 3. Depo Mimarisi

Depo, modüler katmanlı bir mimariyi takip eder:

- `data/` — ham ve kanonik veri artefaktları (CoNLL bölmeleri ve `ENER-tagset.tsv`).
- `ner_stats/` — çekirdek kütüphane:
  - `conll_reader.py` — dosya alımı
  - `data_utils.py` — BIO doğrulama, normalizasyon, etiket haritaları
  - `tagset.py` — etiket kümesi yükleme & karşılaştırma
  - `statistics.py` — korpus toplamları ve görselleştirme
  - `embeddings.py` — `TransformerEmbedder` (cihaz-duyarlı)
  - `cva.py` — prototip vektörleri, CVA, kosinüs sınıflandırıcı
  - `evaluation.py` — karışıklık matrisleri ve token düzeyinde metrikler
  - `spans.py` — token→karakter aralığı haritalama
  - `timing.py` — çıkarım zaman ölçümü
- `scripts/` — orkestrasyon CLI'ları:
  - `run_analysis.py` — korpus analitiği & kalite kontrol
  - `train_ner.py` — tam BERT + CVA deneyi
- `results/` & `outputs/` — deney çıktıları ve model checkpoint'leri
- `results_example/` — hızlı inceleme için örnek çıktılar
- `requirements.txt` — çalışma zamanı bağımlılıkları

Kavramsal veri akışı:

raw dosyalar (`data/`) -> `conll_reader` -> `data_utils` (doğrula/normalleştir) -> `statistics` + `tagset` QC
                                                                \
                                                                 -> tokenizasyon (HF tokenizer)
                                                                    -> model eğitimi (HF Trainer)
                                                                    -> gömme çıkarımı (`TransformerEmbedder`)
                                                                    -> CVA prototipleri (`cva.py`)
                                                                    -> değerlendirme (`evaluation.py`, `seqeval`)

---

## 4. Bilimsel Motivasyon

İnce taneli varlık taksonomileri, ilişki çıkarımı, bilgi tabanı doldurma ve domain-spesifik bilgi çıkarımı gibi görevler için daha yüksek değerli anlamsal etiketlemeler sağlar. Ancak ince taneli ENER etiket seyrekliği, anotasyon karmaşıklığı ve değerlendirme gereksinimlerini artırır. Bu depo tasarlanırken amaçlar:

1. Kanonik etiket kümesinin korpus kullanımına uygunluğunu sağlamak için yeniden üretilebilir kalite-kontrol adımları sunmak.
2. İki tamamlayıcı paradigma sunan temel modeller sağlamak: denetimli token sınıflandırma (dönüştürücü ince ayarı) ve zengin bağlamsal gömmeleri kullanan benzerlik/prototip tabanlı sınıflandırma (CVA).
3. Morfolojik karmaşıklıkla (karakter-düzey sınırlar), prototiplerin ve gömmelerin geliştirilmesi (kontrastif öğrenme) ve hiyerarşik ölçeklenebilir NER sistemleri tasarımı üzerine araştırma yapmak için bir platform sağlamak.

---

## 5. Türkçe NLP Zorlukları

Türkçe, NER için belirli ve ciddi zorluklar getirir:

- Birleşik (agglutinative) morfoloji: anlamı ve varlık üyeliğini değiştiren birçok ek içeren uzun kelimeler.
- Tokenizasyon duyarlılığı: altkelime tokenizer'ları (WordPiece/BPE) varlık gövdelerini altparçalara ayırabilir; alttokenlara basit etiket aktarımı, hizalama olmadan yanlış olur.
- Sınır belirsizliği: ekler ve klitik formlar varlık ifadelerinin kısmını kapsayabilir veya hariç bırakabilir; bu nedenle kesin anotasyon kuralları gerekir.
- Etiket seyrekliği: ince taneli türler uzun kuyruk dağılımlarına neden olur; birçok sınıf için örnek sayısı azdır.

Bu nedenle, sağlam token→karakter hizalaması, karakter-duyarlı modeller, prototip/istikrar teknikleri ve hiyerarşik modelleme Türkçe ENER için özellikle önemlidir.

---

## 6. Desteklenen ENER Taksonomisi

Depo, derin bir taksonomiyi temsil eden kanonik etiket kümesi `data/ENER-tagset.tsv` ile gelir (100+ alt tür örneği). Örnekler:

- LOC_CITY, LOC_REGION
- FAC_AIRPORT, FAC_STATION
- ORG_POLITICAL, ORG_FINANCIAL
- PRO_LANGUAGE, PRO_PROGRAM
- DISEASE, EVENT, ACT (aktivite), AGE ve daha fazlası.

`ner_stats/tagset.py`, bu etiket kümesini yükleme, öneklerine göre gruplama ve kanonik şemayı korpusta gözlenen türlerle karşılaştırma (kullanılmayan veya bilinmeyen türleri belirleme) yardımcıları sağlar.

---

## 7. Proje İş Akışı

Yüksek seviyeli boru hattı aşamaları:

1. INCEpTION tarzı klasörleri veya düz CoNLL dosyalarını al.
2. `data_utils.validate_bio_labels` ile BIO dizilerini doğrula — uyarıları ve normalize edilmiş çıktıları kaydet.
3. `statistics.compute_corpus_statistics` ile korpus istatistikleri ve grafikler üret.
4. İlk-alttoken hizalaması ile dönüştürücü token-sınıflandırıcı eğit veya yükle.
5. Cihaz-duyarlı `TransformerEmbedder` ile kelime-hizalanmış gömmeleri çıkar.
6. CVA sınıf vektörlerini (ortalama veya SVD ile çıkarılmış ortak vektörler) hesapla.
7. BERT ve CVA ile tahmin et, token- ve span-düzeyinde metrikleri hesapla, karışıklık matrisleri ve zamanlama raporları üret.
8. Sonuçları yeniden üretilebilir artefaktlar (JSON/CSV/PNG) olarak `results/` içinde kaydet.

---

## 8. Kurulum

Önkoşullar:

- Python 3.8+
- GPU kullanılıyorsa CUDA toolkit
- Git

Bağımlılıkları yükle:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

veya (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` temsilî olarak şunları içerir:

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

Yeniden üretilebilirlik için sürümleri bir lockfile içinde sabitleyin veya Poetry kullanın.

---

## 9. Ortam Kurulumu

- Bir Python sanal ortamı oluşturun ve bağımlılıkları yükleyin (Kurulum bölümüne bakın).
- GPU kullanıyorsanız `torch`'u CUDA desteği ile kurun ve gömme/model çıkarımı sırasında betiklere `--device cuda` verin.

Örnek (GPU):

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

---

## 10. Veri Seti Yapısı

`data/` içindeki asgari yapı:

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

- `annotation/` altındaki doküman klasörleri INCEpTION çıktıları içerir (birleşik metin + belge başına anotasyon CAS).
- `_smoke_*` bölmeleri hızlı yerel hata ayıklama ve CI smoke testleri için sağlanmıştır.

---

## 11. INCEP TION / CoNLL Entegrasyonu

- `conll_reader.py` parser'ı, token'in ilk boşluklu sütun ve son sütunun BIO etiketi olduğu CoNLL satırlarını bekler.
- Boş satırlar cümleleri ayırır.
- Boru hattı, korpus istatistikleri için `admin.conll`'u yetkili anotasyon dosyası olarak tercih eder; `INITIAL_CAS.conll` anotasyonsuz temel olarak bulunur.

---

## 12. Korpus Analiz Boru Hattı

Korpus analizini çalıştırın:

```bash
python scripts/run_analysis.py \
  --data-root data/annotation \
  --results-dir results
```

Çıktılar:

- `results/stats.json`
- `results/label_counts.csv`
- `results/type_counts.csv`
- `results/plots/top20_entity_types.png`
- `results/plots/entity_sentence_ratio.png`
- `results/plots/sentence_length_hist.png`
- `results/unused_tags_in_corpus.txt`
- `results/unknown_types_in_tagset_comparison.txt`

`statistics.py` şunları hesaplar:

- toplam doküman klasör sayısı
- anotasyonlu/anotasyonsuz doküman sayıları
- toplam cümle ve token sayıları
- en az bir varlık içeren cümle sayısı ve varlık-cümle oranı
- ortalama cümle uzunluğu
- etiket ve tür frekans sayaçları

---

## 13. Dönüştürücü Tabanlı NER Eğitimi

`train_ner.py` tam bir Dönüştürücü eğitim + değerlendirme akışı uygular:

- Tokenizasyon, `AutoTokenizer` kullanılarak `is_split_into_words=True` ile yapılır.
- Etiket hizalaması ilk-alttoken hizalamasını kullanır: yalnızca ilk alttoken etiketi alır; sonraki alttokenlar `-100` ile maskelenir.
- Model: `AutoModelForTokenClassification` (konfigüre edilebilir `--model-name`), Hugging Face `Trainer` ile ince ayar yapılır.
- Sıra metrikleri: eğitim sırasında `seqeval` (varlık-düzeyinde precision/recall/F1) kullanılır; token-düzeyi metrikler/karışıklık matrisleri çevrimdışı hesaplanır.
- Model checkpoint'leri `Trainer` konvansiyonlarını kullanır ve `--skip-bert-train` ile önceden eğitilmiş bir checkpoint yeniden kullanılabilir.

Smoke çalıştırma örneği:

```bash
python scripts/train_ner.py \
  --train-file data/_smoke_train.conll \
  --eval-file data/_smoke_eval.conll \
  --device cuda \
  --output-dir outputs/bert-ner-smoke \
  --results-dir results/model_comparison_smoke
```

Ana eğitim ayrıntıları:

- Varsayılan model: `dbmdz/bert-base-turkish-cased`
- Varsayılan öğrenme hızı: `2e-5`
- Batch boyutları ve epoch sayısı CLI argümanlarıyla yapılandırılabilir
- BIO uyarıları normalize edilmeden önce ve sonra `results/` içine yazılır

---

## 14. CVA Prototip Sınıflandırması

CVA temel yöntemi, aynı dönüştürücü omurgasından çıkarılan gömmeleri kullanarak sınıf prototiplerini oluşturan ve kosinüs benzerliğine göre sınıflandırma yapan bir temsil tabanlı sınıflandırıcıdır.

Gerekçe: CVA, embedding geometrisinin (görev-özgü ince ayar olmadan) genişletilmiş varlık sınıflarını ne kadar ayırabildiğini test eden hafif bir temel sağlar. Düşük-kaynak ve uzun kuyruk rejimleri için özellikle ilgilidir.

Boru hattı adımları:

1. Tüm eğitim tokenları için (ilk-alttoken gömmeleri) kelime-hizalanmış gömmeleri çıkar.
2. Gömmeleri sınıf etiketiyle gruplandırıp her sınıf için `X_c ∈ ℝ^{n_c × d}` matrisini oluştur.
3. Sınıf prototipini hesapla:
   - Ortalama vektör: basit prototip
   - CVA ortak vektörü: sınıf içi altuzaya (SVD ile elde edilen) projeksiyonun çıkarılmasıyla sınıf "çekirdeğini" izole et
4. Değerlendirme token gömmelerini sınıf prototipleriyle kosinüs benzerliğine göre sınıflandır.

Matematiksel özet:

- Kosinüs benzerliği:

```math
\mathrm{cosine}(a,b)=\frac{a^\top b}{\|a\|\|b\|}
```

- Merkezi sınıf matrisi için SVD:

```math
A = U \Sigma V^\top
```

- Sınıf ortalaması \(\bar{x}\) için, iç-sınıf altuzayına projeksiyonun çıkarılması:

```math
p = V_r V_r^\top \bar{x}
\quad
\text{common_vector} = \bar{x} - p
```

Geri dönüş: SVD sıfır sıra veya sayısal kararsızlık gösterirse, ortalama vektöre geri dönülür.

Avantajlar:

- Küçük-orta ölçekli veri setleri için hesaplaması basit.
- Yorumlanabilir prototipler ve hızlı çıkarım (kosinüs aramaları).
- Denetimli ince ayara tamamlayıcı bir temel sunar.

Sınırlamalar:

- Sınıf başına SVD ölçeklendiğinde pahalı olabilir.
- Çok-modal sınıf dağılımları birden fazla prototipe veya karışım modellerine ihtiyaç duyar.
- Nadir sınıflar gürültülü prototiplere yol açabilir.

---

## 15. Cihaz-Duyarlı Gömme Refaktörü

Tekrarlayan dönüştürücü yüklemelerini ortadan kaldırmak ve GPU kullanımını iyileştirmek için `TransformerEmbedder` (`ner_stats/embeddings.py`) yeniden kullanılabilir ve cihaz-duyarlı şekilde uygulandı.

Tasarım hedefleri:

- `AutoTokenizer` ve `AutoModel`'ı tek bir noktadan başlatıp modeli belirtilen cihaza (`cpu` veya `cuda`) yerleştirmek.
- `encode_sentence_subwords()` ve `encode_words()` gibi çağrılarının gömülere embedder'ın cihazında döndürülmesini sağlamak.
- CVA ve zamanlama fonksiyonlarına tekrar kullanılmak üzere bir `embedder` örneğinin geçirilmesine izin vermek.
- Sağlanan embedder yoksa geriye dönük uyumluluk için geçici embedder oluşturan yardımcılar sunmak.

Neden önemli:

- Tek bir embedder kullanmak, aynı HF modelinin tekrar tekrar yüklenmesinden doğan I/O ve bellek giderlerini ortadan kaldırır.
- Cihaza özel yerleştirme, ana bilgisayar-cihaz transferlerini azaltır ve GPU kullanıldığında verimi önemli ölçüde artırır.
- Gömme çıkarımı deterministik olduğundan deney yeniden üretilebilirliği kolaylaşır (seed ve checkpoint sabitliyine bağlı olarak).

Kullanım örneği:

```python
from ner_stats.embeddings import TransformerEmbedder

embedder = TransformerEmbedder(model_name="dbmdz/bert-base-turkish-cased", device="cuda")
tokens, embeddings = embedder.encode_sentence_subwords("Ankara havalimanı açık.")
```

---

## 16. Değerlendirme Metrikleri

- Sıra-düzeyi: `seqeval` metrikleri (varlık-düzeyinde precision, recall, F1).
- Token-düzeyi: karışıklık matrisi, sınıf başına precision/recall/F1, doğruluk (`ner_stats/evaluation.py` tarafından hesaplanır).
- Zamanlama: BERT ve CVA için cümle başına ortalama ve toplam çıkarım zamanı (warmup dahil) ölçülür.
- Raporlama: sonuçlar JSON (metrikler) ve CSV (karışıklık matrisleri) olarak dışa aktarılır, böylece tablo ve istatistiksel toplama yapılabilir.

---

## 17. Örnek Komutlar

Korpus analizi:

```bash
python scripts/run_analysis.py \
  --data-root data/annotation \
  --results-dir results \
  --device cpu
```

Transformers NER eğitimi (smoke):

```bash
python scripts/train_ner.py \
  --train-file data/_smoke_train.conoll \
  --eval-file data/_smoke_eval.conoll \
  --device cuda \
  --output-dir outputs/bert-ner-smoke
```

Önceden eğitilmiş bir checkpoint'i yeniden kullan (eğitimi atla):

```bash
python scripts/train_ner.py \
  --train-file data/full_train.conoll \
  --eval-file data/full_eval.conoll \
  --skip-bert-train \
  --output-dir outputs/bert-ner-full \
  --device cuda \
  --results-dir results/model_comparison_full
```

---

## 18. Depo Yapısı Ağacı

```text
extented-named-entity-recognation-ener/
├── data/                       # annotation klasörleri, tagset, train/eval bölmeleri
├── ner_stats/                  # çekirdek kütüphane: parserlar, yardımcılar, gömmeler, CVA, değerlendirme
│   ├── __init__.py
│   ├── conll_reader.py
│   ├── data_utils.py
│   ├── tagset.py
│   ├── statistics.py
│   ├── embeddings.py          # cihaz-duyarlı TransformerEmbedder
│   ├── cva.py                 # CVA, prototipler, kosinüs sınıflandırıcı
│   ├── evaluation.py
│   ├── spans.py
│   └── timing.py
├── scripts/
│   ├── run_analysis.py         # korpus analitiği + kalite kontrol
│   └── train_ner.py            # uçtan uca BERT + CVA boru hattı
├── outputs/                    # model checkpointleri / HF-benzeri klasörler
├── results/                    # üretilen istatistikler, grafikler, metrikler, karışıklık matrisleri
├── results_example/            # örnek çıktılar
├── requirements.txt
└── README.report.md            # İngilizce rapor
```

Her klasörün rolü:

- `data/` — kanonik veri ve hızlı iterasyon için smoke bölmeleri.
- `ner_stats/` — araştırma kütüphanesi; betikler ve deneylerde import edilmek üzere tasarlandı.
- `scripts/` — deneyleri yeniden üretmek için kullanıcıya yönelik giriş noktaları.
- `outputs/` — HF Hub'a yüklenebilecek model artefaktları.
- `results/` — raporlama ve şekiller için deterministik çıktıların saklandığı yer.

---

## 19. Matematiksel Temeller

### BIO etiketleme

BIO, varlık span sınırlarını ve türünü token-düzeyinde kodlar. Tokenlar \(x_1, \ldots, x_n\) için etiketler \(y_i \in \{O\} \cup \{B\text{-}t, I\text{-}t : t \in \mathcal{T}\}\) şeklindedir; burada \(\mathcal{T}\) taksonomidir. Geçerli geçişler, tutarlı spanları temsil etmek için BIO kısıtlarına uymalıdır.

### Dönüştürücü token sınıflandırması

Alttoken gömmeleri \(h_i \in \mathbb{R}^d\) için sınıflandırma logits:

```math
z_i = W h_i + b
```

\(z_i\) üzerinde softmax, token-başına dağılımlar verir; çapraz entropi kaybı yalnızca ilk-alttoken pozisyonlarında uygulanır.

### Kosinüs benzerliği ile sınıflandırma

Prototip \(v_c\) ve sorgu gömme \(q\) için:

```math
\mathrm{cosine}(q, v_c) = \frac{q^\top v_c}{\|q\| \|v_c\|}
```

Tahmin: \(\hat{c} = \arg\max_c \mathrm{cosine}(q, v_c)\).

### SVD ve CVA

Merkezi sınıf matrisi \(A \in \mathbb{R}^{n \times d}\) için kompakt SVD:

```math
A = U \Sigma V^\top
```

Üst r sağ-singular vektörleri \(V_r\) alındığında, ortalama \(\bar{x}\)'in iç-sınıf altuzayına projeksiyonu \(p = V_r V_r^\top \bar{x}\) olur. CVA ortak vektörü \(c = \bar{x} - p\) olarak hesaplanır.

---

## 20. Araştırma Yol Haritası

Depo, aşağıdaki araştırma maddelerini desteklemek üzere tasarlanmıştır (kısa açıklamalar):

- **Çoklu-seed deneyleri & istatistiksel testler**: eğitimi birden fazla seed ile çalıştır, ortalama±std ve güven aralıklarını topla, BERT ile CVA arasındaki farklar için anlamlılık testleri uygula (paired bootstrap / t-test).
- **Prototip sağlamlığı**: ortalama vs. CVA prototiplerini, prototip küçültmeyi ve çok-modal sınıflar için prototip karışımlarını değerlendir.
- **Kontrastif öğrenme**: encoder'ı denetimli veya denetimsiz kontrastif amaçlarla ön-eğiterek, uzun kuyruk sınıflarda prototip ayırma kabiliyetini artır.
- **Hiyerarşik sınıflandırma**: `ENER-tagset.tsv` öneklerinden ebeveyn-çocuk ontolojisi çıkarıp, kaba→ince boru hatları uygulama (ör. önce FAC vs LOC, sonra alt tür).
- **Karakter düzeyinde sınır tespiti**: token+karakter hibrit modelleri uygulayarak agglutinatif Türkçede sınır tespitini iyileştir.
- **Hugging Face dataset & model yayınlama**: veri kümesi meta verilerini standartlaştır, dataset card hazırla, modelleri ve betikleri HF Hub'a yükle.

---

## 21. Performans ve Ölçeklenebilirlik Notları

- **Gömme yeniden kullanımı**: Tek bir `TransformerEmbedder` örneğinin yeniden kullanılması bellek ayak izini önemli ölçüde azaltır ve aynı HF modelinin tekrar başlatılmasını engeller.
- **Cihaz-duyarlı yürütme**: model ve girdilerin aynı cihaza yerleştirilmesi ana-cihaz senkronizasyon maliyetlerinden kaçınır ve çıkarım verimini artırır.
- **CVA SVD maliyeti**: Sınıf-başı SVD'nin hesaplama maliyeti \(O(\min(n d^2, n^2 d))\) şeklindedir. Büyük sınıflar veya yüksek boyutlar için rastgeleleştirilmiş/truncate SVD (`sklearn.utils.extmath.randomized_svd`) veya artımlı PCA önerilir.
- **Batching & önbellekleme**: Embedder yöntemleri toplu (batched) girişleri işleyebilecek şekilde genişletilebilir; prototipleri disk'e (`np.save`) kaydederek tekrar hesaplamadan kaçınılabilir.
- **Önerilen donanım**: Tam deneyler için ≥16GB GPU, büyük CVA hesapları için 64GB+ RAM veya dağıtık/truncate SVD önerilir.

---

## 22. Gelecek Çalışmalar

Kısa vadeli (mühendislik):

- Birim testleri ekle (BIO normalizasyonu, span hizalaması, embedding API'leri).
- `_smoke_*` bölmeleri ile GitHub Actions tabanlı CI smoke-run testleri oluştur.
- Ortam yeniden üretilebilirliği için bağımlılık kilit dosyası ekle.

Orta vadeli (araştırma):

- Kontrastif ön-eğitim ile gömme iyileştirmeleri.
- Etiket taksonomisi kullanarak hiyerarşik sınıflandırma deneyleri.
- Karakter-düzey ve span-tabanlı modellerle token-düzey BIO performansını karşılaştır.

Uzun vadeli (yayın & dağıtım):

- HF Dataset ve Model Hub artefaktları, dataset_card ve model_card ile birlikte sağla.
- Deney takibi entegrasyonu (MLflow / Weights & Biases) ekle.
- Anotasyon QA ve insan-düzeyinde düzeltme için hafif bir web UI geliştir.

---

## 23. Atıf

Bu kodu veya veri setini araştırmalarınızda kullanırsanız, lütfen depoyu ve ENER veri setini referans verin. (Bir DOI veya resmi referans hazır olduğunda atıf bloğunu ekleyin.)

---

## 24. Lisans

Mevcut depo anlık görüntüsünde lisans yer almamaktadır. Kamusal dağıtım öncesi uygun bir lisans (MIT, Apache-2.0 veya benzeri) ekleyin; veri izinleri ve kurumsal gereksinimlerle uyumlu olmalıdır.

---

## 25. Teşekkür

Bu depo, Türkçe ENER denemeleri için bir araştırma platformu olarak geliştirildi. Veri sağlayıcılar, anotatörler ve finansman kuruluşları için teşekkür notları eklendiğinde burada belirtiniz.

---

## 26. Ek: CI/test planları ve detaylı araştırma yol haritaları

### CI / Test Planları

- `tests/` altında birim testleri ekle:
  - `data_utils.normalize_bio_sequence` (kenar durumlar, bozuk BIO dizileri)
  - `spans.bio_to_character_spans` (tekrar eden tokenlar ve noktalama ile hizalama testleri)
  - `embeddings.TransformerEmbedder` (cihaz yerleşimi ve toplu çağrı davranışı)
- GitHub Actions hattı:
  - `lint` (flake8/black)
  - `unit-tests` (`pytest` ile `_smoke_*` veri setleri kullanılarak çalıştırma)
  - `smoke-run` (`scripts/run_analysis.py` ve `scripts/train_ner.py`'yi smoke argümanlarla çalıştırma)

### Kontrastif Öğrenme Yol Haritası

- Denetimli kontrastif kaybı uygulayın; pozitif çiftler aynı-sınıf mention'lardır (augmentasyonlu) ve negatifler sınıflar arası örneklenir.
- Augmentasyonlar: morfolojik ek perturbasyonu, back-translation, rastgele token maskesi.
- CVA prototip ayrışmasını değerlendirmek için küme metrikleri (silhouette, Davies-Bouldin) kullanın.

### Hiyerarşik Sınıflandırma Yol Haritası

- `ENER-tagset.tsv`'den ebeveyn-çocuk ontolojisi çıkarın (önek → çocuklar).
- Kaba sınıflandırıcıyı önekler için (ör. FAC vs LOC) ve koşullu ince sınıflandırıcıyı alt türler için uygulayın.
- Ortak encoder ve ayrı başlıklarla çok görevli öğrenme yaklaşımlarını keşfedin.

### Karakter Düzey Modelleme Yol Haritası

- Karakter-seviye kodlayıcı (CNN/BiLSTM) inşa edin ve token gömmeleriyle birleştirin.
- Başlangıç/bitiş indekslerini tahmin eden saf karakter-span dedektörü uygulayın ve sınıf etiketini atayın.
- Token-düzeyi BIO tabanlı baseline ile sınır hatırlama ve mikro/makro F1 karşılaştırmaları yapın.

### Hugging Face Yayın Planları

- `dataset_card` ve `dataset_infos.json` ile HF Dataset formatına uygunluk sağlayın.
- Eğitim/değerlendirme bölmelerini standart JSON/CoNLL biçiminde dışa aktarın.
- En iyi checkpoint'leri ve tokenizer'ı HF Hub'a yükleyin, `model_card` içinde eğitim rejimini ve veri kökenini açıklayın.

---

## İletişim / Katkı

Katkılar memnuniyetle karşılanır. İyileştirme, hata düzeltme veya deney önerileri için lütfen issue veya pull request açın. Anotasyon politikası, veri kökeni veya deney tasarımı ile ilgili sorular için issue açarak tartışmayı izleyelim.

---

Bu `README.report.tr.md` dosyası hem deneyleri yeniden üretmek isteyen kullanıcılar için bir kılavuz hem de araştırma amaçlı açıklayıcı bir rapor olarak düzenlenmiştir.
