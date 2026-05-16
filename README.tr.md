# Türkçe Genişletilmiş Varlık Tanıma (ENER)

Bu depo, korpus analizi, kalite kontrolü, model eğitimi ve karşılaştırmalı değerlendirme için bir Türkçe Genişletilmiş Varlık Tanıma (ENER) iş akışı içerir. CoNLL tarzı anotasyon ayrıştırma, korpus istatistikleri, görselleştirme, BERT tabanlı token sınıflandırma, transformer gömme çıkarımı ve cümle düzeyinde çıkarım için Ortak Vektör Yaklaşımı (CVA) sınıflandırıcısını destekler.

## Proje Özeti

Proje iki tamamlayıcı hedef etrafında organize edilmiştir:

1. Anotasyonlu korpusu istatistikler, etiket kümesi kontrolleri ve görsel özetlerle karakterize etmek.
2. Aynı BIO-hizalı değerlendirme düzenini kullanarak ince ayarlı bir BERT NER modelini CVA tabanlı benzerlik sınıflandırıcısıyla karşılaştırmak.

Depo, Türkçe ENER verilerinin tekrarlanabilir analizini destekleyecek şekilde tasarlanmıştır ve boru hattını komut satırından çalıştırılabilecek kadar basit tutar.

## Temel Özellikler

- INCEpTION tarzı belge klasörleri için CoNLL anotasyon ayrıştırması.
- Token, cümle, varlık yoğunluğu, etiket sayıları ve varlık-tipi dağılımı gibi korpus istatistikleri.
- `data/ENER-tagset.tsv` ile etiket kümesi kalite kontrolü.
- Varlık sıklığı, varlık/cümle oranı ve cümle uzunluğu dağılımları için görselleştirmeler.
- İlk alttoken hizalaması ile BERT tabanlı token sınıflandırma.
- CVA sınıf vektörleri oluşturmak için transformer gömme çıkarımı.
- Sınıf vektörleri üzerinde kosinüs benzerliği kullanılarak CVA benzerlik sınıflandırması.
- Kesinlik, duyarlılık, F1, karışıklık matrisleri ve çıkarım zamanlaması için paylaşılan değerlendirme.

## Proje Yapısı

```text
extented-named-entity-recognation-ener/
├── data/
│   ├── annotation/             # CoNLL anotasyon klasörleri
│   ├── ENER-tagset.tsv         # QC için kullanılan kanonik etiket kümesi
│   ├── _smoke_train.conll      # Hızlı çalıştırmalar için küçük eğitim örneği
│   ├── _smoke_eval.conll       # Hızlı çalıştırmalar için küçük değerlendirme örneği
│   ├── full_train.conll        # Tam eğitim bölmesi
│   └── full_eval.conll         # Tam değerlendirme bölmesi
├── ner_stats/                  # Korpus analizi, etiket kümesi, span ve değerlendirme yardımcıları
├── scripts/
│   ├── run_analysis.py         # Korpus analizi CLI
│   └── train_ner.py            # Uçtan uca BERT + CVA boru hattı
├── results/                    # Oluşturulmuş analiz ve karşılaştırma çıktıları
├── results_example/            # Örnek özet çıktılar
├── outputs/                    # Model kontrol noktaları ve ara artefaktlar
├── requirements.txt
└── README.md
```

## Kurulum

### Gereksinimler

- Python 3.8 veya daha yenisi
- Git

### Kurulum

#### Windows PowerShell

```powershell
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux veya macOS

```bash
git clone https://github.com/MoussaBane/extented-named-entity-recognation-ener.git
cd extented-named-entity-recognation-ener
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

### 1. Korpus analizi

Korpus istatistiklerini hesaplamak, etiket kümesini doğrulamak ve grafikler oluşturmak için analiz hattını çalıştırın:

```bash
python scripts/run_analysis.py --data-root data/annotation --results-dir results
```

Beklenen çıktılar:

- `results/stats.json`
- `results/label_counts.csv`
- `results/type_counts.csv`
- `results/plots/*.png`
- `results/unused_tags_in_corpus.txt`
- `results/unknown_types_in_tagset_comparison.txt`

### 2. Uçtan uca BERT + CVA deneyi

Sağlanan smoke bölmesi üzerinde NER karşılaştırma hattını çalıştırın:

```bash
python scripts/train_ner.py --train-file data/_smoke_train.conll --eval-file data/_smoke_eval.conll --output-dir outputs/bert-ner-smoke --results-dir results/model_comparison_smoke
```

Beklenen çıktılar:

- `metrics_bert_with_o.json`
- `metrics_bert_without_o.json`
- `metrics_cva_with_o.json`
- `metrics_cva_without_o.json`
- `confusion_matrix_bert_with_o.csv`
- `confusion_matrix_cva_with_o.csv`
- `comparison_summary.json`
- `bio_validation_warnings.txt`

### 3. Kaydedilmiş model kontrol noktasını yeniden kullanma

Eğer `--skip-bert-train` ayarlanmış ve `--output-dir` bir Hugging Face kontrol noktası içeriyorsa, betik değerlendirme için bu kontrol noktasını yeniden kullanır:

```bash
python scripts/train_ner.py --train-file data/full_train.conll --eval-file data/full_eval.conll --output-dir outputs/bert-ner-full --results-dir results/model_comparison_full --skip-bert-train
```

Eğer `--output-dir` içinde yüklenebilir bir kontrol noktası yoksa, betik `--model-name` parametresine döner.

## Boru Hattı Açıklaması

1. Eğitim ve değerlendirme dosyalarından CoNLL BIO anotasyonlarını oku.
2. BIO etiketlerini doğrula ve etiket dizilerini normalleştir.
3. BERT ve CVA tarafından paylaşılan etiket haritasını oluştur.
4. Cümleleri tokenize et ve etiketleri her kelimenin ilk alttoken'ına hizala.
5. Bir transformer token-sınıflandırma modelini ince ayarla veya istenirse mevcut bir kontrol noktasını yükle.
6. Aynı transformer omurgasından token gömmelerini çıkar.
7. Hizalanmış eğitim gömmelerinden CVA sınıf vektörlerini oluştur.
8. Hem BERT hem de CVA ile değerlendirme etiketlerini tahmin et.
9. Kesinlik, duyarlılık, F1, doğruluk ve karışıklık matrislerini hesapla.
10. BERT ve CVA için cümle düzeyinde çıkarım hızını ölç.
11. Daha sonra raporlama için metrikleri, uyarıları, zamanlama özetlerini ve CSV/JSON çıktılarını kaydet.

## Yöntemler

### BERT tabanlı NER

BERT hattı, ilk alttoken hizalaması kullanarak token sınıflandırma için bir transformer modelini ince ayarlar. Her kelimenin yalnızca ilk alttoken'ı eğitim etiketlerine katkıda bulunur; bu, token düzeyindeki BIO denetimini kelime-segmentasyonu ile tutarlı tutar.

Varsayılan eğitim ayarları komut satırı argümanları aracılığıyla sağlanır; bunlar model adı, öğrenme hızı, batch boyutu, epoch sayısı, maksimum sıra uzunluğu ve seed'i içerir.

Örnek varsayılanlar:

- Model: `dbmdz/bert-base-turkish-cased`
- Öğrenme hızı: `2e-5`
- Eğitim batch boyutu: `8`
- Değerlendirme batch boyutu: `8`
- Epoch: `3`
- Seed: `42`

### Ortak Vektör Yaklaşımı (CVA)

CVA, transformer gömmelerini paylaşılan bir temsil uzayı olarak kullanır. Boru hattı:

1. Hizalanmış kelimeler için gömmeleri çıkarır.
2. Eğitim gömmelerini BIO etiketiyle gruplayarak toplar.
3. Bu etiket gruplarından sınıf vektörlerini hesaplar.
4. Her değerlendirme gömme'sini kosinüs benzerliğine göre en yakın sınıf vektörüne atar.

Bu, ince ayarlı sınıflandırıcıya hafif bir benzerlik tabanlı alternatif sağlar ve aynı veri bölmesinde çıkarım hızını karşılaştırılabilir kılar.

## Değerlendirme

Proje aşağıdaki değerlendirme artefaktlarını raporlar:

- Kesinlik, duyarlılık ve F1-skoru.
- `O` etiketi dahil veya hariç makro ve sınıf başına metrikler.
- BERT ve CVA tahminleri için karışıklık matrisleri.
- Her iki yöntem için cümle düzeyinde çıkarım zamanlaması.

Ana değerlendirme çıktıları JSON ve CSV olarak `results/model_comparison_*` içine yazılır; bu, makale tabloları veya şekiller için yeniden kullanılabilir.

### Karışıklık Matrisi

Karışıklık matrisleri CSV dosyaları olarak dışa aktarılır ve değerlendirme sırasında kullanılan etiket sırasını korur. Hata analizi, varlık başına başarısızlık incelemesi ve yayın şekilleri için kullanışlıdır.

### Hız Karşılaştırması

Boru hattı, hem BERT hem de CVA için ortalama ve toplam cümle çıkarım süresini ölçer. Bu, ince ayarlı sınıflandırıcı ile benzerlik tabanlı yöntemin pratik maliyetini karşılaştırmak için faydalıdır.

## Sonuç Özeti

Depo, `results_example/stats.json` içinde örnek korpus istatistikleri içerir.

Örnek sonuçlardan gözlemler:

- Toplam belge klasörü: 130
- Anotasyonlu belgeler: 34
- Anotasyonsuz belgeler: 96
- Toplam cümle: 1142
- Varlık içeren cümleler: 980
- Varlık-cümle oranı: 0.8581
- Toplam token: 29195
- BIO etiketleri: 161
- Varlık tipleri: 97
- Ortalama cümle uzunluğu: 25.56 token

Depo ayrıca etiket ve tip sayımları için örnek CSV çıktıları `results_example/` içinde içerir.

Bu README'de makale düzeyinde model skorları verilmemektedir çünkü depo henüz sonlandırılmış bir sonuç tablosu sağlamıyor. Tam sonuçlar için `results/model_comparison_*` altındaki üretilen dosyaları kullanın.

## Bilimsel Katkı

Bu depo, korpus analizi, model eğitimi ve benzerlik tabanlı sınıflandırmayı bir arada sunarak Türkçe ENER araştırmaları için yeniden kullanılabilir bir deneysel boru hattı sağlar. Aşağıda kullanışlıdır:

- Özel genişletilmiş NER etiket kümesinin yapısını belgelemek,
- Modellemeden önce anotasyon kalitesini doğrulamak,
- Aynı BIO değerlendirme protokolü altında BERT'i CVA karşısında kıyaslamak,
- Analiz artefaktlarını doğrudan makale için yeniden kullanılabilecek biçimde üretmek.

## Gelecek Çalışmalar

- Tam veri kümesi skorlarıyla makale hazır bir sonuç tablosu eklemek.
- Veri kaynağı, bölme politikası ve anotasyon yönergelerini belgelemek.
- Çoklu-seed değerlendirme ve anlamlılık testlerini eklemek.
- Depoyu karakter düzeyinde sınır tespiti deneyleriyle genişletmek.
- Proje hedeflerinde bahsedilen kontrastif öğrenme ve arttırma deneylerini eklemek.
- Genel kullanılabilirlik için lisans dosyası eklemek.

## Makale Tarzı Yöntemler ve Sonuçlar

### Yöntemler

Uçtan uca bir Türkçe Genişletilmiş Varlık Tanıma (ENER) boru hattı geliştirdik; bu boru hattı korpus analizi, sıra etiketleme ve benzerlik tabanlı sınıflandırmayı birleştirir. Girdi verileri CoNLL formatında saklanır ve INCEpTION tarzı anotasyon klasörlerinden ayrıştırılır. Modellemeden önce BIO etiketleri doğrulanır ve normalleştirilir; hem denetimli hem de benzerlik-tabanlı yaklaşımların aynı etiket uzayında çalışmasını sağlamak için paylaşılan bir etiket haritası oluşturulur.

Denetimli temel için, ilk alttoken hizalamasıyla bir transformer tabanlı token sınıflandırma modeli ince ayarlanır. Bu kuruluma göre her kelime yalnızca ilk alttoken'ını eğitim hedeflerine katkıda bulunur; bu, kelime düzeyindeki BIO denetimini korurken alt-kelime tokenizasyonu ile uyumlu kalır. Model, öğrenme hızı, batch boyutu, sıra uzunluğu, epoch sayısı ve rastgele seed dahil olmak üzere yapılandırılabilir hiperparametrelerle eğitilir. Değerlendirme, ayrılmış bir bölme üzerinde yapılır ve kesinlik, duyarlılık, F1-skoru ve karışıklık matrisi kullanılarak raporlanır.

Hafif bir alternatif sağlamak için Ortak Vektör Yaklaşımı (CVA) sınıflandırıcısı uygularız. Aynı transformer omurgası gömme çıkarıcı olarak kullanılır ve hizalanmış kelime gömmeleri BIO etiketine göre gruplanarak sınıf vektörleri hesaplanır. Çıkarım zamanında her gömme kosinüs benzerliğine göre en yakın sınıf vektörüne atanır. Bu, ince ayarlı sınıflandırma ile benzerlik tabanlı bir temel arasında doğrudan karşılaştırma sağlar.

Modellemenin yanı sıra, boru hattı korpus düzeyinde istatistikler ve kalite kontrol raporları hesaplar. Bunlar belge, cümle, token, varlık içeren cümle sayıları, BIO etiketleri ve varlık tipleri sayımlarını; ayrıca varlık sıklığı, varlık/cümle oranı ve cümle uzunluğu dağılımlarını gösteren grafiklerini içerir. Etiket kümesi, referans şema tarafından kapsanmayan kullanılmayan etiketleri ve tipleri belirlemek için kanonik ENER etiket kümesine karşı kontrol edilir.

### Sonuçlar

Depo `results_example/stats.json` içinde örnek korpus istatistikleri içerir. Bu özette, korpus 130 belge klasörü içerir; bunlardan 34'ü anotasyonlu ve 96'sı anotasyonsuzdur. Anotasyonlu bölüm 1.142 cümle ve 29.195 token içerir. Varlıklar 980 cümlede görülür; bu da 0.8581 varlık-cümle oranına karşılık gelir. Korpus 161 BIO etiketi, 97 farklı varlık tipi ve ortalama 25.56 token uzunluğunda cümlelere sahiptir.

Analiz hattı ayrıca etiket ve tip sıklığı tabloları ile cümle uzunluğu ve varlık dağılımı grafikleri üretir. Bu çıktılar, model eğitimi öncesi korpus incelemesi için betikler sağlar.

BERT ve CVA karşılaştırma hattı tam olarak uygulanmıştır ve her iki yöntem için de kesinlik, duyarlılık, F1-skoru, karışıklık matrisleri ve çıkarım zaman özetlerini dışa aktarır. Ancak, depo şu anda tam veri kümesi performans tablolarını içermediği için README içinde makale düzeyinde nihai model skorları raporlanmamıştır.

## Lisans

Depoda şu anda bir lisans dosyası bulunmamaktadır.
