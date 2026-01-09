# Türkçe Duygu Analizi - Transformer Modelleri Karşılaştırması

Türkçe metinler üzerinde duygu analizi yapmak için çeşitli Transformer tabanlı modellerin performansını karşılaştıran kapsamlı bir makine öğrenmesi projesi.

## 🎯 Proje Hakkında

Bu proje, Türkçe metinlerdeki duyguları (pozitif, negatif, nötr) tespit etmek için üç farklı Transformer modelinin performansını analiz eder. Çalışma, farklı veri boyutlarında (5.000 - 20.000 örnek) model performanslarını karşılaştırarak en uygun model ve veri setini belirlemeyi amaçlar.

### Temel Özellikler
- ✅ Dengeli veri seti örneklemesi
- ✅ Türkçeye özel metin ön işleme
- ✅ GPU hızlandırma desteği (Tesla T4)
- ✅ Kapsamlı performans metrikleri
- ✅ Epoch bazlı detaylı izleme

## 🤖 Kullanılan Modeller

### 1. XLM-RoBERTa Base
- **Model:** `FacebookAI/xlm-roberta-base`
- **Açıklama:** Çok dilli RoBERTa modeli, 100 farklı dili destekler
- **Avantajları:** Cross-lingual transfer learning

### 2. BERTurk
- **Model:** `dbmdz/bert-base-turkish-cased`
- **Açıklama:** Türkçe için özel eğitilmiş BERT modeli
- **Avantajları:** Türkçe dil yapısına optimize

### 3. DistilBERTurk
- **Model:** `dbmdz/distilbert-base-turkish-cased`
- **Açıklama:** BERTurk'ün hafif ve hızlı versiyonu
- **Avantajları:** %40 daha küçük, %60 daha hızlı

## 📊 Veri Seti

### Kaynak
- **Dataset:** `winvoker/turkish-sentiment-analysis-dataset`
- **Platform:** Hugging Face
- **Orijinal Boyut:** 40.000+ örnek

### Sınıf Dağılımı
Dengeli örnekleme ile her sınıftan eşit sayıda veri:

### Deneysel Veri Boyutları
- 5.000 örnek
- 10.000 örnek
- 15.000 örnek
- 20.000 örnek

## 🛠️ Kurulum

### Google Colab Üzerinde Çalıştırma
1. Notebook'u Google Colab'a yükleyin
2. Runtime > Change runtime type > GPU (T4) seçin
3. Tüm hücreleri sırayla çalıştırın

## 🚀 Kullanım

### 1. Veri Yükleme
```python
from datasets import load_dataset

dataset = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
df = dataset['train'].to_pandas()
```

### 2. Veri Temizleme
```python
# Otomatik temizleme fonksiyonu
temiz_metin = veri_temizleme(metin)
```

Temizleme adımları:
- Küçük harfe çevirme
- Kullanıcı adı temizleme (@mentions)
- Noktalama ve sayı kaldırma
- Türkçe stopwords temizliği
- Stemming (kök bulma)

### 3. Model Eğitimi
```python
# Örnek: XLM-RoBERTa ile eğitim
model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base", 
    num_labels=3
)

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

## 📈 Metodoloji

### Veri Ön İşleme Süreci
1. **Metin Normalleştirme**
   - Tüm metinler küçük harfe çevrilir
   - Özel karakterler ve noktalama işaretleri kaldırılır

2. **Stopwords Temizliği**
   - NLTK Türkçe stopwords listesi kullanılır
   - Anlamsız kelimeler filtrelenir

3. **Stemming**
   - TurkishStemmer ile kelime köklerine indirgeme
   - Kelime çeşitliliğini azaltma

4. **Tokenizasyon**
   - Model-spesifik tokenizer'lar kullanılır
   - Max length: 128 token
   - Padding ve truncation uygulanır

### Eğitim Parametreleri
- **Epoch:** 3
- **Batch Size:** 16
- **Learning Rate:** Otomatik (AdamW)
- **Train/Test Split:** 80/20
- **Evaluation Strategy:** Her epoch sonunda

### Performans Metrikleri
- **Accuracy:** Genel doğruluk oranı
- **F1-Score:** Precision ve Recall'ın harmonik ortalaması
- **Precision:** Pozitif tahminlerin doğruluğu
- **Recall:** Gerçek pozitifleri bulma oranı
- **Loss:** Training ve Validation loss

## 📊 Sonuçlar

### 5.000 Veri ile Sonuçlar (3. Epoch)
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| XLM-RoBERTa | 0.8040 | 0.8029 | 0.8022 | 0.8040 |
| BERTurk | **0.8710** | **0.8708** | **0.8707** | **0.8710** |
| DistilBERTurk | 0.8690 | 0.8680 | 0.8678 | 0.8690 |

### 10.000 Veri ile Sonuçlar (3. Epoch)
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| XLM-RoBERTa | 0.8665 | 0.8658 | 0.8656 | 0.8665 |
| BERTurk | **0.9070** | **0.9067** | **0.9065** | **0.9070** |
| DistilBERTurk | 0.8775 | 0.8769 | 0.8767 | 0.8775 |

### 15.000 Veri ile Sonuçlar (3. Epoch)
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| XLM-RoBERTa | 0.8710 | 0.8701 | 0.8698 | 0.8710 |
| BERTurk | **0.9037** | **0.9034** | **0.9034** | **0.9037** |
| DistilBERTurk | 0.8800 | 0.8794 | 0.8790 | 0.8800 |

### 20.000 Veri ile Sonuçlar (3. Epoch)
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| XLM-RoBERTa | 0.8892 | 0.8888 | 0.8886 | 0.8892 |
| BERTurk | **0.9130** | **0.9130** | **0.9130** | **0.9130** |
| DistilBERTurk | 0.8938 | 0.8936 | 0.8935 | 0.8938 |

### Temel Bulgular

#### 🏆 En İyi Performans: BERTurk
- Tüm veri boyutlarında en yüksek accuracy
- 20.000 veri ile **%91.3 accuracy** (en yüksek performans)
- Türkçe dil yapısına özgü eğitim avantajı
- F1-Score'da tutarlı üstünlük

#### ⚡ En Hızlı Model: DistilBERTurk
- BERTurk'e yakın performans (%89.4 ile %91.3 arasında)
- ~%50 daha hızlı eğitim süresi
- Üretim ortamları için ideal alternatif

#### 🌍 Çok Dilli Alternatif: XLM-RoBERTa
- Kabul edilebilir performans (%80-89 arası)
- Cross-lingual transfer learning
- Çok dilli projelerde kullanılabilir

### Veri Boyutu Etkisi
- **5.000 → 10.000:** Ortalama **~5-6% performans artışı** (en büyük sıçrama)
- **10.000 → 15.000:** Ortalama **~0.5% performans artışı** (düşüş)
- **15.000 → 20.000:** Ortalama **~1.5% performans artışı**
- **Optimal veri boyutu:** 20.000 örnek (maksimum performans için)
- **Maliyet-Etkin seçim:** 10.000 örnek (iyi performans/hız dengesi)

### Model Performans Gelişimi (Veri Boyutuna Göre)

**BERTurk Gelişimi:**
- 5K: 87.1% → 10K: 90.7% → 15K: 90.4% → 20K: **91.3%**
- En stabil ve tutarlı performans

**DistilBERTurk Gelişimi:**
- 5K: 86.9% → 10K: 87.8% → 15K: 88.0% → 20K: **89.4%**
- BERTurk'e en yakın performans/hız dengesi

**XLM-RoBERTa Gelişimi:**
- 5K: 80.4% → 10K: 86.7% → 15K: 87.1% → 20K: **88.9%**
- En büyük gelişim gösterdi (5K'dan 20K'ya +8.5%)

## 🔬 Teknolojiler

### Kütüphaneler
- **Transformers:** Hugging Face model hub
- **PyTorch:** Deep learning framework
- **Datasets:** Veri yükleme ve işleme
- **scikit-learn:** Metrikler ve train/test split
- **NLTK:** Türkçe NLP işlemleri
- **TurkishStemmer:** Türkçe kök bulma

### Araçlar
- **Google Colab:** GPU-accelerated notebook ortamı
- **Weights & Biases:** Devre dışı (opsiyonel tracking)
- **Matplotlib & Seaborn:** Görselleştirme

## 📁 Proje Yapısı

```
dataSci_Proje_4.ipynb
├── 1. Veri Seti Çekme ve Kütüphane Kurulumu
├── 2. Veri Temizleme Fonksiyonu
├── 3. Veri Yükleme ve İnceleme
├── 4. Etiket Dönüşümü ve Veri Bölme
├── 5. Model Eğitimi
│   ├── 5.1 XLM-RoBERTa (5k, 10k, 15k, 20k)
│   ├── 5.2 BERTurk (5k, 10k, 15k, 20k)
│   └── 5.3 DistilBERTurk (5k, 10k, 15k, 20k)
└── 6. Sonuç Analizi ve Karşılaştırma
```

## 💡 Öneriler

### Üretim Ortamı İçin
1. **Yüksek Doğruluk Önceliği:** BERTurk kullanın
2. **Hız Önceliği:** DistilBERTurk kullanın
3. **Çok Dilli Destek:** XLM-RoBERTa kullanın

### İyileştirme Fikirleri
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Data augmentation teknikleri
- [ ] Ensemble model yaklaşımı
- [ ] Fine-tuning için domain-specific veri
- [ ] Class imbalance handling (weighted loss)
- [ ] Cross-validation implementasyonu

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

**Furkan Poyraz**
- Computer Engineering Student
- Machine Learning & NLP Enthusiast

## 🙏 Teşekkürler

- Hugging Face ekibine Transformers kütüphanesi için
- dbmdz ekibine BERTurk modelleri için
- winvoker'a Türkçe sentiment analysis dataset için

---

**Not:** Bu proje Google Colab üzerinde Tesla T4 GPU ile test edilmiştir. Farklı ortamlarda performans süreleri değişiklik gösterebilir.
