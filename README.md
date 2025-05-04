# Akademik-Makale-Benzerlik-Motoru
# Akademik-Makale-Benzerlik-Motoru
# Metin Tabanlı Veri Setleri ile Yapay Zekâ Modelleri Geliştirme

## 📌 Proje Açıklaması

Bu proje, doğal dil işleme (NLP) teknikleri kullanılarak metin verisinin işlenmesi, analiz edilmesi ve çeşitli yapay zekâ modellerinin (TF-IDF, Word2Vec) geliştirilmesini amaçlamaktadır. Projede `arxiv_sample_5000.csv` adlı akademik özetlerden oluşan veri kümesi kullanılmıştır.

## 🔧 Kullanılan Teknolojiler ve Kütüphaneler

- Python 3
- pandas, numpy
- matplotlib
- nltk
- sklearn
- gensim

---

## 📈 Proje Aşamaları

### 1. Veri Yükleme ve İnceleme
- `arxiv_sample_5000.csv` dosyası yüklendi ve temel bilgiler incelendi.

### 2. Ham Metin Üzerinden Zipf Analizi
- Tüm özetler birleştirilerek frekans analizi yapıldı.
- Log-log ekseninde Zipf diyagramı çizildi.

### 3. Veri Ön İşleme
- HTML etiketleri ve özel karakterler temizlendi.
- Küçük harfe çevrildi, stopword’ler çıkarıldı.
- Hem stemming hem de lemmatization uygulandı.
- Temiz veri `.csv` dosyalarına kaydedildi.

### 4. Temiz Veri Üzerinden Zipf Analizi
- Hem stemmed hem lemmatized veri için Zipf grafikleri yeniden çizildi.

### 5. TF-IDF Vektörleştirme
- Her iki temiz veri üzerinde TF-IDF hesaplandı.
- Çıktılar `.csv` olarak kaydedildi.

### 6. Word2Vec Model Eğitimi
- CBOW ve Skip-gram olmak üzere iki model türü kullanıldı.
- 2 ve 4 pencere (window) boyutu; 100 ve 300 vektör boyutu test edildi.
- Her parametre kombinasyonu için 8 model `.model` dosyası olarak kaydedildi.

---

## 📌 Notlar

- Projede NLTK ve Gensim gibi NLP için temel kütüphaneler kullanılmıştır.
- Çalışmalar `abstract` kolonunda bulunan metinler üzerinden gerçekleştirilmiştir.
- Grafikler `matplotlib` ile oluşturulmuştur.

---
