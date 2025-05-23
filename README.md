# Akademik Makale Benzerlik Motoru
# Metin Tabanlı Veri Setleri ile Yapay Zekâ Modelleri Geliştirme ve Değerlendirme

## 📌 Proje Açıklaması

Bu proje, doğal dil işleme (NLP) teknikleri kullanılarak metin verisinin işlenmesi, analiz edilmesi ve çeşitli yapay zekâ modellerinin (TF-IDF, Word2Vec) geliştirilmesini ve bu modeller aracılığıyla metinler arası benzerliklerin hesaplanarak karşılaştırmalı bir şekilde değerlendirilmesini amaçlamaktadır. Projede, akademik makale özetlerinden oluşan `arxiv_sample_5000.csv` adlı veri kümesi kullanılmıştır.

## 🔧 Kullanılan Teknolojiler ve Kütüphaneler

- Python 3
- pandas, numpy
- matplotlib
- nltk
- scikit-learn (sklearn)
- gensim

---

## 📈 Proje Aşamaları

### **Ödev 1: Veri Ön İşleme ve Model Eğitimi**

#### 1. Veri Yükleme ve İnceleme
- `arxiv_sample_5000.csv` dosyası yüklendi ve veri setinin yapısı, boyutu gibi temel bilgiler incelendi.
- Örnek ham veriler görüntülendi.

#### 2. Ham Metin Üzerinden Zipf Analizi
- Veri setindeki tüm makale özetleri birleştirilerek kelime frekans analizi yapıldı.
- Kelime sıralaması ve frekansları kullanılarak log-log ekseninde Zipf diyagramı çizildi.

#### 3. Veri Ön İşleme
- Metinler üzerinde standart ön işleme adımları uygulandı:
    - HTML etiketleri (varsa) ve özel karakterler temizlendi.
    - Tüm metinler küçük harfe çevrildi.
    - İngilizce için yaygın kullanılan stopword’ler (etkisiz kelimeler) çıkarıldı.
- İki farklı normalleştirme tekniği uygulandı:
    - **Stemming (Kök Bulma):** Porter Stemmer kullanılarak kelimeler köklerine indirgendi.
    - **Lemmatization (Lemmalama):** WordNet Lemmatizer kullanılarak kelimelerin sözlükteki temel formları (lemma) bulundu.
- Hem stemming hem de lemmatization uygulanmış temizlenmiş veri setleri (`stemmed_data.csv`, `lemmatized_data.csv`) oluşturuldu ve kaydedildi.

#### 4. Temiz Veri Üzerinden Zipf Analizi
- Stemming ve lemmatization uygulanmış temiz veri setleri için ayrı ayrı Zipf analizleri tekrarlandı ve Zipf diyagramları çizilerek ön işlemenin kelime dağılımı üzerindeki etkisi incelendi.

#### 5. TF-IDF Vektörleştirme
- Hem stemmed hem de lemmatized temiz veri setleri kullanılarak TF-IDF (Term Frequency-Inverse Document Frequency) vektörleri hesaplandı.
    - *Parametreler:* `max_df=0.8` (dokümanların %80'inden fazlasında geçen kelimeler ihmal edildi), `min_df=5` (en az 5 dokümanda geçen kelimeler dikkate alındı).
- Oluşturulan TF-IDF matrisleri (`tfidf_stemmed.csv`, `tfidf_lemmatized.csv`) sonraki aşamalarda kullanılmak üzere kaydedildi.

#### 6. Word2Vec Model Eğitimi
- Hem stemmed hem de lemmatized temiz veri setleri üzerinde Word2Vec kelime gömme (word embedding) modelleri eğitildi.
- Farklı parametre kombinasyonları denendi:
    - **Model Türleri:** CBOW (Continuous Bag of Words) ve Skip-gram.
    - **Pencere Boyutları (Window Size):** 2 ve 4.
    - **Vektör Boyutları (Vector Size/Dimension):** 100 ve 300.
- Her bir ön işleme türü (stemmed/lemmatized) için 8 farklı konfigürasyonda olmak üzere toplam 16 adet Word2Vec modeli eğitildi ve `.model` dosyaları olarak kaydedildi.
- Her model eğitimi sonrası, örnek bir kelime ("data") için en benzer kelimeler listelendi.

---

### **Ödev 2: Eğitilen Modellerle Metin Benzerliği Hesaplama ve Değerlendirme**

#### 7. Giriş Metninin Belirlenmesi ve Hazırlanması
- `arxiv_sample_5000.csv` veri setinden rastgele bir makale özeti (`arXiv:2405.01502`) giriş metni olarak seçildi.
- Bu giriş metninin ham, stemmed ve lemmatized versiyonları hazırlandı.

#### 8. TF-IDF Modelleri ile Benzerlik Hesaplama
- Daha önce oluşturulan `tfidf_stemmed.csv` ve `tfidf_lemmatized.csv` matrisleri kullanıldı.
- Giriş metninin (ilgili ön işlenmiş haline göre) TF-IDF vektörü ile korpustaki diğer tüm dokümanların vektörleri arasında **Kosinüs Benzerliği** hesaplandı.
- Her iki TF-IDF modeli için giriş metnine en benzer ilk 5 doküman ve benzerlik skorları belirlendi.

#### 9. Word2Vec Modelleri ile Benzerlik Hesaplama
- Ödev 1'de eğitilen 16 adet Word2Vec modeli (`.model` dosyaları) teker teker yüklendi.
- Giriş metninin (modelin eğitildiği ön işleme türüne göre stemmed veya lemmatized token listesi) ve korpustaki diğer tüm dokümanların ortalama kelime vektörleri hesaplandı. (Modelin kelime dağarcığında olmayan kelimeler ortalamaya dahil edilmedi).
- Giriş metninin ortalama vektörü ile diğer dokümanların ortalama vektörleri arasında **Kosinüs Benzerliği** hesaplandı.
- Her bir 16 Word2Vec modeli için giriş metnine en benzer ilk 5 doküman ve benzerlik skorları belirlendi.

#### 10. Modellerin Değerlendirilmesi ve Karşılaştırılması
- **Anlamsal Değerlendirme (Subjective Evaluation):**
    - Tüm modellerin (2 TF-IDF + 16 Word2Vec) ürettiği ilk 5 benzer metin, giriş metniyle olan anlamsal yakınlıklarına göre subjektif olarak puanlandı (1-5 arası).
    - Her model için ortalama anlamsal puan hesaplandı.
    - Hangi modellerin daha başarılı olduğu, TF-IDF ile Word2Vec arasındaki farklar ve model yapılandırmalarının (ön işleme, algoritma, pencere boyutu, vektör boyutu) anlamsal başarıya etkileri yorumlandı.
- **Sıralama Tutarlılığı Değerlendirmesi (Ranking Agreement):**
    - Farklı modellerin aynı giriş metnine verdiği ilk 5 benzer doküman listeleri arasındaki örtüşme **Jaccard Benzerliği** kullanılarak ölçüldü.
    - 18x18 boyutunda bir Jaccard benzerlik matrisi oluşturuldu.
    - Matris yorumlanarak birbirine en çok benzeyen (tutarlı) sonuçlar üreten modeller belirlendi. Bu tutarlılığın anlamsal başarıyla ilişkisi incelendi.
    - Model yapılandırmalarının sıralama tutarlılığına etkileri analiz edildi.

---

## 📌 Notlar

- Projede NLTK (Natural Language Toolkit), scikit-learn ve Gensim gibi doğal dil işleme ve makine öğrenmesi için temel Python kütüphaneleri etkin bir şekilde kullanılmıştır.
- Tüm analizler ve modellemeler, `arxiv_sample_5000.csv` veri setindeki `abstract` kolonunda bulunan akademik makale özetleri üzerinden gerçekleştirilmiştir.
- Veri görselleştirmeleri (Zipf diyagramları) `matplotlib` kütüphanesi ile oluşturulmuştur.
- Benzerlik hesaplamalarında temel metrik olarak Kosinüs Benzerliği tercih edilmiştir.
- Jaccard Benzerliği, modellerin sıralama sonuçlarındaki tutarlılığı ölçmek için kullanılmıştır.

---
