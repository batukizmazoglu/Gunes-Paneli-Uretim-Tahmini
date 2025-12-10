# ☀️ Güneş Enerjisi Üretim Tahmini ve Enerji Verimliliği Öneri Sistemi

Bu proje, hava durumu verilerini kullanarak ev tipi güneş panellerinin enerji üretimini tahmin eden ve bu tahminlere dayanarak kullanıcılara **en verimli enerji tüketim saatlerini** öneren bir yapay zeka uygulamasıdır. 

Sosyal Sorumluluk Dersi kapsamında **Semih DEMİR** ve **Batu KIZMAZOĞLU** tarafından geliştirilmiştir.

---

## 🚀 Projenin Amacı

Projenin temel amacı, yenilenebilir enerji kaynaklarının verimliliğini artırmaktır. Sistem, geçmiş üretim verileri ve hava durumu parametrelerini (sıcaklık, bulutluluk, radyasyon) analiz ederek gelecekteki üretimi tahmin eder. 

**Kullanıcıya Sağladığı Faydalar:**
- ⚡ **Üretim Tahmini:** Önümüzdeki günlerde panelinizin ne kadar elektrik üreteceğini (Watt/Saat cinsinden) tahmin eder.
- 💡 **Akıllı Öneriler:** "Çamaşır makinesini Saat 13:00'te çalıştırın" gibi somut önerilerle, şebekeden çekilen elektriği azaltmanıza ve kendi ürettiğiniz enerjiyi kullanmanıza yardımcı olur.

---

## 📂 Proje Dosya Yapısı

Klasör içerisindeki önemli dosyaların açıklamaları aşağıdadır:

### 1. Ana Kod Dosyaları
- **`solar_prediction.py` (EĞİTİM MODÜLÜ):** 
  - Makine öğrenimi modellerini eğiten ana dosyadır.
  - Ham verileri (`csv`) okur, temizler ve işler.
  - Linear Regression, Random Forest, **XGBoost** (Önerilen), MLP, LightGBM gibi modelleri eğitir ve kıyaslar.
  - En başarılı modeli `best_solar_model.joblib` olarak kaydeder.

- **`solar_wizard.py` (KULLANICI MODÜLÜ - SİHİRBAZ):** 
  - Son kullanıcının çalıştıracağı dosyadır.
  - Eğitilmiş modeli (`solar_model_xgboost.joblib`) ve hava durumu tahmin verisini (`json`) kullanarak geleceğe yönelik tahmin yapar.
  - Kullanıcıya günlük ve saatlik raporlar sunar, cihaz kullanım tavsiyeleri verir.

- **`prepare_data.py`:** 
  - Ham veri dosyalarını birleştirip temiz bir veri seti (`dataset_final.csv`) oluşturmak için kullanılan yardımcı betiktir.

### 2. Veri Dosyaları
- **`open-meteo-35.19N33.50E87m.csv`:** Model eğitimi için kullanılan geçmiş hava durumu verileri.
- **`Energy and power...csv`:** Panelden alınan geçmiş gerçek üretim verileri.
- **`forecast_data.json` / `5-10tarihleri.json`:** Gelecek günlerin (tahmin yapılacak günlerin) saatlik hava durumu verisi. (Open-Meteo API formatında).

### 3. Model Dosyaları
- **`solar_model_xgboost.joblib` / `best_solar_model.joblib`:** `solar_prediction.py` tarafından eğitilmiş ve kaydedilmiş yapay zeka modelleridir.

---

## 🛠️ Kurulum (Installation)

Projeyi çalıştırmak için bilgisayarınızda **Python 3.8+** yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib openpyxl
```

---

##  ▶️ Nasıl Çalıştırılır?

Projenin iki temel aşaması vardır: **Model Eğitimi** ve **Tahmin (Kullanım)**.

### Adım 1: Modeli Eğitmek (Opsiyonel - Eğer model yoksa)
Eğer `solar_model_xgboost.joblib` dosyası yoksa veya yeni verilerle modeli güncellemek istiyorsanız:

1. Terminali açın.
2. `solar_prediction.py` dosyasını çalıştırın:
   ```bash
   python solar_prediction.py
   ```
3. İşlem bittiğinde en iyi model kaydedilecektir.

### Adım 2: Tahmin Yapmak ve Öneri Almak
Sistemi kullanmak ve "Yarın çamaşır makinesini ne zaman çalıştırayım?" sorusuna cevap bulmak için:

1. Terminalde `solar_wizard.py` dosyasını çalıştırın:
   ```bash
   python solar_wizard.py
   ```
2. Program sizden hava durumu dosyasını isteyecektir (Enter'a basarak varsayılan `json` dosyasını seçebilirsiniz).
3. Günlük toplam üretim tahminlerini göreceksiniz.
4. Detaylı saatlik döküm ve **Akıllı Öneriler** için listeden bir tarih girin (Örn: `2025-12-10`).
5. Sistem size en uygun saat aralıklarını (Zirve, Yüksek Verim, Orta Verim) ve hangi cihazları kullanmanız gerektiğini söyleyecektir.

---

## 📊 Kullanılan Teknolojiler ve Algoritmalar

Bu projede **Gözetimli Öğrenme (Supervised Learning)** yöntemleri kullanılmıştır.
- **Algoritmalar:** XGBoost (En yüksek başarı), Random Forest, Linear Regression, MLP (Neural Network).
- **Girdiler (Features):** Sıcaklık, Güneş Radyasyonu (Shortwave, Diffuse, Direct), Bulutluluk Oranı, Saat, Ay.
- **Başarı Metriği:** R² Skoru ve MAE (Ortalama Mutlak Hata).

---

## 📝 Notlar
- Güneş paneli üretim verileri 15 dakikalık aralıklarla kaydedilmiştir.
- Tahminlerde bulutluluk oranı çok yüksekse (%90 üzeri), sistem otomatik kalibrasyon yaparak tahmini düşürür (Bulutlu gün optimizasyonu).
- Proje sunumunda `solar_wizard.py` ekranındaki "Akıllı Planlama" çıktısını göstermek, projenin sosyal etkisini vurgulamak için önemlidir.

**İletişim:**
Batu KIZMAZOĞLU & Semih DEMİR
