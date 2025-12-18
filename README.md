# ☀️ Güneş Enerjisi Üretim Tahmini ve Akıllı Enerji Yönetim Sistemi

Bu proje, makine öğrenimi tekniklerini kullanarak ev tipi güneş panellerinin enerji üretimini tahmin eden ve bu tahminlere dayanarak kullanıcılara **enerji tasarrufu ve verimlilik önerileri** sunan kapsamlı bir yapay zeka uygulamasıdır. 

Sosyal Sorumluluk Dersi kapsamında **Batu KIZMAZOĞLU** ve **Semih DEMİR** tarafından geliştirilmiştir.

---

## 🎯 Projenin Amacı ve Sosyal Etkisi

Projenin temel misyonu, yenilenebilir enerji kaynaklarının bireysel kullanım verimliliğini artırarak **karbon ayak izini azaltmak** ve **enerji tasarrufunu teşvik etmektir**.

Sistem, geçmiş üretim verileri ile hava durumu parametrelerini (sıcaklık, radyasyon, bulutluluk) analiz eder ve şu katma değerleri sağlar:
- 📈 **Hassas Üretim Tahmini:** Panellerin 15 dakikalık aralıklarla ne kadar güç (Watt) üreteceğini yüksek doğrulukla öngörür.
- 💡 **Akıllı Planlama:** "Zirve" üretim saatlerini belirleyerek; çamaşır, bulaşık ve elektrikli araç şarjı gibi yüksek enerji tüketen işlerin şebekeye yük binmeden "bedava ve yeşil" enerjiyle yapılmasını sağlar.
- 🌍 **Davranışsal Dönüşüm:** Tüketicileri, enerjiyi sadece tüketen değil, aynı zamanda verimli yöneten "aktif üretici-tüketici" (prosumer) olmaya yönlendirir.

---

## 📂 Proje Mimarisi

| Dosya / Dizin | Açıklama |
| :--- | :--- |
| **`solar_prediction.py`** | **Model Eğitim Motoru:** Veri temizleme, özellik mühendisliği ve çoklu algoritma (XGBoost, Random Forest, etc.) eğitimi yapar. |
| **`solar_wizard.py`** | **Akıllı Asistan (Sihirbaz):** Son kullanıcı için hazırlanan, tahminleri ve önerileri sunan ana arayüz dosyasıdır. |
| **`prepare_data.py`** | Ham verileri birleştirip eğitim için hazır hale getiren ön işleme betiği. |
| **`solar_model_xgboost.joblib`** | Projenin "beyni" olan, eğitilmiş en iyi model dosyası. |
| **`forecast_data.json`** | Tahmin aşamasında kullanılan gelecek günlerin hava durumu verileri. |

---

## 🛠️ Kurulum ve Gereksinimler

Projenin çalışması için **Python 3.8+** gereklidir. Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib matplotlib
```

---

## ▶️ Kullanım Kılavuzu

### 1. Aşama: Modeli Eğitmek (Geliştiriciler İçin)
Eğer mevcut modelleri güncellemek veya yeni veri setleriyle eğitmek isterseniz:
```bash
python solar_prediction.py
```
*Bu işlem; Linear Regression, Random Forest, XGBoost, MLP ve LightGBM modellerini eğitir, R² ve MAE skorlarını kıyaslar ve en iyi modeli kaydeder.*

### 2. Aşama: Akıllı Planlama Sihirbazını Çalıştırmak (Son Kullanıcı)
Gelecek günlerin üretim tahminini görmek ve kullanım önerisi almak için:
```bash
python solar_wizard.py
```
**Sihirbazın Adımları:**
1. Default hava durumu dosyasını (`5-10tarihleri.json`) onaylayın veya kendi dosyanızı seçin.
2. Karşınıza gelecek **Günlük Üretim Özeti** listesinden bir tarih seçin (Örn: `2025-12-08`).
3. Sistem size o güne özel **Saatlik Üretim Grafiği** (Metin tabanlı) ve **Akıllı Planlama** listesi sunacaktır.

---

## 🔬 Teknik Detaylar ve İnovasyonlar

- **Veri Hassasiyeti:** Model, anlık üretim dalgalanmalarını yakalamak için **15 dakikalık** veri sıklığıyla çalışmaktadır.
- **Akıllı Kalibrasyon (Yeni):** Sistem, bulutluluk oranının %90'ın üzerinde olduğu ve güneş radyasyonunun çok düşük olduğu "ağır kapalı" günlerde otomatik olarak bir ceza katsayısı uygular. Bu sayede modelin bulutlu günlerdeki aşırı iyimser tahminleri gerçekçi seviyelere çekilir.
- **Özellik Mühendisliği (Features):** Sadece sıcaklık değil; *kısa dalga radyasyon, difüz radyasyon, doğrudan normal radyasyon, bulutluluk, günün saati ve yılın ayı* gibi değişkenler kullanılarak tahmin doğruluğu maksimize edilmiştir.
- **Algoritma Karşılaştırması:** Testlerimizde en yüksek başarıyı **XGBoost** algoritması vermiştir.

---

## 📝 Hazırlayanlar
Bu proje bir **Sosyal Sorumluluk** projesidir.

**Geliştirici Ekibi:**
- **Batu KIZMAZOĞLU**
- **Semih DEMİR**

*Modern enerji çözümleriyle daha yeşil bir gelecek için...* 🌿
