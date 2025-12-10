import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def generate_mock_forecast():
    """
    Gerçek bir API olmadığı için, yarın için örnek bir hava durumu verisi oluşturur.
    Güneşli bir gün simülasyonu.
    """
    
    # Yarının tarihini al
    tomorrow = datetime.now() + timedelta(days=1)
    month = tomorrow.month
    dayofyear = tomorrow.timetuple().tm_yday
    
    hours = list(range(24))
    
    # Veri iskeleti
    data = {
        'hour': hours,
        'month': [month] * 24,
        'dayofyear': [dayofyear] * 24,
        'temperature_2m (°C)': [],
        'shortwave_radiation (W/m²)': [],
        'diffuse_radiation (W/m²)': [],
        'direct_normal_irradiance (W/m²)': [],
        'cloud_cover (%)': []
    }
    
    # Basit bir fiziksel simülasyon (Güneş 06:00 - 19:00 arası)
    for h in hours:
        # Sıcaklık: Gece 15C, Gündüz 25C'ye kadar çıksın (Basit sinüs eğrisi)
        temp = 15 + 10 * np.sin((h - 4) * np.pi / 12) if 6 <= h <= 18 else 15
        data['temperature_2m (°C)'].append(max(temp, 10)) # Min 10
        
        # Radyasyon: Çan eğrisi
        if 6 <= h <= 19:
            # Öğle saatlerinde zirve (h=12-13)
            peak = 800 # max W/m2
            rad = peak * np.sin((h - 6) * np.pi / 13)
            rad = max(0, rad)
            
            data['shortwave_radiation (W/m²)'].append(rad)
            data['diffuse_radiation (W/m²)'].append(rad * 0.3) # %30 diffuse varsayımı
            data['direct_normal_irradiance (W/m²)'].append(rad * 0.7)
            data['cloud_cover (%)'].append(10) # Açık hava (%10 bulut)
        else:
            data['shortwave_radiation (W/m²)'].append(0)
            data['diffuse_radiation (W/m²)'].append(0)
            data['direct_normal_irradiance (W/m²)'].append(0)
            data['cloud_cover (%)'].append(5)
            
    return pd.DataFrame(data)

def get_suggestions(predictions, df_forecast):
    """
    Tahminlere göre gelişmiş, zaman aralıklı öneriler üretir.
    En yüksek 3 saatlik dilimi ve diğer verimli saatleri belirler.
    """
    suggestions = []
    
    # Eşik değerler
    HIGH_THRESHOLD = 2000 
    MEDIUM_THRESHOLD = 800 
    
    # 1. En İyi 3 Saatlik Aralığı Bul (Moving Sum)
    best_window_sum = 0
    best_window_start = -1
    window_size = 3
    
    hours = df_forecast['hour'].values
    
    for i in range(len(predictions) - window_size + 1):
        # Gece saatlerini (20:00 - 06:00) pas geçmek mantıklı olabilir ama 
        # üretim zaten 0 olacağı için toplama etki etmez.
        current_sum = np.sum(predictions[i : i+window_size])
        if current_sum > best_window_sum:
            best_window_sum = current_sum
            best_window_start = i
            
    best_window_indices = []
    if best_window_start != -1 and best_window_sum > (window_size * MEDIUM_THRESHOLD):
        # Anlamsız düşük üretimlerde "En iyi" dememek için bir kontrol
        best_end = best_window_start + window_size
        best_window_indices = list(range(best_window_start, best_end))
        
        avg_prod = best_window_sum / window_size
        start_h = int(hours[best_window_start])
        end_h = int(hours[best_end - 1]) + 1 # Bitiş saati (exclusive)
        
        suggestions.append(f"🔥 EN YÜKSEK VERİM (ZİRVE) SAATLERİ: {start_h:02d}:00 - {end_h:02d}:00")
        suggestions.append(f"   Ortalama Üretim: {avg_prod:.0f} W")
        suggestions.append("   ✅ ÖNERİLEN CİHAZLAR: Çamaşır Makinesi, Bulaşık Makinesi, Fırın, Elektrikli Araç Şarjı.")
        suggestions.append("   -> En çok enerji tüketen işlerinizi bu aralığa sıkıştırın!\n")
    
    # 2. Diğer Verimli Saatleri Bul (Peak dışındaki yüksek/orta saatler)
    secondary_high = []
    secondary_medium = []
    
    for i, pred in enumerate(predictions):
        if i in best_window_indices:
            continue # Zaten zirve aralığında
            
        h = int(hours[i])
        if pred >= HIGH_THRESHOLD:
            secondary_high.append(h)
        elif pred >= MEDIUM_THRESHOLD:
            secondary_medium.append(h)
            
    # Gruplama yardımcı fonksiyonu (ardışık saatleri birleştirir: [9, 10, 11] -> "09:00-12:00")
    def group_hours(hour_list):
        if not hour_list:
            return []
        ranges = []
        start = hour_list[0]
        end = start
        for h in hour_list[1:]:
            if h == end + 1:
                end = h
            else:
                ranges.append((start, end + 1))
                start = h
                end = h
        ranges.append((start, end + 1))
        return ranges

    # İkincil Yüksek (Zirve kadar değil ama yüksek)
    if secondary_high:
        ranges = group_hours(secondary_high)
        time_strs = [f"{s:02d}:00-{e:02d}:00" for s, e in ranges]
        suggestions.append(f"⚡ YÜKSEK VERİM SAATLERİ: {', '.join(time_strs)}")
        suggestions.append("   ✅ ÖNERİLEN CİHAZLAR: Ütü, Elektrikli Süpürge, Ketıl.")
        suggestions.append("   -> Zirve saatleri kaçırırsanız en iyi alternatifler bunlardır.\n")
        
    # Orta Verim
    if secondary_medium:
        ranges = group_hours(secondary_medium)
        time_strs = [f"{s:02d}:00-{e:02d}:00" for s, e in ranges]
        suggestions.append(f"🔋 ORTA VERİM SAATLERİ: {', '.join(time_strs)}")
        suggestions.append("   ✅ ÖNERİLEN CİHAZLAR: Laptop/Telefon Şarjı, TV, Aydınlatma.")
        suggestions.append("   -> Bataryalı cihazları şarj etmek için idealdir.\n")
        
    # Eğer hiç üretim yoksa
    if not best_window_indices and not secondary_high and not secondary_medium:
        suggestions.append("❌ DÜŞÜK ÜRETİM GÜNÜ")
        suggestions.append("   Bugün güneş enerjisi üretimi oldukça düşük.")
        suggestions.append("   -> Zorunlu olmayan yüksek tüketimli işleri erteleyin.")

    print("\n--- DETAYLI SAATLİK TAHMİN ---")
    for i, pred in enumerate(predictions):
        print(f"{int(hours[i]):02d}:00 : {pred:.0f} W")

    print("\n--- GÜNLÜK AKILLI PLANLAMA ---")
    return suggestions

def main():
    print("Sistem başlatılıyor...")
    
    # 1. Modeli Yükle
    model_path = 'solar_model_xgboost.joblib'
    try:
        model = joblib.load(model_path)
        print(f"Model yüklendi: {model_path}")
    except FileNotFoundError:
        print("Hata: Model dosyası bulunamadı! Önce 'solar_prediction.py'yi çalıştırın.")
        return

    # 2. Hava Durumu Verisini Hazırla (Mock)
    print("Hava durumu verileri alınıyor (Simülasyon)...")
    df_forecast = generate_mock_forecast()
    
    # Modelin beklediği sütun sırası (solar_prediction.py ile aynı olmalı)
    features = [
        'temperature_2m (°C)', 
        'shortwave_radiation (W/m²)', 
        'diffuse_radiation (W/m²)', 
        'direct_normal_irradiance (W/m²)', 
        'cloud_cover (%)',
        'hour', 
        'month', 
        'dayofyear'
    ]
    
    X_forecast = df_forecast[features]
    
    # 3. Tahmin Yap
    print("Üretim tahmini yapılıyor...")
    predictions = model.predict(X_forecast)
    # Negatif tahminleri 0'a çek (Fiziksel olarak negatif üretim olmaz)
    predictions = [max(0, p) for p in predictions]
    
    # 4. Önerileri Oluştur ve Sun
    advice_list = get_suggestions(np.array(predictions), df_forecast)
    
    for line in advice_list:
        print(line)

if __name__ == "__main__":
    main()
