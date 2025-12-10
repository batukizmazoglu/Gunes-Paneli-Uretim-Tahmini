import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime

def load_model(model_path='solar_model_xgboost.joblib'):
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası ({model_path}) bulunamadı. Lütfen önce modeli eğitin veya doğru dizinde olduğunuzdan emin olun.")
        sys.exit(1)
    return joblib.load(model_path)

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
    
    if len(predictions) < window_size:
         return ["Veri aralığı öneri üretmek için çok kısa."]

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

    return suggestions

def process_forecast(json_path, model):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
    except FileNotFoundError:
        print("Hata: Dosya bulunamadı.")
        return None
    except json.JSONDecodeError:
        print("Hata: Geçersiz JSON formatı.")
        return None

    # DataFrame Oluştur (15 dakikalık veriler)
    if 'minutely_15' not in data_json:
        print("Hata: JSON dosyasında 'minutely_15' verisi bulunamadı.")
        return None
        
    minutely_data = data_json['minutely_15']
    
    # Tüm dizilerin uzunluklarını kontrol et ve en kısa olana göre eşitle
    lengths = {k: len(v) for k, v in minutely_data.items() if isinstance(v, list)}
    if not lengths:
        print("Hata: Veri bulunamadı.")
        return None
        
    min_len = min(lengths.values())
    max_len = max(lengths.values())
    
    if min_len != max_len:
        print(f"Uyarı: Veri dizileri eşit uzunlukta değil (Min: {min_len}, Max: {max_len}).")
        print("En kısa uzunluğa göre kırpılıyor...")
        for k in minutely_data:
            if isinstance(minutely_data[k], list):
                 minutely_data[k] = minutely_data[k][:min_len]

    df = pd.DataFrame(minutely_data)

    # Zamanı datetime'a çevir
    df['time'] = pd.to_datetime(df['time'])

    # Özellik Çıkarımı (Feature Engineering)
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['dayofyear'] = df['time'].dt.dayofyear
    
    # Sütun isimlerini modelin özelliklerine eşle
    column_mapping = {
        'temperature_2m': 'temperature_2m (°C)',
        'shortwave_radiation': 'shortwave_radiation (W/m²)',
        'diffuse_radiation': 'diffuse_radiation (W/m²)',
        'direct_normal_irradiance': 'direct_normal_irradiance (W/m²)',
        'cloud_cover': 'cloud_cover (%)'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Modelin beklediği özellik sütunları
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
    
    # Eksik sütun kontrolü
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Hata: Şu sütunlar eksik: {missing_cols}")
        return None

    X = df[features]
    
    # Tahmin Yap
    predictions_power_w = model.predict(X)
    
    # Negatif tahminleri 0'a eşitle
    predictions_power_w = np.maximum(predictions_power_w, 0)

    # --- KALİBRASYON ADIMI ---
    prediction_series = pd.Series(predictions_power_w, index=X.index)
    cloud_cover = X['cloud_cover (%)']
    direct_rad = X['direct_normal_irradiance (W/m²)'] # Doğrudan ışık
    
    # Kural: Bulut > %90 VE Doğrudan Işık < 50 W/m² ise tahmini 0.32 ile çarp
    heavy_cloud_mask = (cloud_cover > 90) & (direct_rad < 50)
    
    # Mevcut tahminleri katsayı ile güncelle
    prediction_series.loc[heavy_cloud_mask] *= 0.32
    
    # Güncellenmiş değerleri geri al
    predictions_power_w = prediction_series.values
    # -------------------------
    
    # Enerji Hesabı (Watt -> Watt-Saat)
    # Veriler 15 dakikalık olduğu için
    predictions_energy_wh = predictions_power_w * 0.25
    
    df['Predicted_Power_W'] = predictions_power_w
    df['Predicted_Energy_Wh'] = predictions_energy_wh
    df['Date'] = df['time'].dt.date
    
    return df

def main():
    print("=============================================")
    print("   GÜNEŞ ENERJİSİ ÜRETİM TAHMİN SİSTEMİ")
    print("=============================================")
    
    model = load_model()
    print("Model başarıyla yüklendi.")
    
    while True:
        json_path = input("\nLütfen hava durumu JSON dosyasının yolunu girin (Varsayılan: 5-10tarihleri.json): ").strip()
        if not json_path:
            json_path = '5-10tarihleri.json'
        
        if os.path.exists(json_path):
            break
        else:
            print(f"Hata: '{json_path}' bulunamadı. Lütfen tekrar deneyin.")

    print(f"\n'{json_path}' işleniyor...")
    df_result = process_forecast(json_path, model)
    
    if df_result is None:
        print("İşlem başarısız oldu. Program sonlandırılıyor.")
        return

    # Günlük Toplamları Hesapla
    daily_production = df_result.groupby('Date')['Predicted_Energy_Wh'].sum()
    
    print("\n--- GÜNLÜK ÜRETİM TAHMİNLERİ ---")
    print(f"{'Tarih':<15} | {'Toplam Üretim (Wh)':<20} | {'Toplam Üretim (kWh)':<20}")
    print("-" * 60)
    
    total_period_production = 0
    available_dates = []
    
    for date, energy_wh in daily_production.items():
        energy_kwh = energy_wh / 1000
        total_period_production += energy_wh
        available_dates.append(str(date))
        print(f"{str(date):<15} | {energy_wh:>18.2f} Wh | {energy_kwh:>18.2f} kWh")
        
    print("-" * 60)
    print(f"TOPLAM ({len(daily_production)} Gün) : {total_period_production:>18.2f} Wh | {(total_period_production/1000):>18.2f} kWh")
    
    while True:
        print("\nDetaylı görmek istediğiniz bir gün var mı?")
        print(f"Mevcut Tarihler: {', '.join(available_dates)}")
        choice = input("Tarih girin (YYYY-MM-DD formatında) veya çıkmak için 'q'/'exit' yazın: ").strip()
        
        if choice.lower() in ['q', 'exit', 'hayır', 'yok']:
            print("Program sonlandırılıyor. İyi günler!")
            break
            
        if choice not in available_dates:
            print("Hatalı tarih girişi! Lütfen listedeki tarihlerden birini girin.")
            continue
            
        # Seçilen günün verilerini filtrele
        selected_date = datetime.strptime(choice, "%Y-%m-%d").date()
        day_df = df_result[df_result['Date'] == selected_date].copy()
        
        # 15 dakikalık veriyi saatlik veriye dönüştür (Resample)
        # Ancak burada basitçe 'hour' sütununa göre ortalama alarak da yapabiliriz
        # Power anlık güçtür, energy kümülatif.
        
        # Saatlik ortalama güç ve toplam enerji
        hourly_stats = day_df.groupby('hour').agg({
            'Predicted_Power_W': 'mean',
            'Predicted_Energy_Wh': 'sum'
        }).reset_index()
        
        print(f"\n--- {choice} DETAYLI SAATLİK TAHMİN ---")
        print(f"{'Saat':<10} | {'Ortalama Güç (W)':<20}")
        print("-" * 35)
        
        # Saatlik tabloyu yazdır
        predictions_for_suggestions = [] # Sadece güç değerlerini tutalım (W)
        hours_for_suggestions = []
        
        # Tam 24 saati doldurmak için (eksik saat varsa 0 basmak gerekebilir ama
        # group by sadece olan saatleri verir. Öneri motoru sıralı 24 saat bekliyor olabilir.)
        # Smart suggestion mantığına bakalım: 'prediction' dizisi bekliyor.
        # Bu dizinin indislerinin saat 0..23'e denk geldiğini varsayıyor mu?
        # get_suggestions kodunda: `hours = df_forecast['hour'].values` kullanıyor.
        # Yani hangi saatlerin verisi varsa onu kullanıyor.
        
        # Bizim day_df 15 dakikalık. Suggestion fonksiyonu bir dizi prediction ve bir df bekliyor.
        # En iyisi suggestion fonksiyonuna saatlik veri göndermek.
        
        # 15 dakikalık veriyi saatlik tekil satırlara indirmemiz lazım suggestion için.
        # 'smart_suggestion.py' örneğine göre 'predictions' doğrudan model çıktısıydı (saatlik).
        # Bizim modelimiz 15 dakikalık çalışıyor.
        # Suggestion fonksiyonunu 15 dakikalık veriye uyarlamak ya da veriyi saatliğe resample etmek lazım.
        # Basitlik için saatlik ortalamayı alıp suggestion fonksiyonuna verelim.
        
        # Tam 24 saatlik bir şablon oluşturalım
        full_day = pd.DataFrame({'hour': range(24)})
        hourly_merged = pd.merge(full_day, hourly_stats, on='hour', how='left').fillna(0)
        
        hourly_predictions = hourly_merged['Predicted_Power_W'].values
        
        for index, row in hourly_merged.iterrows():
            print(f"{int(row['hour']):02d}:00      | {row['Predicted_Power_W']:>15.0f} W")
            
        print("\n--- GÜNLÜK AKILLI PLANLAMA ---")
        
        # Suggestion fonksiyonu için 'df_forecast' benzeri bir yapı lazım (sadece 'hour' sütunu kritik)
        df_for_suggestion = pd.DataFrame({'hour': range(24)})
        
        advice_list = get_suggestions(hourly_predictions, df_for_suggestion)
        for line in advice_list:
            print(line)

if __name__ == "__main__":
    main()
