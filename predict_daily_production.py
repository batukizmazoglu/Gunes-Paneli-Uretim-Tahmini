import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

def main():
    print("Tahmin işlemi başlatılıyor...")

    # 1. JSON Verisini Yükle
    try:
        with open('forecast_data.json', 'r') as f:
            data_json = json.load(f)
    except FileNotFoundError:
        print("Hata: forecast_data.json dosyası bulunamadı.")
        return

    # 2. DataFrame Oluştur (15 dakikalık veriler)
    minutely_data = data_json['minutely_15']
    df = pd.DataFrame(minutely_data)

    # Zamanı datetime'a çevir
    df['time'] = pd.to_datetime(df['time'])

    # 3. Tarih Aralığını Filtrele (10 Aralık - 19 Aralık)
    # Başlangıç: 2025-12-10 00:00:00
    # Bitiş: 2025-12-19 23:59:59 (yani 2025-12-20'den küçük)
    start_date = "2025-12-10"
    end_date = "2025-12-20" # Bu tarih dahil değil
    
    mask = (df['time'] >= start_date) & (df['time'] < end_date)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        print("Belirtilen tarih aralığında veri bulunamadı.")
        return

    # 4. Özellik Çıkarımı (Feature Engineering)
    df_filtered['hour'] = df_filtered['time'].dt.hour
    df_filtered['month'] = df_filtered['time'].dt.month
    df_filtered['dayofyear'] = df_filtered['time'].dt.dayofyear
    
    # Sütun isimlerini modelin özelliklerine eşle
    # JSON'daki isimler -> Modelin beklediği isimler
    column_mapping = {
        'temperature_2m': 'temperature_2m (°C)',
        'shortwave_radiation': 'shortwave_radiation (W/m²)',
        'diffuse_radiation': 'diffuse_radiation (W/m²)',
        'direct_normal_irradiance': 'direct_normal_irradiance (W/m²)',
        'cloud_cover': 'cloud_cover (%)'
    }
    df_filtered.rename(columns=column_mapping, inplace=True)

    # Modelin beklediği özellik sütunları (Sırası önemli olabilir, ama XGBoost genelde isme bakar)
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
    missing_cols = [col for col in features if col not in df_filtered.columns]
    if missing_cols:
        print(f"Hata: Şu sütunlar eksik: {missing_cols}")
        return

    X = df_filtered[features]

    # 5. Modeli Yükle ve Tahmin Yap
    model_path = 'solar_model_xgboost.joblib'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Hata: Model dosyası ({model_path}) bulunamadı. Lütfen önce modeli eğitin.")
        return

    predictions_power_w = model.predict(X)
    
    # Negatif tahminleri 0'a eşitle
    predictions_power_w = np.maximum(predictions_power_w, 0)

    # --- KALİBRASYON ADIMI ---
    # Kullanıcı Geri Bildirimi: 10 Aralık'ta model 5.62 kWh tahmin etti, gerçekleşen 1.79 kWh.
    # Bu, %100 bulutlu ve düşük ışıkta modelin fazla iyimser olduğunu gösteriyor (Factor ~0.32).
    # Bu durumu düzeltmek için "Ağır Bulutluluk Cezası" ekliyoruz.
    
    # Pandas serisi olarak işlem yapmak daha kolay
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
    
    # 6. Enerji Hesabı (Watt -> Watt-Saat)
    # Veriler 15 dakikalık olduğu için, o 15 dakika boyunca ortalama gücün bu olduğunu varsayıyoruz.
    # Enerji (Wh) = Güç (W) * Süre (h) = W * (15/60) = W * 0.25
    predictions_energy_wh = predictions_power_w * 0.25
    
    df_filtered['Predicted_Power_W'] = predictions_power_w
    df_filtered['Predicted_Energy_Wh'] = predictions_energy_wh
    
    # 7. Sonuçları Günlük Olarak Grupla ve Yazdır
    df_filtered['Date'] = df_filtered['time'].dt.date
    daily_production = df_filtered.groupby('Date')['Predicted_Energy_Wh'].sum()
    
    print("\n--- 10-19 ARALIK GÜNLÜK GÜNEŞ ENERJİSİ ÜRETİM TAHMİNİ ---")
    print(f"{'Tarih':<15} | {'Toplam Üretim (Wh)':<20} | {'Toplam Üretim (kWh)':<20}")
    print("-" * 60)
    
    total_period_production = 0
    
    for date, energy_wh in daily_production.items():
        energy_kwh = energy_wh / 1000
        total_period_production += energy_wh
        print(f"{str(date):<15} | {energy_wh:>18.2f} Wh | {energy_kwh:>18.2f} kWh")
        
    print("-" * 60)
    print(f"TOPLAM (10 Gün) : {total_period_production:>18.2f} Wh | {(total_period_production/1000):>18.2f} kWh")

if __name__ == "__main__":
    main()
