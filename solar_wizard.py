import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime
import plotext as plt 

# --- MODEL YÜKLEME ---
def load_model():
    model_path = 'final_best_model.joblib'
    if not os.path.exists(model_path):
        print("HATA: Model seçilmemiş!")
        print("Lütfen önce 'compare_and_select.py' dosyasını çalıştırın.")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        model_type = type(model).__name__
        print(f"✓ Aktif Model: {model_type} (Otomatik Seçildi)")
        return model
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        sys.exit(1)

# --- GRAFİK FONKSİYONLARI ---
def draw_terminal_bar_chart(dates, values):
    try:
        plt.clf(); plt.theme('pro')
        str_dates = [str(d) for d in dates]
        plt.bar(str_dates, values, color='yellow', fill=True)
        plt.title("Gunluk Uretim Tahmini (Wh)")
        plt.show()
    except: pass

def draw_terminal_line_chart(hours, power_values, date_str):
    try:
        plt.clf(); plt.theme('pro')
        plt.plot(hours, power_values, color='green', marker="dot")
        plt.title(f"{date_str} - Saatlik Guc Uretimi (W)")
        if len(power_values) > 0: plt.ylim(0, max(power_values) * 1.1)
        plt.show()
    except: pass

# --- DETAYLI ÖNERİ MOTORU (ESKİ VERSİYON) ---
def get_suggestions(predictions, hours_list):
    """
    Tahminlere göre gelişmiş, zaman aralıklı öneriler üretir.
    En yüksek 3 saatlik dilimi ve diğer verimli saatleri belirler.
    """
    suggestions = []
    
    # Eşik değerler (Watt cinsinden)
    HIGH_THRESHOLD = 2000 
    MEDIUM_THRESHOLD = 800 
    
    # 1. En İyi 3 Saatlik Aralığı Bul (Moving Sum)
    best_window_sum = 0
    best_window_start = -1
    window_size = 3
    
    if len(predictions) < window_size:
         return ["Veri aralığı öneri üretmek için çok kısa."]

    for i in range(len(predictions) - window_size + 1):
        current_sum = np.sum(predictions[i : i+window_size])
        if current_sum > best_window_sum:
            best_window_sum = current_sum
            best_window_start = i
            
    best_window_indices = []
    
    # Zirve saatler bulunduysa ekle
    if best_window_start != -1 and best_window_sum > (window_size * MEDIUM_THRESHOLD):
        best_end = best_window_start + window_size
        best_window_indices = list(range(best_window_start, best_end))
        
        avg_prod = best_window_sum / window_size
        
        # Saat listesinden gerçek saati çek
        start_h = int(hours_list[best_window_start])
        # Bitiş saati (Liste dışına taşarsa 24 yap)
        end_idx = best_window_start + window_size - 1
        if end_idx < len(hours_list) - 1:
            end_h = int(hours_list[end_idx]) + 1
        else:
            end_h = int(hours_list[-1]) + 1

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
            
        h = int(hours_list[i])
        if pred >= HIGH_THRESHOLD:
            secondary_high.append(h)
        elif pred >= MEDIUM_THRESHOLD:
            secondary_medium.append(h)
            
    # Gruplama yardımcı fonksiyonu (ardışık saatleri birleştirir: [9, 10, 11] -> "09:00-12:00")
    def group_hours(hour_list):
        if not hour_list: return []
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

# --- ANA PROGRAM ---
def main():
    print("====================================")
    print("   SOLAR WIZARD - AKILLI ASİSTAN")
    print("====================================")
    
    model = load_model()
    
    json_path = input("\nJSON dosya adı (Enter=Varsayılan): ").strip() or '5-10tarihleri.json'
    
    # 1. JSON OKUMA VE DÜZELTME
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            if 'minutely_15' in json_content:
                data = json_content['minutely_15']
            else:
                print("Hata: JSON içinde 'minutely_15' verisi bulunamadı.")
                return
                
        # --- BOYUT EŞİTLEME (HATA DÜZELTİCİ) ---
        lengths = {k: len(v) for k, v in data.items() if isinstance(v, list)}
        if lengths:
            min_len = min(lengths.values())
            for k in data:
                if isinstance(data[k], list):
                    data[k] = data[k][:min_len]
        # ---------------------------------------

    except Exception as e:
        print(f"Hata: Dosya okunamadı ({e})")
        return

    # 2. DATAFRAME OLUŞTURMA
    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        print(f"Veri hatası: {e}")
        return

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['dayofyear'] = df['time'].dt.dayofyear
    
    # Sütun İsimlerini Eşle
    cols = {
        'temperature_2m': 'temperature_2m (°C)',
        'shortwave_radiation': 'shortwave_radiation (W/m²)',
        'diffuse_radiation': 'diffuse_radiation (W/m²)',
        'direct_normal_irradiance': 'direct_normal_irradiance (W/m²)',
        'cloud_cover': 'cloud_cover (%)'
    }
    df.rename(columns=cols, inplace=True)
    
    features = ['temperature_2m (°C)', 'shortwave_radiation (W/m²)', 'diffuse_radiation (W/m²)', 
                'direct_normal_irradiance (W/m²)', 'cloud_cover (%)', 'hour', 'month', 'dayofyear']
    
    # Eksik Sütun Kontrolü
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"Hata: JSON verisinde eksik sütunlar: {missing}")
        return

    # 3. TAHMİN VE HESAPLAMA
    preds = model.predict(df[features])
    preds = np.maximum(preds, 0)
    
    # Kalibrasyon (Opsiyonel ama önerilir)
    prediction_series = pd.Series(preds, index=df.index)
    heavy_cloud = (df['cloud_cover (%)'] > 90) & (df['direct_normal_irradiance (W/m²)'] < 50)
    prediction_series.loc[heavy_cloud] *= 0.32
    preds = prediction_series.values

    df['Predicted_W'] = preds
    df['Date'] = df['time'].dt.date
    
    # 4. GÜNLÜK ÖZET
    daily = df.groupby('Date')['Predicted_W'].sum() * 0.25 # Wh hesabı
    
    print(f"\n{'Tarih':<12} | {'Toplam (Wh)':<15}")
    print("-" * 30)
    for d, v in daily.items():
        print(f"{str(d):<12} | {v:.2f}")
    
    print("\n[Günlük Grafik]")
    draw_terminal_bar_chart(daily.index, daily.values)

    # 5. DETAYLI ANALİZ DÖNGÜSÜ
    available_dates = [str(d) for d in daily.index]
    
    while True:
        sel = input("\nDetaylı analiz için tarih gir (YYYY-MM-DD) veya 'q': ").strip()
        if sel.lower() in ['q', 'exit']: break
        
        if sel not in available_dates:
            print("Geçersiz tarih! Listeden seçin.")
            continue
            
        target_date = datetime.strptime(sel, "%Y-%m-%d").date()
        day_data = df[df['Date'] == target_date]

        # Saatlik Ortalama (Groupby ile)
        hourly_stats = day_data.groupby('hour')['Predicted_W'].mean()
        
        # 0-23 Arası tüm saatlerin olduğundan emin ol (Öneri motoru için önemli)
        full_hours = pd.DataFrame({'hour': range(24)})
        merged = pd.merge(full_hours, hourly_stats, on='hour', how='left').fillna(0)
        
        print(f"\n--- {sel} SAATLİK GRAFİK ---")
        draw_terminal_line_chart(merged['hour'].tolist(), merged['Predicted_W'].tolist(), f"{sel} Üretim")
        
        print("\n💡 AKILLI EV ÖNERİLERİ:")
        tips = get_suggestions(merged['Predicted_W'].values, merged['hour'].values)
        for t in tips: print(t)

if __name__ == "__main__":
    main()