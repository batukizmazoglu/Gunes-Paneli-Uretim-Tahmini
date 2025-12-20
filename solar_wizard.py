import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime
import plotext as plt 

def load_model():
    model_path = 'final_best_model.joblib'
    if not os.path.exists(model_path):
        print("HATA: Model seçilmemiş!")
        print("Lütfen önce 'compare_and_select.py' dosyasını çalıştırın.")
        sys.exit(1)
    
    # Modelin ne olduğunu (adını) öğrenmek için basit bir kontrol
    model = joblib.load(model_path)
    model_type = type(model).__name__
    print(f"✓ Aktif Model: {model_type} (Otomatik Seçildi)")
    return model

def get_suggestions(predictions, hours):
    """Basit ve etkili öneri sistemi"""
    suggestions = []
    
    # Verileri birleştir (Saat ve Tahmin)
    data = list(zip(hours, predictions))
    
    # En yüksek verimli 3 saati bul
    data.sort(key=lambda x: x[1], reverse=True)
    top_hours = data[:3]
    top_hours_sorted = sorted(top_hours, key=lambda x: x[0]) # Saate göre sırala
    
    if top_hours_sorted and top_hours_sorted[0][1] > 500: # Eğer üretim varsa
        start = int(top_hours_sorted[0][0])
        end = int(top_hours_sorted[-1][0]) + 1
        avg_prod = sum(p for h, p in top_hours) / len(top_hours)
        
        suggestions.append(f"🔥 ZİRVE SAATLER: {start:02d}:00 - {end:02d}:00 arası.")
        suggestions.append(f"   Ortalama Güç: {avg_prod:.0f} Watt")
        suggestions.append("   ✅ ÖNERİ: Çamaşır/Bulaşık makinesini bu aralıkta çalıştırın.")
    else:
        suggestions.append("❌ Düşük üretim günü. Tasarruflu olun.")
        
    return suggestions

def draw_chart(hours, values, title):
    try:
        plt.clf()
        plt.theme('pro')
        plt.plot(hours, values, marker="dot")
        plt.title(title)
        plt.show()
    except: pass

def main():
    print("====================================")
    print("   SOLAR WIZARD - AKILLI ASİSTAN")
    print("====================================")
    
    model = load_model()
    
    json_path = input("\nJSON dosya adı (Enter=Varsayılan): ").strip() or '5-10tarihleri.json'
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
            # 'minutely_15' anahtarını kontrol et
            if 'minutely_15' in json_content:
                data = json_content['minutely_15']
            else:
                print("Hata: JSON içinde 'minutely_15' verisi bulunamadı.")
                return
    except Exception as e:
        print(f"Hata: Dosya okunamadı ({e})")
        return

    # --- HATA DÜZELTME KISMI (Burayı Ekledik) ---
    # Tüm listelerin uzunluklarını kontrol et ve en kısa olana eşitle
    lengths = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    if lengths:
        min_len = min(lengths.values())
        for k in data:
            if isinstance(data[k], list):
                data[k] = data[k][:min_len] # Fazlalıkları kırp
    # --------------------------------------------

    # Veri İşleme
    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        print(f"Veri hatası: {e}")
        return

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['dayofyear'] = df['time'].dt.dayofyear
    
    # Sütun isimlerini düzelt
    cols = {
        'temperature_2m': 'temperature_2m (°C)',
        'shortwave_radiation': 'shortwave_radiation (W/m²)',
        'diffuse_radiation': 'diffuse_radiation (W/m²)',
        'direct_normal_irradiance': 'direct_normal_irradiance (W/m²)',
        'cloud_cover': 'cloud_cover (%)'
    }
    df.rename(columns=cols, inplace=True)
    
    # Modelin beklediği sütunlar
    features = ['temperature_2m (°C)', 'shortwave_radiation (W/m²)', 'diffuse_radiation (W/m²)', 
                'direct_normal_irradiance (W/m²)', 'cloud_cover (%)', 'hour', 'month', 'dayofyear']
    
    # Eksik sütun kontrolü
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"Hata: JSON verisinde şu sütunlar eksik: {missing}")
        return

    # Tahmin
    preds = model.predict(df[features])
    preds = np.maximum(preds, 0) # Negatifleri sıfırla
    
    df['Predicted_W'] = preds
    df['Date'] = df['time'].dt.date
    
    # Günlük Özet
    daily = df.groupby('Date')['Predicted_W'].sum() * 0.25 # Wh hesabı
    
    print(f"\n{'Tarih':<12} | {'Toplam (Wh)':<15}")
    print("-" * 30)
    for d, v in daily.items():
        print(f"{str(d):<12} | {v:.2f}")
    
    # Grafik çizimi (Günlük)
    try:
        plt.clf()
        plt.theme('pro')
        plt.bar([str(d) for d in daily.index], daily.values, color='yellow')
        plt.title("Günlük Toplam Üretim")
        plt.show()
    except: pass

    # Detay ve Öneri
    while True:
        sel = input("\nDetay için tarih gir (YYYY-MM-DD) veya 'q': ").strip()
        if sel.lower() in ['q', 'exit']: break
        
        try:
            target_date = datetime.strptime(sel, "%Y-%m-%d").date()
            day_data = df[df['Date'] == target_date]
            
            if day_data.empty:
                print("Bu tarih için veri yok.")
                continue

            # Saatlik Ortalama
            hourly = day_data.groupby('hour')['Predicted_W'].mean()
            
            print(f"\n--- {sel} SAATLİK GRAFİK ---")
            draw_chart(hourly.index.tolist(), hourly.values.tolist(), f"{sel} Üretim")
            
            print("\n💡 GÜNLÜK TAVSİYE:")
            tips = get_suggestions(hourly.values, hourly.index.tolist())
            for t in tips: print(t)
            
        except ValueError:
            print("Geçersiz tarih formatı!")

if __name__ == "__main__":
    main()