import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Dosya isimlerini senin projene göre ayarladım
    weather_file = "open-meteo-35.19N33.50E87m.csv"
    energy_file = "Energy and power - PV - Week - Mustafa Kizmazoglu  - 2025-11-29 - 2025-12-05.csv"
    
    # Hava Durumu Okuma
    try:
        df_weather = pd.read_csv(weather_file, skiprows=3)
    except FileNotFoundError:
        print(f"HATA: {weather_file} bulunamadı!")
        exit()
        
    df_weather.columns = [col.strip() for col in df_weather.columns]
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # Enerji Verisi Okuma
    try:
        df_energy = pd.read_csv(energy_file, skiprows=11, sep=';')
    except FileNotFoundError:
        print(f"HATA: {energy_file} bulunamadı!")
        exit()
        
    df_energy.columns = [col.replace('"', '').strip() for col in df_energy.columns]
    df_energy['Time period'] = pd.to_datetime(df_energy['Time period'], format='%m/%d/%Y %I.%M %p')
    
    # Temizlik
    df_energy['Power [W]'] = df_energy['Power [W]'].astype(str).str.replace('"', '').str.replace(',', '').str.replace('.', '')
    df_energy['Power [W]'] = pd.to_numeric(df_energy['Power [W]'], errors='coerce').fillna(0)
    
    # Birleştirme
    df_merged = pd.merge(df_energy, df_weather, left_on='Time period', right_on='time', how='inner')
    
    # Özellik Üretimi
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['dayofyear'] = df_merged['time'].dt.dayofyear
    
    features = [
        'temperature_2m (°C)', 'shortwave_radiation (W/m²)', 'diffuse_radiation (W/m²)', 
        'direct_normal_irradiance (W/m²)', 'cloud_cover (%)', 'hour', 'month', 'dayofyear'
    ]
    
    X = df_merged[features]
    y = df_merged['Power [W]']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)