import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

def load_and_process_data():
    print("Veriler yükleniyor...")
    
    # 1. Hava Durumu Verisi
    weather_file = "open-meteo-35.19N33.50E87m.csv"
    df_weather = pd.read_csv(weather_file, skiprows=3)
    df_weather.columns = [col.strip() for col in df_weather.columns]
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # 2. Üretim Verisi
    energy_file = "Energy and power - PV - Week - Mustafa Kizmazoglu  - 2025-11-29 - 2025-12-05.csv"
    df_energy = pd.read_csv(energy_file, skiprows=11, sep=';')
    df_energy.columns = [col.replace('"', '').strip() for col in df_energy.columns]
    df_energy['Time period'] = pd.to_datetime(df_energy['Time period'], format='%m/%d/%Y %I.%M %p')
    
    # Temizlik
    df_energy['Power [W]'] = df_energy['Power [W]'].astype(str).str.replace('"', '').str.replace(',', '').str.replace('.', '')
    df_energy['Power [W]'] = pd.to_numeric(df_energy['Power [W]'], errors='coerce').fillna(0)
    
    print("Veri birleştiriliyor...")
    df_merged = pd.merge(df_energy, df_weather, left_on='Time period', right_on='time', how='inner')
    
    # Tarihsel özellikler
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['dayofyear'] = df_merged['time'].dt.dayofyear
    
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
    
    X = df_merged[features]
    y = df_merged['Power [W]']
    
    return X, y, df_merged

def define_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "MLP (Neural Network)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0, allow_writing_files=False),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42)
    }

if __name__ == "__main__":
    try:
        # --- VERİ HAZIRLIĞI ---
        X, y, df_full = load_and_process_data()
        print(f"\nToplam Veri Sayısı: {len(X)} satır.")
        
        # Train/Test Split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # --- MODEL EĞİTİM DÖNGÜSÜ ---
        models = define_models()
        results = []
        trained_models = {}

        print(f"\n{'MODEL ADI':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2 SKOR':<10}")
        print("-" * 65)

        for name, model in models.items():
            # Eğit
            model.fit(X_train, y_train)
            
            # Test Et
            y_pred = model.predict(X_test)
            
            # Skorla
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Listeye ekle
            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
            
            # Eğitilmiş modeli sözlüğe kaydet
            trained_models[name] = model
            
            print(f"{name:<25} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.4f}")

        # --- SONUÇLARI KAYDETME ---
        print("-" * 65)
        
        # 1. Tüm modelleri topluca kaydet
        joblib.dump(trained_models, 'solar_models_all.joblib')
        print("\n[OK] Tüm modeller 'solar_models_all.joblib' dosyasına kaydedildi.")

        # 2. En iyi modeli bul ve ayrıca kaydet
        results_df = pd.DataFrame(results)
        best_model_row = results_df.sort_values(by='R2', ascending=False).iloc[0]
        best_model_name = best_model_row['Model']
        best_model = trained_models[best_model_name]
        
        joblib.dump(best_model, 'best_solar_model.joblib')
        
        print(f"[OK] EN İYİ MODEL: '{best_model_name}' (R2: {best_model_row['R2']:.4f})")
        print(f"     Bu model 'best_solar_model.joblib' olarak ayrıca kaydedildi.")

    except Exception as e:
        print(f"KRİTİK HATA: {e}")