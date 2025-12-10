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
import matplotlib.pyplot as plt

def load_and_process_data():
    print("Veriler yükleniyor...")
    
    # 1. Hava Durumu Verisi
    weather_file = "open-meteo-35.19N33.50E87m.csv"
    # Satır 4'te (0-indexli 3) başlıklar var
    df_weather = pd.read_csv(weather_file, skiprows=3)
    
    # Sütun isimlerini temizle
    df_weather.columns = [col.strip() for col in df_weather.columns]
    # Zaman sütunu
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # 2. Üretim Verisi
    energy_file = "Energy and power - PV - Week - Mustafa Kizmazoglu  - 2025-11-29 - 2025-12-05.csv"
    # Satır 12'de (0-indexli 11) başlıklar var, ayırıcı noktalı virgül
    df_energy = pd.read_csv(energy_file, skiprows=11, sep=';')
    
    # Sütun isimlerini temizle (tırnak işaretleri kalabilir, temizleyelim)
    df_energy.columns = [col.replace('"', '').strip() for col in df_energy.columns]
    
    # Zaman sütununu parse et (Örn: "11/29/2025 12.15 AM")
    df_energy['Time period'] = pd.to_datetime(df_energy['Time period'], format='%m/%d/%Y %I.%M %p')
    
    # 'Power [W]' sütununu sayıya çevir (binlik ayracı virgül olabilir, tırnaklar olabilir)
    # Önce string yap, tırnakları ve virgülleri temizle
    df_energy['Power [W]'] = df_energy['Power [W]'].astype(str).str.replace('"', '').str.replace(',', '').str.replace('.', '')
    # Boş stringleri 0 yap
    df_energy['Power [W]'] = pd.to_numeric(df_energy['Power [W]'], errors='coerce').fillna(0)
    
    print("Veri ön işleme ve birleştirme yapılıyor...")
    
    # İki veriyi zaman sütunu üzerinden birleştir (Merge)
    # Hava durumu verisi 'time', enerji verisi 'Time period'
    df_merged = pd.merge(df_energy, df_weather, left_on='Time period', right_on='time', how='inner')
    
    # Gereksiz sütunları at
    df_final = df_merged.drop(columns=['Time period', 'time', 'is_day ()']) # 'is_day' bazen string olabiliyor, model için sayısal lazım veya çıkarabiliriz
    
    # Tarihsel özellikler ekle
    df_merged['hour'] = df_merged['time'].dt.hour
    df_merged['month'] = df_merged['time'].dt.month
    df_merged['dayofyear'] = df_merged['time'].dt.dayofyear
    
    # Model (X) ve Hedef (y) değişkenleri
    target = 'Power [W]'
    
    # Kullanılacak öznitelikler
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
    y = df_merged[target]
    
    return X, y, df_merged

def define_models():
    print("Modeller tanımlanıyor...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "MLP (Neural Network)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0, allow_writing_files=False),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42)
    }
    
    return models

if __name__ == "__main__":
    try:
        # 1. Veri Hazırla
        X, y, df_full = load_and_process_data()
        print(f"Veri seti hazırlandı. Boyut: {X.shape}")
        
        # 2. Eğitim/Test Ayrımı (%80 Eğitim, %20 Test - Zaman Serisi olduğu için karıştırmadan)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        print(f"Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")
        
        # 3. Modelleri Tanımla
        models = define_models()
        
        print("\n--- MODEL ÖZETLERİ ---")
        for name, model in models.items():
            print(f"- {name}: {model}")
            
        # 4. Sadece Linear Regression Modelini Eğit ve Test Et
        print("\n--- LINEAR REGRESSION EĞİTİMİ ---")
        lr_model = models["Linear Regression"]
        lr_model.fit(X_train, y_train)
        print("Model eğitildi.")
        
        # Tahmin Yap
        y_pred = lr_model.predict(X_test)
        
        # Değerlendirme
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\n--- SONUÇLAR ---")
        print(f"MAE (Ortalama Mutlak Hata): {mae:.2f} W")
        print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f} W")
        print(f"R² Skoru: {r2:.4f}")
        
        # Sonuçları Görselleştirme (Opsiyonel - Konsol çıktısı için kapatılabilir ama dosyaya kaydedilebilir)
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_test.values, label='Gerçek')
        # plt.plot(y_pred, label='Tahmin', linestyle='--')
        # plt.legend()
        # plt.title("Linear Regression: Gerçek vs Tahmin")
        # plt.savefig("lr_results.png")
        # print("Grafik 'lr_results.png' olarak kaydedildi.")

        # 5. Random Forest Modelini Eğit ve Test Et
        print("\n--- RANDOM FOREST EĞİTİMİ ---")
        rf_model = models["Random Forest"]
        rf_model.fit(X_train, y_train)
        print("Model eğitildi.")
        
        y_pred_rf = rf_model.predict(X_test)
        
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)
        
        print("\n--- RANDOM FOREST SONUÇLARI ---")
        print(f"MAE: {mae_rf:.2f} W")
        print(f"RMSE: {rmse_rf:.2f} W")
        print(f"R² Skoru: {r2_rf:.4f}")

        # 6. XGBoost Modelini Eğit ve Test Et
        print("\n--- XGBOOST EĞİTİMİ ---")
        xgb_model = models["XGBoost"]
        xgb_model.fit(X_train, y_train)
        print("Model eğitildi.")
        
        y_pred_xgb = xgb_model.predict(X_test)
        
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        r2_xgb = r2_score(y_test, y_pred_xgb)
        
        print("\n--- XGBOOST SONUÇLARI ---")
        print(f"MAE: {mae_xgb:.2f} W")
        print(f"RMSE: {rmse_xgb:.2f} W")
        print(f"R² Skoru: {r2_xgb:.4f}")

        # En iyi model XGBoost olduğu için (User outcome), bu modeli kaydedelim.
        import joblib
        joblib.dump(xgb_model, 'solar_model_xgboost.joblib')
        print("XGBoost modeli 'solar_model_xgboost.joblib' olarak kaydedildi.")

        # 7. MLP Modelini Eğit ve Test Et
        print("\n--- MLP (Neural Network) EĞİTİMİ ---")
        mlp_model = models["MLP (Neural Network)"]
        mlp_model.fit(X_train, y_train)
        print("Model eğitildi.")
        
        y_pred_mlp = mlp_model.predict(X_test)
        
        mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
        rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
        r2_mlp = r2_score(y_test, y_pred_mlp)
        
        print("\n--- MLP SONUÇLARI ---")
        print(f"MAE: {mae_mlp:.2f} W")
        print(f"RMSE: {rmse_mlp:.2f} W")
        print(f"R² Skoru: {r2_mlp:.4f}")

        # 8. Yeni Gelişmiş Modelleri Eğit ve Test Et
        new_models = ["LightGBM", "CatBoost", "Extra Trees"]
        
        for name in new_models:
            print(f"\n--- {name.upper()} EĞİTİMİ ---")
            model = models[name]
            model.fit(X_train, y_train)
            print("Model eğitildi.")
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n--- {name.upper()} SONUÇLARI ---")
            print(f"MAE: {mae:.2f} W")
            print(f"RMSE: {rmse:.2f} W")
            print(f"R² Skoru: {r2:.4f}")
        
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
