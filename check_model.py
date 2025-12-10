import joblib
import pandas as pd
import numpy as np

# 1. Modeli yükle
try:
    print("Model yükleniyor...")
    models = joblib.load('solar_model_xgboost.joblib')
    print(f"Model türü: {type(models)}")
    
    # Eğer 7 ayrı model varsa (örneğin sözlük veya liste içinde)
    if isinstance(models, dict):
        print(f"Model sayısı: {len(models)}")
        first_model_key = list(models.keys())[0]
        print(f"İlk model anahtarı: {first_model_key}")
        
        # İlk modelin ağaç sayısına bakarak eğitilip eğitilmediğini kontrol et
        if hasattr(models[first_model_key], 'n_estimators'):
            print(f"Ağaç sayısı (n_estimators): {models[first_model_key].n_estimators}")
        
        # Feature importance kontrolü (Eğer hepsi 0 ise model öğrenmemiştir)
        if hasattr(models[first_model_key], 'feature_importances_'):
            print(f"Öznitelik Önem Düzeyleri (İlk 5): {models[first_model_key].feature_importances_[:5]}")
            if np.sum(models[first_model_key].feature_importances_) == 0:
                print("!!! UYARI: Modelin öznitelik önem değerleri 0. Model hiçbir şey öğrenmemiş!")
            else:
                print("Model öznitelikleri kullanmış görünüyor.")
                
    else:
        print("Model yapısı beklenen sözlük formatında değil.")

except Exception as e:
    print(f"Hata oluştu: {e}")

# 2. Veri setini kontrol et
try:
    df = pd.read_csv('dataset_final.csv')
    print(f"\nVeri seti boyutu: {df.shape}")
    if df.empty:
        print("!!! KRİTİK HATA: dataset_final.csv BOŞ!")
    else:
        print("Veri seti dolu görünüyor.")
except Exception as e:
    print(f"Veri seti okunamadı: {e}")