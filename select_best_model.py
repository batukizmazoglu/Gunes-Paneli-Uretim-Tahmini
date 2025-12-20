import joblib
import pandas as pd
import numpy as np
import os
import shutil  # Dosya kopyalamak için
from sklearn.metrics import r2_score, mean_absolute_error
from utils import load_data

def main():
    print("==========================================")
    print("   EN İYİ MODELİ OTOMATİK SEÇME ARACI")
    print("==========================================")
    
    # 1. Test Verisini Yükle
    print("Veriler yükleniyor...")
    _, X_test, _, y_test = load_data()
    
    # 2. Klasördeki Modelleri Bul
    # "final_best_model.joblib" hariç diğer joblibleri al (kendisiyle kıyaslamasın)
    model_files = [f for f in os.listdir('.') 
                   if f.endswith('.joblib') 
                   and f != 'final_best_model.joblib'
                   and f != 'solar_models_all.joblib']
    
    if not model_files:
        print("HATA: Hiçbir .joblib model dosyası bulunamadı!")
        print("Lütfen önce 'train_xgboost.py' vb. dosyaları çalıştırın.")
        return

    results = []
    
    print(f"\n{'MODEL DOSYASI':<35} | {'R2 SKOR':<10} | {'MAE':<10}")
    print("-" * 60)
    
    # 3. Tüm Modelleri Test Et
    for m_file in model_files:
        try:
            model = joblib.load(m_file)
            
            # Modelin tahmin fonksiyonu var mı kontrol et
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                
                # Puanla
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"{m_file:<35} | {r2:<10.4f} | {mae:<10.2f}")
                
                results.append({
                    'file': m_file,
                    'score': r2,
                    'mae': mae
                })
        except Exception as e:
            print(f"{m_file} yüklenemedi: {e}")

    print("-" * 60)

    # 4. Şampiyonu Belirle
    if not results:
        print("Hiçbir model başarıyla test edilemedi.")
        return

    # R2 skoruna göre sırala (En büyük en iyi)
    best_result = sorted(results, key=lambda x: x['score'], reverse=True)[0]
    
    best_file = best_result['file']
    best_score = best_result['score']
    
    print(f"\n🏆 KAZANAN MODEL: {best_file}")
    print(f"⭐ BAŞARI SKORU (R2): {best_score:.4f}")
    
    # 5. Kazananı 'final_best_model.joblib' Olarak Kopyala
    print(f"\n'{best_file}' dosyası 'final_best_model.joblib' olarak kopyalanıyor...")
    shutil.copy(best_file, 'final_best_model.joblib')
    print("✅ İŞLEM TAMAMLANDI! Sihirbaz artık bu modeli kullanacak.")

if __name__ == "__main__":
    main()