import joblib
import os
import shutil
from sklearn.metrics import r2_score, mean_absolute_error
from utils import load_data

def main():
    print("\n--- MODELLER KARŞILAŞTIRILIYOR ---")
    
    # Test verisini getir
    _, X_test, _, y_test = load_data()
    
    # Klasördeki model_ ile başlayan joblib dosyalarını bul
    model_files = [f for f in os.listdir('.') if f.startswith('model_') and f.endswith('.joblib')]
    
    if not model_files:
        print("HATA: Hiçbir model dosyası bulunamadı! Önce train_ dosyalarını çalıştırın.")
        return

    results = []
    
    print(f"\n{'MODEL ADI':<30} | {'R2 SKOR':<10} | {'MAE':<10}")
    print("-" * 55)
    
    for m_file in model_files:
        try:
            model = joblib.load(m_file)
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"{m_file:<30} | {r2:<10.4f} | {mae:<10.2f}")
                
                results.append({'file': m_file, 'score': r2})
        except Exception as e:
            print(f"{m_file} hatası: {e}")

    print("-" * 55)
    
    # En iyiyi seç
    if results:
        best_model = sorted(results, key=lambda x: x['score'], reverse=True)[0]
        print(f"\n🏆 KAZANAN: {best_model['file']} (Skor: {best_model['score']:.4f})")
        
        # En iyiyi 'final_best_model.joblib' olarak kopyala
        shutil.copy(best_model['file'], 'final_best_model.joblib')
        print(f"✅ Bu model 'final_best_model.joblib' olarak ayarlandı.")
        print("Sihirbaz artık bu modeli kullanacak!")

if __name__ == "__main__":
    main()