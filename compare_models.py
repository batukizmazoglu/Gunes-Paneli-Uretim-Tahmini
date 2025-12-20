import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_data

def main():
    print("--- MODELLERİN KARŞILAŞTIRILMASI ---")
    
    # Test verisini al
    _, X_test, _, y_test = load_data()
    
    # Klasördeki .joblib modellerini bul
    model_files = [f for f in os.listdir('.') if f.endswith('.joblib') and 'solar_models_all' not in f]
    
    results = []
    
    print(f"\n{'MODEL DOSYASI':<30} | {'R2 SKOR':<10} | {'MAE':<10}")
    print("-" * 55)
    
    for m_file in model_files:
        try:
            model = joblib.load(m_file)
            # Eğer model bir sözlük değilse (direkt model objesiyse)
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"{m_file:<30} | {r2:<10.4f} | {mae:<10.2f}")
                results.append({'Model': m_file, 'R2': r2})
        except:
            pass
            
    print("-" * 55)
    
if __name__ == "__main__":
    main()