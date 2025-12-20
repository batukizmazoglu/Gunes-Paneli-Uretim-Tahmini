import joblib
import catboost as cb
from utils import load_data
print("Eğitiliyor: CatBoost...")
X_train, X_test, y_train, y_test = load_data()
model = cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0, allow_writing_files=False)
model.fit(X_train, y_train)
joblib.dump(model, 'model_catboost.joblib')
print("✅ Kaydedildi: model_catboost.joblib")