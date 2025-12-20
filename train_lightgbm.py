import joblib
import lightgbm as lgb
from utils import load_data
print("Eğitiliyor: LightGBM...")
X_train, X_test, y_train, y_test = load_data()
model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
model.fit(X_train, y_train)
joblib.dump(model, 'model_lightgbm.joblib')
print("✅ Kaydedildi: model_lightgbm.joblib")