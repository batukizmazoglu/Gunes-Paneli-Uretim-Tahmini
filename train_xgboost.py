import joblib
import xgboost as xgb
from utils import load_data
print("Eğitiliyor: XGBoost...")
X_train, X_test, y_train, y_test = load_data()
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model_xgboost.joblib')
print("✅ Kaydedildi: model_xgboost.joblib")