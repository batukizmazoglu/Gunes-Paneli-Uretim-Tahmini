import joblib
from sklearn.ensemble import RandomForestRegressor
from utils import load_data
print("Eğitiliyor: Random Forest...")
X_train, X_test, y_train, y_test = load_data()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model_randomforest.joblib')
print("✅ Kaydedildi: model_randomforest.joblib")