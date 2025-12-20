import joblib
from sklearn.ensemble import ExtraTreesRegressor
from utils import load_data
print("Eğitiliyor: Extra Trees...")
X_train, X_test, y_train, y_test = load_data()
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model_extratrees.joblib')
print("✅ Kaydedildi: model_extratrees.joblib")