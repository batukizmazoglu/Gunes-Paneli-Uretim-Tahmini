import joblib
from sklearn.neural_network import MLPRegressor
from utils import load_data
print("Eğitiliyor: MLP (Neural Network)...")
X_train, X_test, y_train, y_test = load_data()
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model_mlp.joblib')
print("✅ Kaydedildi: model_mlp.joblib")