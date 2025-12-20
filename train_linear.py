import joblib
from sklearn.linear_model import LinearRegression
from utils import load_data
print("Eğitiliyor: Linear Regression...")
X_train, X_test, y_train, y_test = load_data()
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model_linear.joblib')
print("✅ Kaydedildi: model_linear.joblib")