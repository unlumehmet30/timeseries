import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Veriler
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# LSTM reshape
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Model + scaler
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

# Tahmin
y_pred = model.predict(X_test)

# Ters ölçekleme
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Metrikler
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Grafik
plt.figure(figsize=(10,6))
plt.plot(y_test_rescaled, label="Actual", color="blue", linewidth=2)
plt.plot(y_pred_rescaled, label="Predicted", color="red", linewidth=2)
plt.legend()
plt.show()
plt.savefig("lstm_predictions.png")
