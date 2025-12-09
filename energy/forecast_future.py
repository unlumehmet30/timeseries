import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import os

# Dosya var mı kontrolü (hatayı daha açıklayıcı yapar)
assert os.path.exists("lstm_model.h5"), "lstm_model.h5 bulunamadı!"
assert os.path.exists("scaler.save"), "scaler.save bulunamadı!"
assert os.path.exists("cleaned_power_consumption.csv"), "cleaned_power_consumption.csv bulunamadı!"
assert os.path.exists("X_test.npy"), "X_test.npy bulunamadı! (daha önce kaydettiğiniz isim 'X_test.npy' olmalı)"

# Model ve scaler yükle
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

# Veri
data = pd.read_csv("cleaned_power_consumption.csv", index_col=0, parse_dates=True)

# Son 72 saat (örnekleme)
last_72 = data[-72:].copy()
last_36_real = last_72.iloc[:36].values.flatten()   # geçmiş 36 saat (gerçek)
real_next_36 = last_72.iloc[36:].values.flatten()   # gerçek next 36 saat (karşılaştırma için)

# Test girdisi: X_test'den son pencere
X_test = np.load("X_test.npy")   # dikkat: üstte kaydedilen isim X_test.npy olmalı
# Eğer X_test 2D ise (samples, timesteps) kabul edilir
forecast_input = X_test[-1].copy()   # shape: (window_size,)

# Güvenlik: 1D olduğundan emin ol
forecast_input = forecast_input.reshape(-1)

future_pred = []

# Forecast döngüsü: 36 adım ileri
n_steps_ahead = 36
for _ in range(n_steps_ahead):
    # LSTM bekler: (1, time_steps, features)
    inp = forecast_input.reshape(1, forecast_input.shape[0], 1)
    pred_scaled = model.predict(inp, verbose=0)[0, 0]   # tek skalarlı çıktı (scaled)
    
    # Ters ölçekleme (scaler expects 2D)
    pred_value = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
    future_pred.append(pred_value)
    
    # Yeni girdi oluştur: kaydır ve yeni scaled değeri ekle
    forecast_input = np.concatenate((forecast_input[1:], [pred_scaled]))

# Plot
plt.figure(figsize=(10, 6))
# Eğer elimizde gerçek next 36 yoksa sadece geleceği çizeriz; burada varsa karşılaştır
if len(real_next_36) == len(future_pred):
    plt.plot(real_next_36, label='Real Next 36 Hours', color='blue')
else:
    # Eğer real_next_36 yoksa ya da uzunluk farklıysa uyarı
    print("Uyarı: gerçek next36 verisi ile tahmin uzunluğu eşleşmiyor.")

plt.plot(future_pred, label='Forecasted Next 36 Hours', color='orange')
plt.xlabel('Hours')
plt.ylabel('Power Consumption (kW)')
plt.title('Power Consumption Forecast for Next 36 Hours')
plt.legend()
plt.grid()

# Kaydet ve göster (önce kaydetmek daha güvenli)
plt.savefig("future_forecast.png", dpi=200, bbox_inches='tight')
plt.show()
