import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# cihaz seçimi
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f'Using device: {device}')

# verileri yükle
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# modeli yükle
model = tf.keras.models.load_model('air_condition_model.h5',compile=False)

with tf.device(device):
    y_pred = model.predict(x_test)
    # model çıktısı (n,1) ise flatten etmek mantıklı, (n,) ise zaten OK
    y_pred = y_pred.flatten()

# metrikler
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Test MAE: {mae}')
print(f'Test RMSE: {rmse}')

plt.figure(figsize=(10, 6))
plt.plot(y_test[:200], label='Gerçek Değerler', color='blue',marker='o', markersize=3)
plt.plot(y_pred[:200], label='Tahmin Edilen Değerler', color='red', alpha=0.7,marker="*")
plt.xlabel('Zaman Adımı')
plt.ylabel('Hava Koşullandırma Değeri') 
plt.title('Gerçek ve Tahmin Edilen Hava Koşullandırma Değerleri')
plt.legend()
plt.grid()
plt.savefig('test_predictions.png')
plt.show()
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors[:200], label='Hata Değerleri', color='green')
plt.xlabel('Zaman Adımı')
plt.ylabel('Hata Değeri')
plt.title('Tahmin Hatalarının Dağılımı')
plt.legend()
plt.grid()
plt.savefig('error_distribution.png')
plt.show()