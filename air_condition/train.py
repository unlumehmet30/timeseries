import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f'Using device: {device}')

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

input_shape = x_train.shape[1:]

with tf.device(device):      # >>> MODELİ DEVICE ÜZERİNDE OLUŞTUR
    model = models.Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f'Test MAE: {test_mae}')
model.save('air_condition_model.h5')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig('training_validation_loss.png')
plt.show()
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Training and Validation MAE')
plt.legend()
plt.grid()
plt.savefig('training_validation_mae.png')
plt.show()
