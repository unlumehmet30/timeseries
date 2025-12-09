import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Veri yükleme
df_hourly = pd.read_csv("cleaned_power_consumption.csv",
                        index_col=0,
                        parse_dates=True)

df_hourly.dropna(inplace=True)

# Scaling
values = df_hourly.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_values = scaler.fit_transform(values)

joblib.dump(scaler, "scaler.save")

# Sliding window hazırlama
def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sliding_window(scaled_values, window_size)

# Train-test bölünmesi (shuffle=False → time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Kaydetme
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data preparation completed and saved.")
