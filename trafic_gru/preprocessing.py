import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from sklearn.model_selection import train_test_split
df=pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df['date_time'] = pd.to_datetime(df['date_time'])
df.set_index('date_time',inplace=True)
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

x=df.drop(["traffic_volume","holiday","weather_main","weather_description"],axis=1)
y=df["traffic_volume"]
print(df.head())
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
joblib.dump(scaler_x, 'scaler_x.save')
joblib.dump(scaler_y, 'scaler_y.save')
#windowing
def create_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size])
    return np.array(X_windows), np.array(y_windows)
window_size = 24
X_windows, y_windows = create_windows(x_scaled, y_scaled, window_size)
print("X_windows shape:", X_windows.shape)
print("y_windows shape:", y_windows.shape)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_windows, y_windows, test_size=0.2, random_state=42, shuffle=False
)   
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

