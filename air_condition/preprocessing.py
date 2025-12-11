import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import joblib
df=pd.read_csv("cleaned_weather_history.csv")
print(df.info())
x_columns=["humidity","wind_speed_km/h","pressure_millibars","visibility_km","apparent_temperature_c"]
y_columns="temperature_c"
df_z=df[x_columns+[y_columns]].copy()
z_scores=zscore(df_z)
abs_z_scores=np.abs(z_scores)
df_clean=df_z[(abs_z_scores<3).all(axis=1)]
df_clean.dropna(inplace=True)
print(df_clean.head())
print(df_clean.info())
print(df.isnull().sum())
x=df_clean[x_columns]
y=df_clean[y_columns]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
joblib.dump(scaler,"scaler.pkl")
sequence_length=24
x_seq=[]
y_seq=[]
for i in range(len(x_scaled)-sequence_length):
    x_seq.append(x_scaled[i:i+sequence_length])
    y_seq.append(y.values[i+sequence_length])
x_seq=np.array(x_seq)
y_seq=np.array(y_seq) 
x_train,x_test,y_train,y_test=train_test_split(x_seq,y_seq,test_size=0.2,random_state=42)
print("Shapes after preprocessing:")
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
np.save("x_train.npy",x_train)
np.save("x_test.npy",x_test)
np.save("y_train.npy",y_train)
np.save("y_test.npy",y_test)
print("Preprocessing complete. Data saved to .npy files.")