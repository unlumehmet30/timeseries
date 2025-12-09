import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Veri yükleme ve datetime birleştirme
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    parse_dates={"Date_Time": ["Date", "Time"]},
    infer_datetime_format=True,
    low_memory=False,
    na_values="?"
)

# 2) Date_Time index
df.set_index("Date_Time", inplace=True)

# 3) Sayısala çevir
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

# 4) NaN satırları temizle
df = df.dropna(subset=["Global_active_power"])

# 5) Saatlik ortalama
df_hourly = df["Global_active_power"].resample("H").mean()

print(df_hourly.head())

# 6) Plot
plt.figure(figsize=(10,5))
plt.plot(df_hourly, label='Global Active Power', color='blue')
plt.title("Hourly Average Global Active Power")
plt.xlabel("Date Time")
plt.ylabel("kW")
plt.legend()
plt.show()

# 7) Kaydet
df_hourly.to_csv("cleaned_power_consumption.csv")
