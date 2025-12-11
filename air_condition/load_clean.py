import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("weatherHistory.csv")
print(df.head())
df["Formatted Date"]=pd.to_datetime(df["Formatted Date"])
df.set_index("Formatted Date",inplace=True)
print(df.head())
df.drop(columns=["Summary","Precip Type","Daily Summary"],axis=1,inplace=True)
print(df.head())
df.columns=df.columns.str.lower().str.replace(" ","_").str.replace("(","").str.replace(")","")
print(df.head())
print(df.columns)
print(df.isnull().sum())
print(df.info())

plt.figure(figsize=(10,6))
sns.histplot(df['apparent_temperature_c'], bins=50, kde=True)
plt.title('Distribution of Apparent Temperature')
plt.xlabel('Apparent Temperature (°C)')
plt.ylabel('Frequency')
plt.savefig('apparent_temperature_distribution.png')
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(x=df.index[-1000:], y='apparent_temperature_c', data=df[-1000:])
plt.title('Apparent Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Apparent Temperature (°C)')
plt.savefig('apparent_temperature_over_time.png')
plt.show()
df.to_csv("cleaned_weather_history.csv")