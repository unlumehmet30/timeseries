import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
print(df.head())
#info about the dataset
print(df.info())
print(df.isnull().sum())
print(df.describe())
#date time conversion
df['date_time']=pd.to_datetime(df['date_time'])
#extracting features from date_time 
df.set_index('date_time',inplace=True)
print(df.head())
#visualizations
plt.figure(figsize=(12,6))
plt.plot(df.index, df['traffic_volume'])
plt.title('Traffic Volume Over Time')   
plt.xlabel('Date Time')
plt.ylabel('Traffic Volume')
plt.show()
df['hour']=df.index.hour
hourly_avg=df.groupby('hour')['traffic_volume'].mean()
plt.figure(figsize=(10,5))  
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette='viridis')
plt.title('Average Traffic Volume by Hour of Day') 
plt.xlabel('Hour of Day')
plt.ylabel('Average Traffic Volume')
plt.show()
