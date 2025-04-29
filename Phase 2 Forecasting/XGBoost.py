# XGBoost Forecasting
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb

# Load scaled dataset
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption']

# Prepare simple lag features
df = pd.DataFrame(series)
for i in range(1,6):
    df[f'lag_{i}'] = df['Water_Consumption'].shift(i)

df.dropna(inplace=True)

X = df.drop(columns=['Water_Consumption'])
y = df['Water_Consumption']

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100)
xgb_model.fit(X_train, y_train)

# Predict
xgb_preds = xgb_model.predict(X_test)

# Plot
plt.figure(figsize=(14,6))
plt.plot(range(len(series)-len(y_test), len(series)), y_test, label='True')
plt.plot(range(len(series)-len(y_test), len(series)), xgb_preds, label='XGBoost Forecast', color='cyan')
plt.legend()
plt.title('XGBoost Forecasting')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase3_xgboost_forecast.jpeg', format='jpeg')
plt.show()
