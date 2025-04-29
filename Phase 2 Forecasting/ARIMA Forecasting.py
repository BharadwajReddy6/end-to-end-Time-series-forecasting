# ARIMA Forecasting

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load scaled dataset
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption']

# Train ARIMA model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=100)

# Plot
plt.figure(figsize=(14,6))
plt.plot(series[-500:], label='True')
plt.plot(range(len(series), len(series)+100), forecast, label='ARIMA Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecasting')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase3_arima_forecast.jpeg', format='jpeg')
plt.show()
