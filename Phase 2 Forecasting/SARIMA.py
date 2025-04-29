# SARIMA Forecasting
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load scaled dataset
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption']

# SARIMA Model
sarima_model = SARIMAX(series, order=(2,1,2), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)


# Forecast
sarima_forecast = sarima_fit.forecast(steps=100)

# Plot
plt.figure(figsize=(14,6))
plt.plot(series[-500:], label='True')
plt.plot(range(len(series), len(series)+100), sarima_forecast, label='SARIMA Forecast', color='green')
plt.legend()
plt.title('SARIMA Forecasting')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase3_sarima_forecast.jpeg', format='jpeg')
plt.show()
