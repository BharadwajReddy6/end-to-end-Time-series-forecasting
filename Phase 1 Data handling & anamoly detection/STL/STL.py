# Phase1 - Step3: STL Decomposition (Trend, Seasonality, Residual)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load the scaled data
data = pd.read_csv('synthetic_water_consumption_scaled.csv')

# We will use 'Water_Consumption' for decomposition
water_consumption = data['Water_Consumption']

# Perform STL decomposition
stl = STL(water_consumption, period=365)  # yearly seasonality assumed
result = stl.fit()

# Plot the components
plt.figure(figsize=(14,8))

plt.subplot(4, 1, 1)
plt.plot(water_consumption, color='blue')
plt.title('Original Water Consumption')

plt.subplot(4, 1, 2)
plt.plot(result.trend, color='orange')
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, color='green')
plt.title('Seasonality')

plt.subplot(4, 1, 4)
plt.plot(result.resid, color='red')
plt.title('Residual')

plt.tight_layout()
plt.savefig('phase1_stl_decomposition.jpeg', format='jpeg')
plt.show()

print("✅ STL decomposition done. Plot saved.")
