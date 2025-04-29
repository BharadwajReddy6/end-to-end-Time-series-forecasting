# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:18:40 2025

@author: bhara
"""

# phase0_data_generation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create Date Range
days = 365 * 20  # 20 years of daily data
dates = pd.date_range(start="2005-01-01", periods=days, freq='D')

# Feature 1: Atmospheric Temperature
atm_temp = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, size=days)

# Feature 2: Surface Temperature (Gearbox Temp mimic)
sur_temp = atm_temp + 10 + np.random.normal(0, 1, size=days)

# Feature 3: Rainfall (some seasonality + random)
rainfall = np.maximum(0, 50 * np.sin(2 * np.pi * (dates.dayofyear - 50) / 365) + np.random.normal(0, 10, size=days))

# Feature 4: Humidity (%)
humidity = 50 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 5, size=days)

# Feature 5: Windspeed (km/h)
windspeed = 10 + 5 * np.cos(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, size=days)

# Feature 6: Population (slow growth over time)
population = 100000 + (np.arange(days) * 2) + np.random.normal(0, 500, size=days)

# Feature 7: Hotel Occupancy (%)
hotel_occupancy = 50 + 30 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 10, size=days)

# Feature 8: Water Consumption (target variable for forecasting)
# Depends on population, rainfall, humidity, hotel occupancy
water_consumption = (population * 0.05) + (hotel_occupancy * 500) - (rainfall * 50) + (humidity * 30) + np.random.normal(0, 10000, size=days)

# Assemble into DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Atm_Temp': atm_temp,
    'Sur_Temp': sur_temp,
    'Rainfall': rainfall,
    'Humidity': humidity,
    'Windspeed': windspeed,
    'Population': population,
    'Hotel_Occupancy': hotel_occupancy,
    'Water_Consumption': water_consumption
})

# Introduce Missing Values randomly (simulate real-world noise)
for col in ['Sur_Temp', 'Rainfall', 'Humidity', 'Water_Consumption']:
    missing_idx = np.random.choice(days, size=int(0.01 * days), replace=False)
    data.loc[missing_idx, col] = np.nan

# Save the dataset
data.to_csv('data/synthetic_water_consumption.csv', index=False)

print("✅ Phase 0 Completed: Synthetic multivariate dataset generated and saved!")

# Plot Water Consumption over Time
plt.figure(figsize=(14,5))
plt.plot(data['Date'], data['Water_Consumption'], color='teal')
plt.title('Synthetic Water Consumption Over 20 Years')
plt.xlabel('Date')
plt.ylabel('Water Consumption (Liters/Day)')
plt.grid(True)
plt.tight_layout()
plt.show()
