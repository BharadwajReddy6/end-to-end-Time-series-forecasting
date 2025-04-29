# Phase1 Scaling the features using MinMaxScaler

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the filled data
data = pd.read_csv('synthetic_water_consumption_filled.csv')

# Remove Date column for scaling
features = data.drop(columns=['Date'])

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with scaled data
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# Save the scaled data
scaled_data.to_csv('synthetic_water_consumption_scaled.csv', index=False)

print("✅ Scaling completed using MinMaxScaler. Saved scaled data.")

# OPTIONAL: Plot before vs after scaling for one feature (example: Water_Consumption)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(data['Water_Consumption'][:500], color='orange')
plt.title('Original Water Consumption (first 500 samples)')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(scaled_data['Water_Consumption'][:500], color='green')
plt.title('Scaled Water Consumption (first 500 samples)')
plt.grid(True)

plt.tight_layout()
plt.savefig('phase1_scaling_water_consumption.jpeg', format='jpeg')
plt.show()
