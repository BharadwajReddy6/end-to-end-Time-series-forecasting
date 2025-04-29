# Phase1 - Step5: Quantiles Visualization

import pandas as pd
import matplotlib.pyplot as plt

# Load scaled data
data = pd.read_csv('synthetic_water_consumption_scaled.csv')

# Select Water Consumption
water_consumption = data['Water_Consumption']

# Calculate important quantiles
quantiles = {
    '10th percentile': water_consumption.quantile(0.10),
    '25th percentile': water_consumption.quantile(0.25),
    '50th percentile (Median)': water_consumption.quantile(0.50),
    '75th percentile': water_consumption.quantile(0.75),
    '90th percentile': water_consumption.quantile(0.90)
}

# Plot Water Consumption with quantile lines
plt.figure(figsize=(14,6))
plt.plot(water_consumption, label='Water Consumption', color='blue')

# Plot quantiles
for label, value in quantiles.items():
    plt.axhline(value, linestyle='--', label=label)

plt.title('Water Consumption with Quantile Levels')
plt.xlabel('Time (days)')
plt.ylabel('Scaled Water Consumption')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig('phase1_quantiles_plot.jpeg', format='jpeg')
plt.show()

print("✅ Quantiles plotted and saved.")
