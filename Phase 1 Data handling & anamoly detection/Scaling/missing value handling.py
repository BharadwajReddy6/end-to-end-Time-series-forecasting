#Missing Value Handling (Next Observation Carried Backward + Plotting)

import pandas as pd
import matplotlib.pyplot as plt

# Load the generated data
data = pd.read_csv('synthetic_water_consumption.csv')

# Check missing value summary (optional)
print("Missing values before handling:\n", data.isnull().sum())

# Handling missing values by Next Observation Carried Backward (NOCB)
data_filled = data.bfill()

# Check again
print("\nMissing values after NOCB handling:\n", data_filled.isnull().sum())

# Plot missing values before and after
missing_before = data.isnull().sum()
missing_after = data_filled.isnull().sum()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
missing_before.plot(kind='bar', color='red')
plt.title('Missing Values Before Handling')
plt.ylabel('Count')

plt.subplot(1,2,2)
missing_after.plot(kind='bar', color='green')
plt.title('Missing Values After NOCB Handling')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('phase1_missing_values_handling.jpeg', format='jpeg')  # Save plot
plt.show()

# Save the filled data for next steps
data_filled.to_csv('synthetic_water_consumption_filled.csv', index=False)

print("✅ Missing values handled, plot saved, and filled data saved!")

