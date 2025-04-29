# Correlation Heatmap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scaled dataset
data = pd.read_csv('synthetic_water_consumption_scaled.csv')

# Compute correlation matrix
correlation_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()

# Save plot
plt.savefig('phase1_correlation_heatmap.jpeg', format='jpeg')
plt.show()

print("✅ Correlation Heatmap generated and saved.")
