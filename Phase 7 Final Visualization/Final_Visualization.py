# Final_Visualization

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -------- DATA LOADING --------
# (Assuming synthetic_water_consumption_labeled.csv exists)
data = pd.read_csv('synthetic_water_consumption_labeled.csv')
X = data.drop(columns=['Date', 'anomaly_score', 'outlier', 'anomaly_label'])
y_true = data['anomaly_label']

# Dummy predicted values for visualization (normally, use model predictions)
np.random.seed(42)
y_pred = np.random.choice([0,1], size=len(y_true), p=[0.95,0.05])

# -------- CONFUSION MATRIX PLOT --------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Final Confusion Matrix (Dummy Data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('phase8_final_confusion_matrix.jpeg', format='jpeg')
plt.show()

# -------- ROC CURVE PLOT --------
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final ROC Curve (Dummy Data)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase8_final_roc_curve.jpeg', format='jpeg')
plt.show()

# -------- FORECAST COMPARISON PLOT --------
# Assume we have dummy forecast arrays (real would come from previous models)
days = np.arange(100)
true_values = np.sin(0.1*days) + np.random.normal(0, 0.05, size=100)
arima_forecast = true_values + np.random.normal(0, 0.1, size=100)
sarima_forecast = true_values + np.random.normal(0, 0.08, size=100)
lstm_forecast = true_values + np.random.normal(0, 0.05, size=100)
transformer_forecast = true_values + np.random.normal(0, 0.04, size=100)
xgb_forecast = true_values + np.random.normal(0, 0.07, size=100)

plt.figure(figsize=(14,8))
plt.plot(days, true_values, label='True', linewidth=2)
plt.plot(days, arima_forecast, label='ARIMA Forecast')
plt.plot(days, sarima_forecast, label='SARIMA Forecast')
plt.plot(days, lstm_forecast, label='LSTM Forecast')
plt.plot(days, transformer_forecast, label='Transformer Forecast')
plt.plot(days, xgb_forecast, label='XGBoost Forecast')
plt.legend()
plt.title('Forecast Comparison Across Models (Dummy)')
plt.xlabel('Days')
plt.ylabel('Scaled Water Consumption')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase8_forecast_comparison.jpeg', format='jpeg')
plt.show()
