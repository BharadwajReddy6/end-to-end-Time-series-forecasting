# XGBoost

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

import xgboost as xgb

# -------- DATA LOADING --------
data = pd.read_csv('synthetic_water_consumption_labeled.csv')

X = data.drop(columns=['Date', 'anomaly_score', 'outlier', 'anomaly_label'])
y = data['anomaly_label']

# -------- DATA SPLIT & SCALING --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- MODEL TRAINING --------
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# -------- PREDICTION --------
y_pred_xgb = xgb_model.predict(X_test_scaled)

# -------- EVALUATION --------
print("\n=== XGBoost Classifier Report ===")
print(classification_report(y_test, y_pred_xgb))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.jpeg', format='jpeg')
plt.show()

# -------- ROC CURVE --------
y_probs = xgb_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('xgboost_roc_curve.jpeg', format='jpeg')
plt.show()
