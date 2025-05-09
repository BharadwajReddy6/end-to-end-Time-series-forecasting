# SVM

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

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
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# -------- PREDICTION --------
y_pred_svm = svm_model.predict(X_test_scaled)

# -------- EVALUATION --------
print("\n=== SVM Classifier Report ===")
print(classification_report(y_test, y_pred_svm))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.jpeg', format='jpeg')
plt.show()

# -------- ROC CURVE --------
y_probs = svm_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('svm_roc_curve.jpeg', format='jpeg')
plt.show()
