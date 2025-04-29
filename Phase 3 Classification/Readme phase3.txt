# Phase 4: Classification Models for Anomaly Detection

This phase focuses on building different supervised machine learning models to classify whether the system is operating normally or abnormally based on multiple sensor readings.

---

# 1️⃣ Decision Tree Classifier (Phase4_DecisionTree.py)

- **Algorithm:** Decision Tree (depth-limited to avoid overfitting).
- **Process:**
  - Data standardized using StandardScaler.
  - Model trained to separate anomalies from normal samples.
- **Evaluation Metrics:**
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix Plot
  - ROC Curve Plot with AUC score.
- **Output Files:**
  - `decision_tree_confusion_matrix.jpeg`
  - `decision_tree_roc_curve.jpeg`

---

# 2️⃣ Support Vector Machine (SVM) Classifier (Phase4_SVM.py)

- **Algorithm:** Support Vector Machine with RBF Kernel.
- **Process:**
  - Data standardized.
  - SVM model fitted for high-margin separation between anomaly and normal classes.
- **Evaluation Metrics:**
  - Classification Report
  - Confusion Matrix Plot
  - ROC Curve Plot
- **Output Files:**
  - `svm_confusion_matrix.jpeg`
  - `svm_roc_curve.jpeg`

---

# 3️⃣ XGBoost Classifier (Phase4_XGBoost.py)

- **Algorithm:** Gradient Boosted Decision Trees (XGBoost Classifier).
- **Process:**
  - Data standardized.
  - XGBoost model trained using boosting to increase accuracy and robustness.
- **Evaluation Metrics:**
  - Classification Report
  - Confusion Matrix Plot
  - ROC Curve Plot
- **Output Files:**
  - `xgboost_confusion_matrix.jpeg`
  - `xgboost_roc_curve.jpeg`

---

# ✅ Summary:

- Built multiple classifiers to detect anomalies based on multivariate sensor data.
- Visualized model performance through confusion matrices and ROC curves.
- Compared models based on AUC, Precision, Recall, and F1-scores.

Each model can be further optimized using hyperparameter tuning in later phases.

---
