# Forecasting water consumption on a synthetic generated data Using Machine Learning and Deep Learning. This is done to demonstrate my work for predictive maintenance project. 

---

## Project Overview

This project focuses on forecasting daily water consumption and detecting anomalous usage patterns based on multiple influencing factors such as atmospheric temperature, rainfall, humidity, population, and hotel occupancy. 

By leveraging multivariate time series sensor-like data, we implemented an end-to-end machine learning and deep learning pipeline to:

- Predict future water demand.
- Detect anomalies in consumption behavior.
- Classify normal versus abnormal operating conditions.

The project simulates a realistic, large-scale city water usage dataset over 20 years, using synthetic data generation techniques to mimic real-world behavior.

The complete solution integrates data preprocessing, forecasting, anomaly detection, classification, hyperparameter tuning, GAN-based synthetic data generation, MLflow tracking, and final model deployment preparation.


---

## TechStack Used

- Python (Pandas, Numpy, Matplotlib, Seaborn)
- Scikit-learn (Isolation Forest, Decision Trees, SVM, etc.)
- TensorFlow/Keras (LSTM, Transformer, GANs)
- XGBoost
- MLflow (for experiment tracking)

---

## Core Phases Completed

- Phase 0: Synthetic data generation mimicking real-world scenarios.
- Phase 1: Data cleaning, scaling, decomposition (STL), FFT analysis, and correlation study.Anomaly detection using Isolation Forest.
- Phase 2: Forecasting future failures using ARIMA, SARIMA, LSTM, Transformer, and XGBoost models.
- Phase 3: Classification of system working modes and failure detection using Decision Tree, SVM, and XGBoost classifiers.
- Phase 4: Hyperparameter tuning for LSTM and Transformer models using Grid Search and Bayesian Optimization.
- Phase 5: Synthetic time series data generation using GAN (Generator: LSTM, Discriminator: CNN).
- Phase 6: Full MLflow integration for experiment and model tracking.
- Phase 7: Final visualizations including confusion matrices, ROC curves, and forecasting comparison plots.
- Phase 8: Model deployment preparation including saving scalers and models.

---

## Deployment Readiness

- Saved models (XGBoost and LSTM)
- Saved scaler for data preprocessing
- Project structured for easy deployment in cloud platforms or Dockerized environments.

---

## Conclusion

This project delivers a production-ready predictive Forecasting pipeline demonstrating advanced ML and DL techniques for real-world sensor data forecasting and anomaly detection. It showcases deep expertise across the entire ML lifecycle from data generation to deployment.

---
