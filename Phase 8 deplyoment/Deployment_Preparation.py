# Deployment_Preparation

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------- DATA LOADING --------
data = pd.read_csv('synthetic_water_consumption_labeled.csv')
X = data.drop(columns=['Date', 'anomaly_score', 'outlier', 'anomaly_label'])
y = data['anomaly_label']

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------- STANDARD SCALER --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- SAVE SCALER --------
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved successfully.")

# -------- TRAIN FINAL XGBoost Model --------
final_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
final_model.fit(X_train_scaled, y_train)

# -------- SAVE FINAL MODEL --------
final_model.save_model('xgboost_final_model.json')
print("✅ XGBoost Final Model saved successfully.")

# -------- OPTIONAL: Save Final LSTM Model --------
# Example assuming we had a trained LSTM model
# (Training code similar to Phase7)

# Build a basic LSTM again
look_back = 30
def create_sequences(series, look_back=30):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

# Create dummy LSTM save example
series = pd.read_csv('synthetic_water_consumption_scaled.csv')['Water_Consumption'].values
X_lstm, y_lstm = create_sequences(series)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(look_back, 1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)

# Save LSTM Model
lstm_model.save('lstm_final_model.h5')
print("✅ LSTM Final Model saved successfully.")
