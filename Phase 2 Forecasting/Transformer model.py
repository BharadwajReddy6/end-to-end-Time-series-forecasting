#Transformer Forecasting (Simple Encoder-Decoder)
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout 
# Load scaled dataset
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption']

# Prepare sequence data
def create_sequences(series, look_back=30):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)
look_back = 30
X, y = create_sequences(series.values, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split train-test
train_size = int(len(X) * 0.8)
# Prepare Transformer data
X_tf, y_tf = create_sequences(series.values, look_back)
X_tf = X_tf.reshape((X_tf.shape[0], X_tf.shape[1], 1))

# Split
X_train_tf, X_test_tf = X_tf[:train_size], X_tf[train_size:]
y_train_tf, y_test_tf = y_tf[:train_size], y_tf[train_size:]

# Transformer model
inputs = tf.keras.Input(shape=(look_back, 1))
x = layers.Dense(64)(inputs)
x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)

transformer_model = tf.keras.Model(inputs, outputs)
transformer_model.compile(loss='mse', optimizer='adam')
transformer_model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32, verbose=1)

# Predict
tf_preds = transformer_model.predict(X_test_tf)

# Plot
plt.figure(figsize=(14,6))
plt.plot(range(len(series)-len(y_test_tf), len(series)), y_test_tf, label='True')
plt.plot(range(len(series)-len(y_test_tf), len(series)), tf_preds.flatten(), label='Transformer Forecast', color='purple')
plt.legend()
plt.title('Transformer Forecasting')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase3_transformer_forecast.jpeg', format='jpeg')
plt.show()
