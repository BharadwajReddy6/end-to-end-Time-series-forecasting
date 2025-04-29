# LSTM Forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
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
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict
preds = model.predict(X_test)

# Plot
plt.figure(figsize=(14,6))
plt.plot(range(len(series)-len(y_test), len(series)), y_test, label='True')
plt.plot(range(len(series)-len(y_test), len(series)), preds.flatten(), label='LSTM Forecast', color='orange')
plt.legend()
plt.title('LSTM Forecasting')
plt.grid(True)
plt.tight_layout()
plt.savefig('phase3_lstm_forecast.jpeg', format='jpeg')
plt.show()
