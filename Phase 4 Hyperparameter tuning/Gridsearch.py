# Grid search

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras_tuner import RandomSearch, BayesianOptimization

# -------- DATA LOADING --------
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption']

# -------- SEQUENCE CREATION --------
def create_sequences(series, look_back=30):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

look_back = 30
X, y = create_sequences(series.values, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------- DATA SPLIT --------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- MODEL BUILD FUNCTION --------
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                   input_shape=(look_back, 1)))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='mse')
    return model

# -------- HYPERPARAMETER TUNING: RANDOM SEARCH --------
tuner_rs = RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='lstm_randomsearch',
    project_name='lstm_tuning'
)

tuner_rs.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# -------- HYPERPARAMETER TUNING: BAYESIAN OPTIMIZATION --------
tuner_bayes = BayesianOptimization(
    build_lstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='lstm_bayesopt',
    project_name='lstm_tuning_bayes'
)

tuner_bayes.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

print("✅ LSTM Hyperparameter Tuning completed!")
