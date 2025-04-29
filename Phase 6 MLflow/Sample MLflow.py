# MLFlow

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# -------- SETUP MLflow --------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Assuming local server
mlflow.set_experiment("PredictiveMaintenance-DeepLearning")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- BUILD MODEL FUNCTION --------
def build_simple_lstm_model(units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# -------- MLflow EXPERIMENT LOGGING --------
with mlflow.start_run():

    params = {
        "units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }

    model = build_simple_lstm_model(
        units=params["units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=0.2,
        verbose=1
    )

    # Log parameters
    mlflow.log_params(params)

    # Log final loss
    final_val_loss = history.history['val_loss'][-1]
    mlflow.log_metric("final_val_loss", final_val_loss)

    # Log the model
    mlflow.tensorflow.log_model(model, "model")

    print("✅ MLflow tracking completed and logged to experiment dashboard.")
