# Time series GAN

# -------- IMPORTS --------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -------- DATA LOADING --------
data = pd.read_csv('synthetic_water_consumption_scaled.csv')
series = data['Water_Consumption'].values

# -------- SEQUENCE CREATION --------
def create_sequences(series, look_back=30):
    X = []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
    return np.array(X)

look_back = 30
real_series = create_sequences(series, look_back)
real_series = real_series.reshape((-1, look_back, 1))

# -------- GENERATOR --------
def build_generator():
    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back, 1)))
    model.add(Dense(look_back))
    model.add(Reshape((look_back, 1)))
    return model

# -------- DISCRIMINATOR --------
def build_discriminator():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=(look_back, 1), padding="same"))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# -------- BUILD MODELS --------
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])

# GAN Model (Generator + Discriminator combined)
z = Input(shape=(look_back,1))
generated_series = generator(z)
discriminator.trainable = False
validity = discriminator(generated_series)

gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002))

# -------- TRAINING --------
epochs = 2000
batch_size = 32
half_batch = batch_size // 2

for epoch in range(epochs):

    # Train Discriminator
    idx = np.random.randint(0, real_series.shape[0], half_batch)
    real_samples = real_series[idx]

    noise = np.random.normal(0, 1, (half_batch, look_back, 1))
    gen_samples = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_samples, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, look_back, 1))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# -------- SYNTHETIC DATA GENERATION --------
noise = np.random.normal(0, 1, (10, look_back, 1))
generated_series = generator.predict(noise)

# Plot real vs generated
plt.figure(figsize=(12,6))
for i in range(5):
    plt.plot(real_series[i].flatten(), color='blue', alpha=0.5, label='Real' if i == 0 else "")
    plt.plot(generated_series[i].flatten(), color='red', linestyle='dashed', alpha=0.7, label='Generated' if i == 0 else "")
plt.legend()
plt.title('Real vs Generated Synthetic Sensor Data')
plt.tight_layout()
plt.savefig('phase6_gan_real_vs_generated.jpeg', format='jpeg')
plt.show()

print("✅ GAN Training and Synthetic Data Generation Completed!")
