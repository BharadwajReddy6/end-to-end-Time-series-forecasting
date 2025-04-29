# GAN-Based Synthetic Sensor Data Generation

This phase builds a Generative Adversarial Network (GAN) to create synthetic time series data that mimics the real water consumption sensor data.

---

# Components:

- **Generator:** LSTM-based model that generates fake sensor sequences.
- **Discriminator:** CNN-based model that tries to distinguish real vs fake sequences.
- **GAN:** Combined model where Generator tries to fool the Discriminator.

---

# Process:

1. Load real sensor data.
2. Create sequences (sliding windows).
3. Build LSTM Generator and CNN Discriminator.
4. Train GAN for 2000 epochs.
5. Generate new synthetic sequences.
6. Compare real vs generated samples visually.

---

# Output:

- `phase6_gan_real_vs_generated.jpeg`: Comparison plot between real and generated sequences.

---

# ✅ Summary:

GAN allows generating huge volumes of realistic synthetic sensor data, which can be used for:
- Data augmentation
- Model training on rare event simulations
- Robustness improvements

---
