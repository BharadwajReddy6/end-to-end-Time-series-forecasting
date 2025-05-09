
# Data Preprocessing, Feature Engineering, and Basic Analysis

This phase focuses on preparing the time series sensor dataset for Machine Learning models by applying necessary preprocessing techniques and understanding the data through exploratory plots.

---

## Missing Value Handling (Next Observation Carried Backward)

- **Technique:** Used 'Next Observation Carried Backward (NOCB)' method to fill missing values.
- **Why:** Ensures continuity in sensor signals without introducing artificial biases.
- **Plot:** 
  - Left: Missing values BEFORE filling (red bars)
  - Right: Missing values AFTER filling (green bars, showing zero missing)

- 📷 **Plot file:** `phase1_missing_values_handling.jpeg`

---

## Scaling (MinMaxScaler)

- **Technique:** Applied MinMaxScaler to scale all features between [0,1].
- **Why:** Deep Learning and distance-based models perform better on normalized data.
- **Plot:** 
  - Left: Original Water Consumption signal (large fluctuations)
  - Right: Scaled Water Consumption signal (compressed between 0 and 1)

- 📷 **Plot file:** `phase1_scaling_water_consumption.jpeg`

---

## STL Decomposition (Trend, Seasonality, Residual Extraction)

- **Technique:** Seasonal-Trend decomposition using LOESS (STL).
- **Why:** Helps separate out underlying trend, periodic patterns (seasonality), and random noise (residual).
- **Plot:**
  - First Panel: Original Water Consumption
  - Second Panel: Trend (long-term movement)
  - Third Panel: Seasonal variation (yearly repeating pattern)
  - Fourth Panel: Random noise/residuals

- 📷 **Plot file:** `phase1_stl_decomposition.jpeg`

---

## FFT Spectrum Analysis

- **Technique:** Fast Fourier Transform (FFT) on Water Consumption.
- **Why:** To discover dominant frequency components and periodicity in the data.
- **Plot:**
  - X-axis: Frequencies
  - Y-axis: Signal Magnitude
  - Shows which frequencies dominate (e.g., yearly, seasonal cycles)

- 📷 **Plot file:** `phase1_fft_spectrum.jpeg`

---

## Quantile Analysis

- **Technique:** Calculated statistical quantiles (10th, 25th, 50th, 75th, 90th percentile) for each feature.
- **Why:** Understand data distribution and variability for each sensor signal.
- **Plot:**
  - Bar chart showing quantile values across all features.

- 📷 **Plot file:** `phase1_quantiles.jpeg`

---

## Correlation Analysis

- **Technique:** Pearson Correlation Matrix calculation.
- **Why:** To understand the linear relationships between features.
- **Plot:**
  - Heatmap showing how strongly features are correlated (positive, negative, or no correlation).

- 📷 **Plot file:** `phase1_correlation_heatmap.jpeg`

---

# ✅ Summary:

This phase ensures the dataset is:
- Complete (no missing values)
- Scaled properly
- Understood in terms of temporal patterns, frequency domain patterns, feature distributions, and inter-feature relationships.

Ready for building ML/DL models in next phases!

