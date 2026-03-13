"""
training/train_irrigation.py

Trains a Random Forest Regressor to predict:
  - flow_rate (L/hr)
  - duration (minutes)

From inputs:
  - soil_moisture (%)
  - soil_temperature (°C)
  - crop_type (label encoded)
  - weather (label encoded)
  - humidity (%)

Uses synthetic data that mirrors the Kaggle 'Plant Watering' dataset schema.
Replace DATASET_PATH with real CSV path for production accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES = 800
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "irrigation_model.pkl")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Label encodings (must match routes/irrigation.py)
CROP_ENC  = {"Rice": 0, "Wheat": 1, "Tomato": 2, "Sugarcane": 3, "Cotton": 4}
WEATHER_ENC = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}

rng = np.random.default_rng(RANDOM_SEED)


def generate_synthetic_data(n: int) -> pd.DataFrame:
    """
    Generate realistic synthetic irrigation dataset.
    Domain knowledge encoded:
      - Rainy weather → less water needed
      - High temperature → more evaporation → longer duration
      - Low moisture → more water needed
    """
    crop_labels   = rng.choice(list(CROP_ENC.keys()), n)
    weather_labels = rng.choice(list(WEATHER_ENC.keys()), n, p=[0.5, 0.3, 0.2])

    soil_moisture  = rng.uniform(10, 90, n)
    soil_temp      = rng.uniform(15, 45, n)
    humidity       = np.where(weather_labels == "Rainy", rng.uniform(70, 95, n),
                    np.where(weather_labels == "Cloudy", rng.uniform(50, 75, n),
                                                          rng.uniform(25, 55, n)))

    crop_enc    = np.array([CROP_ENC[c] for c in crop_labels])
    weather_enc = np.array([WEATHER_ENC[w] for w in weather_labels])

    # Target: flow_rate (L/hr) — higher for water-intensive crops, dry soil, hot weather
    base_flow = np.array([{"Rice": 4.0, "Sugarcane": 3.8, "Cotton": 3.0,
                            "Tomato": 2.5, "Wheat": 2.2}[c] for c in crop_labels])
    moisture_adj = 1 + (50 - soil_moisture) / 100          # 0.6 – 1.4
    weather_adj  = np.where(weather_labels == "Rainy", 0.5,
                   np.where(weather_labels == "Cloudy", 0.9, 1.2))
    flow_rate = np.clip(base_flow * moisture_adj * weather_adj + rng.normal(0, 0.2, n), 0.5, 8.0)

    # Target: duration (minutes) — longer when hot, dry, or water-intensive crop
    temp_adj = 1 + (soil_temp - 25) / 50
    duration = np.clip(30 * temp_adj * moisture_adj * weather_adj + rng.normal(0, 5, n), 10, 120)

    return pd.DataFrame({
        "soil_moisture":  soil_moisture,
        "soil_temp":      soil_temp,
        "crop_type":      crop_enc,
        "weather":        weather_enc,
        "humidity":       humidity,
        "flow_rate":      flow_rate.round(2),
        "duration":       duration.round(1),
    })


def train():
    print("📦 Generating synthetic irrigation training data …")
    df = generate_synthetic_data(N_SAMPLES)

    X = df[["soil_moisture", "soil_temp", "crop_type", "weather", "humidity"]].values
    y = df[["flow_rate", "duration"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    print("🔧 Training Random Forest Regressor (multi-output) …")
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae_flow = mean_absolute_error(y_test[:, 0], preds[:, 0])
    mae_dur  = mean_absolute_error(y_test[:, 1], preds[:, 1])
    print(f"✅ MAE — flow_rate: {mae_flow:.3f} L/hr | duration: {mae_dur:.1f} min")

    joblib.dump(model, MODEL_OUT)
    print(f"💾 Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    train()
