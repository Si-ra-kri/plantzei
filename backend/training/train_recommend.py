"""
training/train_recommend.py

Trains a Random Forest Classifier to recommend the best crop.
Dataset schema mirrors Kaggle 'Crop Recommendation Dataset' (22 crops, 2200 rows).

Features: N, P, K, temperature, humidity, ph, rainfall, calamity_enc, season_enc
Target:   crop label (22 classes)

Replace SYNTHETIC mode with real CSV for production accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_SAMPLES    = 2200
MODEL_OUT    = os.path.join(os.path.dirname(__file__), "..", "models", "recommend_model.pkl")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

CROPS = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas",
    "Moth Beans", "Mung Bean", "Black-eyed Peas", "Lentil",
    "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon",
    "Muskmelon", "Apple", "Orange", "Papaya", "Coconut",
    "Cotton", "Jute", "Coffee",
]

CALAMITY_ENC = {"None": 0, "Flood": 1, "Drought": 2, "Cyclone": 3, "Hailstorm": 4}
SEASON_ENC   = {"Kharif": 0, "Rabi": 1, "Zaid": 2}

rng = np.random.default_rng(RANDOM_SEED)


# Crop-specific NPK/climate profiles (domain knowledge)
CROP_PROFILES = {
    "Rice":          {"N":(80,120),  "P":(30,60),  "K":(30,60),  "temp":(20,32), "hum":(70,90), "ph":(5.5,7.0), "rain":(150,250)},
    "Maize":         {"N":(60,100),  "P":(20,50),  "K":(20,50),  "temp":(18,30), "hum":(55,75), "ph":(5.5,7.5), "rain":(60,120)},
    "Chickpea":      {"N":(20,50),   "P":(40,80),  "K":(15,40),  "temp":(15,25), "hum":(40,60), "ph":(6.0,8.0), "rain":(40,80)},
    "Kidney Beans":  {"N":(20,50),   "P":(50,90),  "K":(15,45),  "temp":(15,27), "hum":(45,70), "ph":(6.0,7.5), "rain":(50,100)},
    "Pigeon Peas":   {"N":(15,40),   "P":(50,85),  "K":(15,40),  "temp":(18,32), "hum":(40,70), "ph":(5.5,7.0), "rain":(40,100)},
    "Moth Beans":    {"N":(15,40),   "P":(30,60),  "K":(15,35),  "temp":(27,38), "hum":(20,40), "ph":(6.0,7.8), "rain":(15,50)},
    "Mung Bean":     {"N":(15,45),   "P":(35,70),  "K":(15,40),  "temp":(20,35), "hum":(50,75), "ph":(6.0,7.5), "rain":(40,80)},
    "Black-eyed Peas":{"N":(15,40), "P":(35,65),  "K":(15,40),  "temp":(18,35), "hum":(45,75), "ph":(5.5,7.5), "rain":(40,100)},
    "Lentil":        {"N":(15,40),   "P":(35,75),  "K":(15,40),  "temp":(12,22), "hum":(40,65), "ph":(6.0,8.0), "rain":(30,70)},
    "Pomegranate":   {"N":(20,50),   "P":(30,55),  "K":(20,50),  "temp":(20,35), "hum":(30,55), "ph":(6.5,8.0), "rain":(30,65)},
    "Banana":        {"N":(80,150),  "P":(50,100), "K":(150,250),"temp":(20,35), "hum":(70,90), "ph":(5.5,7.0), "rain":(100,150)},
    "Mango":         {"N":(20,50),   "P":(15,40),  "K":(15,45),  "temp":(22,38), "hum":(40,70), "ph":(5.5,7.0), "rain":(40,100)},
    "Grapes":        {"N":(20,55),   "P":(20,50),  "K":(25,60),  "temp":(15,28), "hum":(35,60), "ph":(5.5,7.0), "rain":(30,70)},
    "Watermelon":    {"N":(50,100),  "P":(30,60),  "K":(50,100), "temp":(22,38), "hum":(60,80), "ph":(5.5,7.0), "rain":(40,80)},
    "Muskmelon":     {"N":(50,100),  "P":(30,60),  "K":(50,100), "temp":(22,38), "hum":(60,80), "ph":(6.0,7.5), "rain":(40,80)},
    "Apple":         {"N":(20,50),   "P":(20,50),  "K":(30,70),  "temp":(8,18),  "hum":(60,80), "ph":(5.5,6.5), "rain":(40,80)},
    "Orange":        {"N":(20,55),   "P":(15,40),  "K":(15,45),  "temp":(20,30), "hum":(55,75), "ph":(5.5,7.0), "rain":(60,120)},
    "Papaya":        {"N":(50,100),  "P":(30,60),  "K":(50,100), "temp":(22,38), "hum":(65,85), "ph":(6.0,7.0), "rain":(50,100)},
    "Coconut":       {"N":(15,40),   "P":(10,30),  "K":(30,70),  "temp":(22,35), "hum":(70,90), "ph":(5.5,7.0), "rain":(100,200)},
    "Cotton":        {"N":(50,100),  "P":(30,60),  "K":(30,60),  "temp":(22,38), "hum":(40,65), "ph":(6.0,8.0), "rain":(50,100)},
    "Jute":          {"N":(60,100),  "P":(20,50),  "K":(20,50),  "temp":(24,38), "hum":(70,90), "ph":(6.0,7.5), "rain":(100,200)},
    "Coffee":        {"N":(80,130),  "P":(40,80),  "K":(30,60),  "temp":(15,28), "hum":(65,85), "ph":(5.5,7.0), "rain":(80,150)},
}

def sample_from_profile(profile, n):
    """Sample feature values within a crop's agronomic profile."""
    def u(key): return rng.uniform(*profile[key], n)
    return (u("N"), u("P"), u("K"), u("temp"), u("hum"), u("ph"), u("rain"))


def generate_data(n_per_crop=100) -> pd.DataFrame:
    rows = []
    calamities = list(CALAMITY_ENC.keys())
    seasons    = list(SEASON_ENC.keys())

    for crop in CROPS:
        profile = CROP_PROFILES[crop]
        N, P, K, temp, hum, ph, rain = sample_from_profile(profile, n_per_crop)
        cal = rng.choice(calamities, n_per_crop, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        sea = rng.choice(seasons, n_per_crop)

        for i in range(n_per_crop):
            rows.append({
                "N": N[i], "P": P[i], "K": K[i],
                "temperature": temp[i], "humidity": hum[i],
                "ph": ph[i], "rainfall": rain[i],
                "calamity": CALAMITY_ENC[cal[i]],
                "season":   SEASON_ENC[sea[i]],
                "label":    crop,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def train():
    print("📦 Generating synthetic crop recommendation dataset …")
    df = generate_data(n_per_crop=100)    # 22 crops × 100 = 2200 rows

    X = df[["N","P","K","temperature","humidity","ph","rainfall","calamity","season"]].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    print("🔧 Training Random Forest Classifier (n_estimators=200) …")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Test Accuracy: {acc:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Bundle label encoder with model so routes can decode class indices
    bundle = {"model": model, "classes": le.classes_.tolist()}
    joblib.dump(bundle, MODEL_OUT)
    print(f"💾 Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    train()
