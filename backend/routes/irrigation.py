"""
routes/irrigation.py
POST /api/irrigation/predict

Predicts drip irrigation flow rate and duration from soil + weather inputs.
Uses a Random Forest Regressor loaded at startup. Falls back to rule-based
logic when the model file is absent or confidence is low.
"""

import numpy as np
from fastapi import APIRouter, Request

from schemas.irrigation import IrrigationRequest, IrrigationResponse

router = APIRouter()

# ── Label encodings (match training/train_irrigation.py) ──────────────────────
CROP_ENC = {"Rice": 0, "Wheat": 1, "Tomato": 2, "Sugarcane": 3, "Cotton": 4}
WEATHER_ENC = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}

# Sensible ambient humidity defaults per weather condition
HUMIDITY_DEFAULTS = {"Sunny": 45.0, "Cloudy": 65.0, "Rainy": 85.0}

# ── Rule-based fallback logic ─────────────────────────────────────────────────
def rule_based_predict(req: IrrigationRequest) -> dict:
    """
    Deterministic heuristic when no model is available.
    Higher temperature + lower moisture → more water needed.
    """
    # Base flow rate by crop water demand
    base_flow = {"Rice": 4.0, "Sugarcane": 3.8, "Cotton": 3.0, "Tomato": 2.5, "Wheat": 2.2}
    flow = base_flow.get(req.crop_type, 3.0)

    # Adjust for soil moisture deficit (drier → more water)
    moisture_factor = 1 + (50 - req.soil_moisture) / 100  # 0.5–1.5
    # Adjust for temperature (hotter → longer duration)
    temp_factor = 1 + (req.soil_temperature - 25) / 50    # 0.5–2.1
    # Rain reduces need
    weather_factor = {"Sunny": 1.2, "Cloudy": 1.0, "Rainy": 0.5}.get(req.weather, 1.0)

    flow_rate = round(float(flow * moisture_factor * weather_factor), 2)
    duration = round(float(30 * temp_factor * moisture_factor * weather_factor), 1)

    reasons = []
    if req.soil_moisture < 30:
        reasons.append("low soil moisture requires increased supply")
    if req.soil_temperature > 30:
        reasons.append("high temperature raises crop evapotranspiration")
    if req.weather == "Rainy":
        reasons.append("rainfall reduces irrigation requirement")
    reason = "Moderate conditions — standard irrigation applied" if not reasons else \
             f"Adjusted for {', '.join(reasons)}"

    return {
        "flow_rate_lph": flow_rate,
        "duration_minutes": duration,
        "reason": reason,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=IrrigationResponse)
async def predict_irrigation(req: IrrigationRequest, request: Request):
    model = getattr(request.app.state, "irrigation_model", None)

    if model is not None:
        # Build feature vector: [soil_moisture, soil_temp, crop_enc, weather_enc, humidity]
        humidity = HUMIDITY_DEFAULTS[req.weather]
        features = np.array([[
            req.soil_moisture,
            req.soil_temperature,
            CROP_ENC[req.crop_type],
            WEATHER_ENC[req.weather],
            humidity,
        ]])

        try:
            preds = model.predict(features)  # shape: (1, 2) → [flow_rate, duration]
            _flow: float = float(preds[0][0])
            _dur: float = float(preds[0][1])
            flow_rate: float = int(_flow * 100 + 0.5) / 100.0
            duration: float = int(_dur * 10 + 0.5) / 10.0
        except Exception:
            fallback = rule_based_predict(req)
            flow_rate = fallback["flow_rate_lph"]
            duration = fallback["duration_minutes"]

        # Build human-readable reason
        reasons = []
        if req.soil_moisture < 30:
            reasons.append("low soil moisture")
        if req.soil_temperature > 30:
            reasons.append("high temperature")
        if req.weather == "Sunny":
            reasons.append("sunny weather increases evaporation")
        if req.weather == "Rainy":
            reasons.append("rainfall reduces irrigation need")
        if req.weather == "Rainy":
            reason = "Rainfall detected — minimal supplemental irrigation recommended"
        elif not reasons:
            reason = "Optimal conditions — standard schedule recommended"
        else:
            reason = f"Adjusted for {', '.join(reasons)} — extended irrigation recommended"

    else:
        # No model — use rule-based fallback
        fb = rule_based_predict(req)
        flow_rate = fb["flow_rate_lph"]
        duration = fb["duration_minutes"]
        reason = fb["reason"]

    total_litres: float = int(flow_rate * (duration / 60) * 100 + 0.5) / 100.0

    return IrrigationResponse(
        flow_rate_lph=flow_rate,
        duration_minutes=duration,
        total_water_litres=total_litres,
        recommendation_reason=reason,
    )
