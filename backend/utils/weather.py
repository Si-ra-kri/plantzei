"""
utils/weather.py
OpenWeatherMap integration with 30-minute in-memory TTL cache.
Also exposes GET /api/weather?location=... endpoint.
"""

import os
import time
import logging

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Query

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
CACHE_TTL = 30 * 60  # 30 minutes in seconds

# Simple in-memory cache: { location_lower: {"data": {...}, "ts": float} }
_cache: dict = {}

# Regional fallback weather defaults (for when API key is missing or request fails)
REGION_DEFAULTS = {
    "maharashtra":   {"temp": 31.0, "humidity": 68.0, "rainfall": 80.0,  "condition": "Partly Cloudy"},
    "vidarbha":      {"temp": 33.0, "humidity": 60.0, "rainfall": 70.0,  "condition": "Sunny"},
    "punjab":        {"temp": 27.0, "humidity": 55.0, "rainfall": 60.0,  "condition": "Clear"},
    "rajasthan":     {"temp": 36.0, "humidity": 30.0, "rainfall": 20.0,  "condition": "Sunny"},
    "kerala":        {"temp": 28.0, "humidity": 88.0, "rainfall": 180.0, "condition": "Humid"},
    "tamil nadu":    {"temp": 30.0, "humidity": 75.0, "rainfall": 90.0,  "condition": "Humid"},
    "west bengal":   {"temp": 29.0, "humidity": 80.0, "rainfall": 120.0, "condition": "Cloudy"},
    "gujarat":       {"temp": 32.0, "humidity": 58.0, "rainfall": 50.0,  "condition": "Sunny"},
    "default":       {"temp": 30.0, "humidity": 65.0, "rainfall": 75.0,  "condition": "Partly Cloudy"},
}


def _default_for(location: str) -> dict:
    loc = location.lower()
    for key, vals in REGION_DEFAULTS.items():
        if key in loc:
            return dict(vals)
    return dict(REGION_DEFAULTS["default"])


async def get_weather(location: str) -> dict:
    """
    Fetch current weather for a location.
    Returns dict with keys: temp, humidity, rainfall, condition.
    Results are cached for 30 minutes per location string.
    Falls back to regional defaults if API key is missing or request fails.
    """
    key = location.strip().lower()
    now = time.time()

    # Cache hit
    cached = _cache.get(key)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached["data"]

    if not API_KEY:
        logger.warning("OPENWEATHER_API_KEY not set — using regional defaults")
        data = _default_for(location)
        _cache[key] = {"data": data, "ts": now}
        return data

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(BASE_URL, params={"q": location, "appid": API_KEY, "units": "metric"})
            resp.raise_for_status()
            raw = resp.json()

        rain_1h = raw.get("rain", {}).get("1h", 0.0)
        condition_raw = raw["weather"][0]["main"].lower() if raw.get("weather") else ""
        if "rain" in condition_raw or rain_1h > 0:
            condition = "Rainy"
        elif "cloud" in condition_raw:
            condition = "Cloudy"
        else:
            condition = "Sunny"

        data = {
            "temp": round(raw["main"]["temp"], 1),
            "humidity": round(raw["main"]["humidity"], 1),
            "rainfall": round(rain_1h, 1),
            "condition": condition,
        }
        _cache[key] = {"data": data, "ts": now}
        return data

    except Exception as e:
        logger.warning(f"Weather API error for '{location}': {e} — using regional defaults")
        data = _default_for(location)
        _cache[key] = {"data": data, "ts": now}
        return data


# ── REST endpoint ─────────────────────────────────────────────────────────────
@router.get("/weather")
async def weather_endpoint(location: str = Query(..., description="City or region name")):
    data = await get_weather(location)
    return {"location": location, **data}
