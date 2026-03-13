"""Pydantic schemas for the Irrigation pillar."""

from enum import Enum
from pydantic import BaseModel, Field


class CropType(str, Enum):
    RICE = "Rice"
    WHEAT = "Wheat"
    TOMATO = "Tomato"
    SUGARCANE = "Sugarcane"
    COTTON = "Cotton"


class WeatherCondition(str, Enum):
    SUNNY = "Sunny"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"


class IrrigationRequest(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage (0–100)")
    soil_temperature: float = Field(..., ge=-10, le=80, description="Soil temperature in °C")
    crop_type: CropType
    weather: WeatherCondition

    model_config = {
        "json_schema_extra": {
            "example": {
                "soil_moisture": 42,
                "soil_temperature": 28,
                "crop_type": "Rice",
                "weather": "Sunny",
            }
        }
    }


class IrrigationResponse(BaseModel):
    flow_rate_lph: float
    duration_minutes: float
    total_water_litres: float
    recommendation_reason: str
