"""Pydantic schemas for the Crop Recommendation pillar."""

from pydantic import BaseModel, Field


class CropRecommendRequest(BaseModel):
    """User only provides a free-text location (district, state)."""

    location: str = Field(
        ...,
        min_length=2,
        description="District and state, e.g. 'Thanjavur, Tamil Nadu'",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "Thanjavur, Tamil Nadu",
            }
        }
    }


class AutoDetectedInfo(BaseModel):
    season: str
    temperature: float
    humidity: float
    rainfall: float
    weather_condition: str
    recent_calamity: str
    calamity_source: str | None = None
    soil_N: float
    soil_P: float
    soil_K: float
    soil_ph: float


class CropRank(BaseModel):
    rank: int
    crop: str
    confidence: float
    reason: str


class CropRecommendResponse(BaseModel):
    location: str
    auto_detected: AutoDetectedInfo
    top_crops: list[CropRank]
    advisory: str
