"""Pydantic schemas for the Crop Health / Disease Detection pillar."""

from typing import Optional
from pydantic import BaseModel


class RegionHealthResult(BaseModel):
    region: str
    status: str
    severity: str                    # "low" | "medium" | "high"
    severity_score: float
    action: str
    pesticide_type: Optional[str]
    dosage_ml_per_litre: Optional[float]
    coverage_percent: int
    confidence: float


class BatchHealthResponse(BaseModel):
    results: list[RegionHealthResult]
    total_regions: int
    regions_affected: int
