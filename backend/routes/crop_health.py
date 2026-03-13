"""
routes/crop_health.py
POST /api/health/analyze        — single image disease detection
POST /api/health/analyze-batch  — up to 10 images

Uses MobileNetV2 CNN (disease_model.h5) when available.
Falls back to a deterministic mock based on image byte patterns when no model is loaded.
"""

import hashlib
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from utils.image_utils import preprocess_image
from schemas.crop_health import RegionHealthResult, BatchHealthResponse

router = APIRouter()

# ── Disease class definitions ─────────────────────────────────────────────────
CLASSES = ["Healthy", "Fungi/Blight", "Bacterial Spot", "Insect Damage", "Rust", "Mosaic Virus"]

PESTICIDE_MAP = {
    "Healthy":         {"type": None,              "dosage": None, "action": "None"},
    "Fungi/Blight":    {"type": "fungicide",       "dosage": 2.0,  "action": "Spray fungicide at 2 ml/litre, 80% coverage"},
    "Bacterial Spot":  {"type": "bactericide",     "dosage": 1.5,  "action": "Spray bactericide at 1.5 ml/litre, 70% coverage"},
    "Insect Damage":   {"type": "insecticide",     "dosage": 1.5,  "action": "Spray insecticide at 1.5 ml/litre, 60% coverage"},
    "Rust":            {"type": "fungicide",       "dosage": 2.5,  "action": "Spray fungicide at 2.5 ml/litre, 90% coverage"},
    "Mosaic Virus":    {"type": "neonicotinoid",   "dosage": 1.0,  "action": "Spray neonicotinoid at 1 ml/litre, 75% coverage"},
}


def severity_from_score(score: float) -> str:
    if score < 0.4:
        return "low"
    elif score < 0.7:
        return "medium"
    return "high"


# ── Rule-based mock when model is absent ──────────────────────────────────────
def mock_predict(image_bytes: bytes) -> tuple[str, float]:
    """
    Deterministic mock using MD5 hash of image bytes as a seed.
    Returns (class_label, confidence).
    """
    digest = int(hashlib.md5(image_bytes).hexdigest(), 16)
    class_idx = digest % len(CLASSES)
    confidence = 0.55 + (digest % 100) / 250  # 0.55 – 0.95
    return CLASSES[class_idx], round(confidence, 2)


# ── Model-based prediction ────────────────────────────────────────────────────
def model_predict(model, image_bytes: bytes) -> tuple[str, float]:
    import numpy as np
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)           # shape: (1, num_classes)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])
    label = CLASSES[class_idx] if class_idx < len(CLASSES) else "Healthy"
    return label, round(confidence, 2)


# ── Build result dict from prediction ────────────────────────────────────────
def build_result(region: str, label: str, confidence: float) -> RegionHealthResult:
    severity_score = confidence if label != "Healthy" else 0.1
    severity = severity_from_score(severity_score)
    coverage = round(severity_score * 100)
    pest = PESTICIDE_MAP.get(label, PESTICIDE_MAP["Healthy"])

    return RegionHealthResult(
        region=region,
        status=label,
        severity=severity,
        severity_score=round(severity_score, 2),
        action=pest["action"],
        pesticide_type=pest["type"],
        dosage_ml_per_litre=pest["dosage"],
        coverage_percent=coverage,
        confidence=confidence,
    )


# ── Single image endpoint ─────────────────────────────────────────────────────
@router.post("/analyze", response_model=RegionHealthResult)
async def analyze_image(
    request: Request,
    image: UploadFile = File(..., description="Crop image (JPEG/PNG, max 10 MB)"),
    region_name: Optional[str] = Form(default="Region A"),
):
    # File size guard
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    content_type = image.content_type or ""
    if content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted")

    model = getattr(request.app.state, "disease_model", None)
    try:
        if model is not None:
            label, confidence = model_predict(model, contents)
        else:
            label, confidence = mock_predict(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return build_result(region_name or "Region A", label, confidence)


# ── Batch endpoint ────────────────────────────────────────────────────────────
@router.post("/analyze-batch", response_model=BatchHealthResponse)
async def analyze_batch(
    request: Request,
    images: list[UploadFile] = File(..., description="Up to 10 crop images"),
    region_names: Optional[str] = Form(default=None, description="Comma-separated region names"),
):
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")

    names = [n.strip() for n in region_names.split(",")] if region_names else []

    model = getattr(request.app.state, "disease_model", None)
    results = []

    for i, img_file in enumerate(images):
        region = names[i] if i < len(names) else f"Region {chr(65 + i)}"
        contents = await img_file.read()

        if len(contents) > 10 * 1024 * 1024:
            continue  # skip oversized files silently

        try:
            if model is not None:
                label, confidence = model_predict(model, contents)
            else:
                label, confidence = mock_predict(contents)
        except Exception:
            label, confidence = "Healthy", 0.5

        results.append(build_result(region, label, confidence))

    affected = sum(1 for r in results if r.status != "Healthy")
    return BatchHealthResponse(results=results, total_regions=len(results), regions_affected=affected)
