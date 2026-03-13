"""
utils/calamity.py
Static calamity lookup by Indian state.
Exposes GET /api/calamity/recent?state=Maharashtra
"""

from fastapi import APIRouter, Query

router = APIRouter()

# Static lookup — manually updated; key = state lowercase
# Structure: { "state": {"calamity": str, "severity": str, "note": str} }
CALAMITY_DATA: dict = {
    "maharashtra":    {"calamity": "Flood",    "severity": "moderate", "note": "Vidarbha region reported flooding in Sep 2024"},
    "gujarat":        {"calamity": "Cyclone",  "severity": "low",      "note": "Cyclone Biparjoy residual effects in coastal areas"},
    "rajasthan":      {"calamity": "Drought",  "severity": "high",     "note": "Western Rajasthan drought advisory active"},
    "punjab":         {"calamity": "None",     "severity": "none",     "note": "No recent calamity reported"},
    "haryana":        {"calamity": "Hailstorm","severity": "low",      "note": "Isolated hailstorm events in Ambala district"},
    "west bengal":    {"calamity": "Cyclone",  "severity": "moderate", "note": "Bay of Bengal cyclone watch active"},
    "odisha":         {"calamity": "Flood",    "severity": "high",     "note": "Mahanadi basin flooding reported"},
    "andhra pradesh": {"calamity": "Cyclone",  "severity": "moderate", "note": "Coastal Andhra under cyclone watch"},
    "telangana":      {"calamity": "Flood",    "severity": "low",      "note": "Minor flooding in Godavari belt"},
    "uttar pradesh":  {"calamity": "Flood",    "severity": "moderate", "note": "Eastern UP — Ganga-Ghaghra flooding"},
    "bihar":          {"calamity": "Flood",    "severity": "high",     "note": "North Bihar annual flood season active"},
    "kerala":         {"calamity": "Flood",    "severity": "moderate", "note": "Heavy monsoon reported in Wayanad"},
    "karnataka":      {"calamity": "Drought",  "severity": "low",      "note": "Northern Karnataka minor drought watch"},
    "tamil nadu":     {"calamity": "Cyclone",  "severity": "low",      "note": "Northeast monsoon related cyclone risk"},
    "himachal pradesh":{"calamity": "Hailstorm","severity":"moderate", "note": "Apple belt hailstorm damage reported"},
    "uttarakhand":    {"calamity": "Flood",    "severity": "low",      "note": "Cloudbursts in Chamoli district"},
    "assam":          {"calamity": "Flood",    "severity": "high",     "note": "Brahmaputra flood — Kaziranga region affected"},
    "chhattisgarh":   {"calamity": "None",     "severity": "none",     "note": "No active calamity"},
    "madhya pradesh": {"calamity": "Drought",  "severity": "low",      "note": "Bundelkhand region low-rainfall advisory"},
    "default":        {"calamity": "None",     "severity": "none",     "note": "No active calamity data for this state"},
}


def get_calamity_for_state(state: str) -> dict:
    key = state.strip().lower()
    for k, v in CALAMITY_DATA.items():
        if k in key or key in k:
            return {"state": state, **v}
    return {"state": state, **CALAMITY_DATA["default"]}


# ── REST endpoint ─────────────────────────────────────────────────────────────
@router.get("/calamity/recent")
async def recent_calamity(state: str = Query(..., description="Indian state name")):
    return get_calamity_for_state(state)
