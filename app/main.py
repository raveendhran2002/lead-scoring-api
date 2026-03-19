# app/main.py
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Lead Scoring API", version="1.0.0")

# ── Load model and encoders ───────────────────────────────────
model       = None
encoders    = None
model_error = None

try:
    model_path    = "model/lead_scorer_v1.pkl"
    encoders_path = "model/encoders.pkl"

    if not os.path.exists(model_path):
        model_error = f"Model file not found: {model_path}"
        print(f"ERROR: {model_error}")
    elif not os.path.exists(encoders_path):
        model_error = f"Encoders file not found: {encoders_path}"
        print(f"ERROR: {model_error}")
    else:
        model    = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        print("SUCCESS: Model and encoders loaded")

except Exception as e:
    model_error = str(e)
    print(f"ERROR: {model_error}")

# ── Exact encoding maps extracted from your encoders.pkl ─────

LEAD_SOURCE_MAP = {
    "Other": 0,
    "Partner Referral": 1,
    "Phone Inquiry": 2,
    "Purchased List": 3,
    "Web": 4
}

INDUSTRY_MAP = {
    "Education": 0,
    "Finance": 1,
    "Healthcare": 2,
    "Manufacturing": 3,
    "Retail": 4,
    "Technology": 5
}

RATING_MAP = {
    "Cold": 0,
    "Hot": 1,
    "Warm": 2
}

STATUS_MAP = {
    "Closed - Not Converted": 0,
    "Open - Not Contacted": 1,
    "Working - Contacted": 2
}

COUNTRY_MAP = {
    "Australia": 0,
    "Canada": 1,
    "India": 2,
    "United Kingdom": 3,
    "United States": 4
}

TITLE_MAP = {
    "Analyst": 0,
    "CEO": 1,
    "CTO": 2,
    "Director": 3,
    "Engineer": 4,
    "Manager": 5
}

def encode(mapping: dict, value: str) -> int:
    # Unknown value → return 0 as safe fallback
    return mapping.get(value, 0)

# ── Schemas ───────────────────────────────────────────────────
class LeadRequest(BaseModel):
    lead_id:             str
    lead_source:         Optional[str]   = "Other"
    industry:            Optional[str]   = "Education"
    rating:              Optional[str]   = "Warm"
    status:              Optional[str]   = "Open - Not Contacted"
    country:             Optional[str]   = "India"
    title:               Optional[str]   = "Manager"
    annual_revenue:      Optional[float] = 0.0
    number_of_employees: Optional[int]   = 0

class ScoreResponse(BaseModel):
    lead_id:    str
    score:      int
    tier:       str
    confidence: float

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded":    model    is not None,
        "encoders_loaded": encoders is not None,
        "model_error":     model_error
    }

@app.post("/score-lead", response_model=ScoreResponse)
def score_lead(lead: LeadRequest):
    if model is None or encoders is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model or encoders not loaded: {model_error}"
        )
    try:
        # Exactly 8 features in exact training order
        features = [[
            encode(LEAD_SOURCE_MAP, lead.lead_source),
            encode(INDUSTRY_MAP,    lead.industry),
            encode(RATING_MAP,      lead.rating),
            encode(STATUS_MAP,      lead.status),
            encode(COUNTRY_MAP,     lead.country),
            encode(TITLE_MAP,       lead.title),
            lead.annual_revenue,
            lead.number_of_employees
        ]]

        prob  = float(model.predict_proba(features)[0][1])
        score = int(prob * 100)
        tier  = "Hot" if score >= 70 else "Warm" if score >= 40 else "Cold"

        return ScoreResponse(
            lead_id=lead.lead_id,
            score=score,
            tier=tier,
            confidence=round(prob, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
