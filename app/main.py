# app/main.py
import joblib
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Lead Scoring API", version="1.0.0")

# ── Load model ────────────────────────────────────────────────
model = None
model_error = None

try:
    model_path = "model/lead_scorer_v1.pkl"
    if not os.path.exists(model_path):
        model_error = f"Model file not found at: {model_path}"
        print(f"ERROR: {model_error}")
    else:
        model = joblib.load(model_path)
        print("SUCCESS: Model loaded")
except Exception as e:
    model_error = str(e)
    print(f"ERROR loading model: {model_error}")

# ── Exact encoding maps — must match what you used in Colab ───
LEAD_SOURCE_MAP = {
    "Web": 0,
    "Phone Inquiry": 1,
    "Partner Referral": 2,
    "Purchased List": 3,
    "Other": 4
}

INDUSTRY_MAP = {
    "Technology": 0,
    "Finance": 1,
    "Healthcare": 2,
    "Retail": 3,
    "Manufacturing": 4,
    "Education": 5,
    "Other": 6
}

RATING_MAP = {
    "Hot": 0,
    "Warm": 1,
    "Cold": 2
}

STATUS_MAP = {
    "Open - Not Contacted": 0,
    "Working - Contacted": 1,
    "Closed - Not Converted": 2,
    "Closed - Converted": 3
}

COUNTRY_MAP = {
    "USA": 0,
    "UK": 1,
    "India": 2,
    "Canada": 3,
    "Australia": 4,
    "Other": 5
}

TITLE_MAP = {
    "CEO": 0,
    "CTO": 1,
    "Manager": 2,
    "Director": 3,
    "VP": 4,
    "Other": 5
}

def encode(mapping: dict, value: str) -> int:
    # If value not found in map, return last index + 1
    return mapping.get(value, len(mapping))

# ── Request / Response schemas ────────────────────────────────
class LeadRequest(BaseModel):
    lead_id:             str
    lead_source:         Optional[str]   = "Other"
    industry:            Optional[str]   = "Other"
    rating:              Optional[str]   = "Warm"
    status:              Optional[str]   = "Open - Not Contacted"
    country:             Optional[str]   = "Other"
    title:               Optional[str]   = "Other"
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
        "model_loaded": model is not None,
        "model_error": model_error
    }

@app.post("/score-lead", response_model=ScoreResponse)
def score_lead(lead: LeadRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Reason: {model_error}"
        )
    try:
        # Build exactly 8 features in exact training order:
        # LeadSource_enc, Industry_enc, Rating_enc,
        # Status_enc, Country_enc, Title_enc,
        # AnnualRevenue, NumberOfEmployees
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