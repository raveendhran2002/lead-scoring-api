# app/main.py
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Lead Scoring API", version="1.0.0")

# Load model safely with error logging
model = None
model_error = None

try:
    model_path = "model/lead_scorer_v1.pkl"
    if not os.path.exists(model_path):
        model_error = f"Model file not found at: {model_path}"
        print(f"ERROR: {model_error}")
    else:
        model = joblib.load(model_path)
        print(f"SUCCESS: Model loaded from {model_path}")
except Exception as e:
    model_error = str(e)
    print(f"ERROR loading model: {model_error}")

class LeadRequest(BaseModel):
    lead_id:             str
    lead_source:         Optional[str]   = "Unknown"
    industry:            Optional[str]   = "Unknown"
    annual_revenue:      Optional[float] = 0.0
    number_of_employees: Optional[int]   = 0

class ScoreResponse(BaseModel):
    lead_id:    str
    score:      int
    tier:       str
    confidence: float

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
        features = [[
            hash(lead.lead_source)  % 10,
            hash(lead.industry)     % 10,
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