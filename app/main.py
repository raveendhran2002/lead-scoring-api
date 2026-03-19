# app/main.py
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Load model once when server starts
model = joblib.load("model/lead_scorer_v1.pkl")

app = FastAPI(title="Lead Scoring API", version="1.0.0")

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
    return {"status": "ok", "model": "lead_scorer_v1"}

@app.post("/score-lead", response_model=ScoreResponse)
def score_lead(lead: LeadRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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