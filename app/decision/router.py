"""
app/decision/router.py

Module 6 — Treatment Decision Engine API.

    POST /decisions/generate            run the rules engine for a survey
    GET  /decisions/{survey_id}         stored decisions
    GET  /decisions/rules/current       the active (auditable) rule table
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.decision import rules_engine, service

router = APIRouter(prefix="/decisions", tags=["Module 6 — Treatment Decision"])


class DecisionRequest(BaseModel):
    survey_id: str


@router.post("/generate")
def generate(request: DecisionRequest):
    try:
        return service.generate_decisions(request.survey_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/rules/current")
def current_rules():
    try:
        return rules_engine.load_rules().raw
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/{survey_id}")
def read(survey_id: str):
    decisions = service.read_decisions(survey_id)
    if not decisions:
        raise HTTPException(status_code=404, detail=f"No decisions for '{survey_id}'")
    rules = rules_engine.load_rules()
    return {"survey_id": survey_id,
            "summary": rules_engine.summarise_decisions(decisions, rules),
            "decisions": decisions}
