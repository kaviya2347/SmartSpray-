"""
app/resource_optimization/router.py

Module 7 — Resource optimisation API.

    GET /resources/{survey_id}   tank-mix plan and chemical/cost comparison
"""

from fastapi import APIRouter, HTTPException

from app.decision import rules_engine
from app.decision.service import read_decisions
from app.mission_planner.service import read_mission
from app.resource_optimization.grouping import load_mission_config

router = APIRouter(prefix="/resources", tags=["Module 7 — Resource Optimisation"])


@router.get("/config")
def mission_config():
    return load_mission_config().to_dict()


@router.get("/{survey_id}")
def resources(survey_id: str):
    decisions = read_decisions(survey_id)
    if not decisions:
        raise HTTPException(status_code=404, detail=f"No decisions for '{survey_id}'")
    rules = rules_engine.load_rules()
    plan = read_mission(survey_id)
    return {
        "survey_id": survey_id,
        "chemical": rules_engine.summarise_decisions(decisions, rules),
        "mix_plan": plan.get("mix_plan", []) if plan else [],
        "tank_loads": plan["metrics"]["sorties"] if plan else None,
    }
