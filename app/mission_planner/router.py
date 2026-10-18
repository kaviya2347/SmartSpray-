"""
app/mission_planner/router.py

Module 8 — Mission planning API (uses Module 7 resource optimisation).

    POST /mission/plan           optimise the spray route + tank loads
    GET  /mission/{survey_id}    stored mission plan
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.mission_planner import service

router = APIRouter(prefix="/mission", tags=["Module 8 — Mission Planning"])


class MissionRequest(BaseModel):
    survey_id: str
    tank_capacity_ml: float | None = Field(None, gt=0)
    cruise_speed_mps: float | None = Field(None, gt=0)
    spray_flow_ml_per_s: float | None = Field(None, gt=0)


@router.post("/plan")
def plan(request: MissionRequest):
    overrides = request.model_dump(exclude={"survey_id"}, exclude_none=True)
    try:
        return service.plan_mission(request.survey_id, **overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{survey_id}")
def read(survey_id: str):
    plan = service.read_mission(survey_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"No mission plan for '{survey_id}'")
    return plan
