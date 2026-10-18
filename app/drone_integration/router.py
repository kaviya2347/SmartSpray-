"""
app/drone_integration/router.py

Module 9 — Drone command API.

    POST /drone/commands/generate                  build the command queue
    GET  /drone/commands/{survey_id}               queued commands
    GET  /drone/mission/{survey_id}/{sortie}.waypoints   QGroundControl file
    POST /drone/simulate                           run the software executor
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.drone_integration import command_queue

router = APIRouter(prefix="/drone", tags=["Module 9 — Drone Integration"])


class CommandRequest(BaseModel):
    survey_id: str


class SimulateRequest(BaseModel):
    survey_id: str
    gnss_noise_m: float = Field(0.05, ge=0)
    seed: int = 7


@router.post("/commands/generate")
def generate(request: CommandRequest):
    try:
        return command_queue.generate_commands(request.survey_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/commands/{survey_id}")
def read(survey_id: str):
    cmds = command_queue.read_commands(survey_id)
    if not cmds:
        raise HTTPException(status_code=404, detail=f"No commands for '{survey_id}'")
    return {"survey_id": survey_id, "total": len(cmds), "commands": cmds}


@router.get("/mission/{survey_id}/{sortie}.waypoints", response_class=PlainTextResponse)
def waypoints(survey_id: str, sortie: int):
    text = command_queue.waypoint_text(survey_id, sortie)
    if text is None:
        raise HTTPException(status_code=404, detail="Unknown survey or sortie")
    return text


@router.post("/simulate")
def simulate(request: SimulateRequest):
    from simulator.executor_simulator import run

    try:
        return run(request.survey_id, request.gnss_noise_m, request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
