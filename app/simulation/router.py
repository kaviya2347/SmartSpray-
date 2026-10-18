"""
app/simulation/router.py

HTTP API used by the 3D mission simulator (simulator/web/, served at
/simulator). Each endpoint is one thing the drone does during a mission.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.simulation import service
from app.simulation.world import aruco_png

router = APIRouter(prefix="/sim", tags=["Mission Simulator"])


class NewSession(BaseModel):
    seed: int = 7
    preset: str = "small"            # "small" (63 plants) | "medium" (117 plants)


class MarkerDrop(BaseModel):
    marker_id: int = Field(ge=0, le=3)
    x_m: float
    y_m: float


class Capture(BaseModel):
    target_id: str
    drone_x_m: float
    drone_y_m: float
    gradcam: bool = True


class Replan(BaseModel):
    position: list[float] = Field(min_length=2, max_length=2)
    tank_ml: float = 0.0
    tank_treatment: Optional[str] = None
    remaining: list[int] = []


class Finish(BaseModel):
    telemetry: dict = {}


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/session")
def new_session(req: NewSession):
    """Create a synthetic field and a new mission."""
    return _call(service.new_session, req.seed, req.preset)


@router.post("/{sim_id}/marker")
def drop_marker(sim_id: str, req: MarkerDrop):
    """The drone dropped marker `marker_id` at (x, y)."""
    return _call(service.drop_marker, sim_id, req.marker_id, req.x_m, req.y_m)


@router.post("/{sim_id}/overview")
def overview(sim_id: str):
    """High-altitude photo: find the markers (boundary) and every plant (data points)."""
    return _call(service.capture_overview, sim_id)


@router.post("/{sim_id}/capture")
def capture(sim_id: str, req: Capture):
    """Low-altitude photo of one plant -> disease model -> severity -> dose."""
    return _call(service.capture_plant, sim_id, req.target_id, req.drone_x_m,
                 req.drone_y_m, req.gradcam)


@router.post("/{sim_id}/plan")
def plan(sim_id: str):
    """Decide every plant, then the shortest capacity-aware spray route."""
    return _call(service.plan, sim_id)


@router.post("/{sim_id}/replan")
def replan(sim_id: str, req: Replan):
    """Shortest route over the remaining targets from the drone's current position."""
    return _call(service.replan, sim_id, tuple(req.position), req.tank_ml,
                 req.tank_treatment, req.remaining)


@router.post("/{sim_id}/finish")
def finish(sim_id: str, req: Finish):
    """Save the flight log and return the mission summary."""
    return _call(service.finish, sim_id, req.telemetry)


@router.get("/{sim_id}/state")
def state(sim_id: str):
    return _call(service.state, sim_id)


@router.get("/{sim_id}/files/{name:path}")
def sim_file(sim_id: str, name: str):
    try:
        path = service.file_path(sim_id, name)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


@router.get("/aruco/{marker_id}.png")
def aruco(marker_id: int):
    if marker_id not in (0, 1, 2, 3):
        raise HTTPException(status_code=404, detail="marker ids are 0-3")
    return Response(aruco_png(marker_id), media_type="image/png",
                    headers={"Cache-Control": "max-age=86400"})
