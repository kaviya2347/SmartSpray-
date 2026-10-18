"""
app/main.py

Entry point for the SmartSpray backend (FastAPI).

    uvicorn app.main:app --reload        ->  http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import init_db

from app.field_survey.router import router as field_survey_router
from app.detection.router import router as detection_router
from app.classification.router import router as classification_router
from app.severity.router import router as severity_router
from app.spatial_mapping.router import router as spatial_mapping_router
from app.decision.router import router as decision_router
from app.resource_optimization.router import router as resource_router
from app.mission_planner.router import router as mission_router
from app.drone_integration.router import router as drone_router
from app.simulation.router import router as simulation_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SmartSpray API",
    description=(
        "Vision-Guided Autonomous UAV for Plant-Level Disease Detection and "
        "Variable-Rate Spraying: survey -> plant detection -> disease "
        "classification -> severity -> field mapping -> treatment decision -> "
        "route -> drone commands. 3D mission simulator at /simulator/."
    ),
    version="0.9.0",
    lifespan=lifespan,
)

app.include_router(field_survey_router)      # Module 1 — Field Survey
app.include_router(detection_router)         # Module 2 — Plant Detection
app.include_router(classification_router)    # Module 3 — Disease Classification
app.include_router(severity_router)          # Module 4 — Severity Estimation
app.include_router(spatial_mapping_router)   # Module 5 — Spatial Mapping
app.include_router(decision_router)          # Module 6 — Treatment Decision
app.include_router(resource_router)          # Module 7 — Resource Optimisation
app.include_router(mission_router)           # Module 8 — Mission Planning
app.include_router(drone_router)             # Module 9 — Drone Integration
app.include_router(simulation_router)        # 3D mission simulator API

SIMULATOR_WEB = REPO_ROOT / "simulator" / "web"
if SIMULATOR_WEB.is_dir():
    app.mount("/simulator", StaticFiles(directory=SIMULATOR_WEB, html=True), name="simulator")


class PipelineRequest(BaseModel):
    survey_id: str
    detector: str | None = None      # "yolo" | "auto" | "classical"
    save_gradcam: bool = False
    simulate: bool = False
    write_report: bool = True


@app.get("/", tags=["System"])
def root():
    return {"name": "SmartSpray API", "docs": "/docs", "health": "/health",
            "simulator": "/simulator/"}


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def favicon():
    # Browsers ask for these automatically; answering keeps the server log
    # free of harmless 404s.
    icon = SIMULATOR_WEB / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/pipeline/run", tags=["Pipeline"])
def pipeline_run(request: PipelineRequest):
    """Run every module (1-9) for a registered survey."""
    from app.pipeline import run_pipeline, stage_statuses

    try:
        result = run_pipeline(request.survey_id, request.detector, request.save_gradcam,
                              request.simulate, request.write_report)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    result["stage_statuses"] = stage_statuses(result)
    return result


@app.get("/reports/{survey_id}", tags=["Pipeline"])
def survey_report(survey_id: str):
    """Serve the HTML report written by the pipeline."""
    path = (REPO_ROOT / "data" / "outputs" / survey_id / "report.html").resolve()
    outputs = (REPO_ROOT / "data" / "outputs").resolve()
    if outputs not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found — run the pipeline first")
    return FileResponse(path, media_type="text/html")
