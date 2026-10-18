"""
app/severity/router.py

API routes for Module 4 — Disease Severity Estimation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.severity.service as severity_service
from app.severity.estimator import get_estimator
from app.severity.database import get_connection, save_severity

router = APIRouter(
    prefix="/severity",
    tags=["Module 4 — Disease Severity Estimation"],
)


class SeverityModel(BaseModel):
    """Base model for severity router pydantic models.

    Suppresses the pydantic ``model_version`` protected-namespace warning
    so that importing the router does not emit a UserWarning during test
    collection (the field name ``model_version`` clashes with pydantic's
    default ``model_`` protected namespace).
    """

    model_config = {"protected_namespaces": ()}


class SeverityRequest(SeverityModel):
    survey_id: str


class SeverityResultItem(SeverityModel):
    plant_id: int
    disease_label: str
    severity_score: float
    severity_category: str
    leaf_px: int
    lesion_px: int
    is_valid: bool
    model_version: str


class SeverityResponse(SeverityModel):
    status: str
    survey_id: str
    total_plants: int
    valid_estimates: int
    invalid_estimates: int
    results: list[SeverityResultItem]


@router.post("")
def estimate_severity(request: SeverityRequest):
    """
    Estimate disease severity for every classified plant in a survey.

    Runs the OpenCV severity estimator on each plant crop and persists
    the results to the plant_severity table.
    """
    if not request.survey_id or not request.survey_id.strip():
        raise HTTPException(
            status_code=400,
            detail="survey_id must be a non-empty string",
        )

    survey_id = request.survey_id.strip()

    try:
        results = severity_service.estimate_survey(survey_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    conn = get_connection()
    try:
        for result in results:
            save_severity(
                conn=conn,
                survey_id=result.survey_id,
                plant_id=result.plant_id,
                severity_score=result.severity_score,
                severity_category=result.severity_category,
                leaf_px=result.leaf_px,
                lesion_px=result.lesion_px,
                is_valid=result.is_valid,
                model_version=result.model_version,
            )
        conn.commit()
    finally:
        conn.close()

    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = sum(1 for r in results if not r.is_valid)

    response = SeverityResponse(
        status="success",
        survey_id=survey_id,
        total_plants=len(results),
        valid_estimates=valid_count,
        invalid_estimates=invalid_count,
        results=[
            SeverityResultItem(
                plant_id=r.plant_id,
                disease_label=r.disease_label,
                severity_score=r.severity_score,
                severity_category=r.severity_category,
                leaf_px=r.leaf_px,
                lesion_px=r.lesion_px,
                is_valid=r.is_valid,
                model_version=r.model_version,
            )
            for r in results
        ],
    )

    return response


class HealthResponse(SeverityModel):
    module: str
    estimator_ready: bool
    model_version: str | None
    bands: list[dict[str, str]]
    status: str
    note: str


@router.get("/health")
def severity_health():
    """
    Health check for the disease severity estimation module.
    """
    try:
        estimator = get_estimator()
        ready = True
        model_version = estimator.model_version
    except Exception:
        ready = False
        model_version = None

    bands = [
        {"low": "<0.20"},
        {"medium": "0.20-0.50"},
        {"high": ">=0.50"},
    ]

    return HealthResponse(
        module="Disease Severity Estimation",
        estimator_ready=ready,
        model_version=model_version,
        bands=bands,
        status="ready" if ready else "estimator_not_initialized",
        note="OpenCV heuristic estimator. No ground-truth severity calibration.",
    )

