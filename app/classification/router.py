"""
app/classification/router.py

API routes for Module 3 — Disease Classification.

    POST /classify                     classify every detected plant in a survey
    GET  /classify/results/{survey_id} saved classifications for a survey
    GET  /classify/health              model readiness + inference settings
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.classification.service as classification_service
from app.classification.classifier import get_classifier
from app.classification.database import (
    ensure_classifications_table,
    fetch_classifications_for_survey,
)
from app.classification.prediction import MODEL_VERSION
from app.db import get_connection


router = APIRouter(
    prefix="/classify",
    tags=["Module 3 — Disease Classification"],
)


class ClassifyRequest(BaseModel):
    survey_id: str
    save_gradcam: bool = False


@router.post("")
def classify(request: ClassifyRequest):
    """
    Run disease classification for all detected plants in a survey.
    Set `save_gradcam: true` to also write a Grad-CAM heatmap per plant.
    """
    kwargs = {"save_gradcam": True} if request.save_gradcam else {}
    try:
        return classification_service.classify_survey(request.survey_id, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/results/{survey_id}")
def classification_results(survey_id: str):
    """Return the classifications stored for a survey (possibly empty)."""
    conn = get_connection()
    try:
        ensure_classifications_table(conn)
        rows = fetch_classifications_for_survey(conn, survey_id)
    finally:
        conn.close()
    return {"survey_id": survey_id, "total": len(rows), "results": rows}


@router.get("/health")
def classification_health():
    """Health check for the disease classification module."""
    info = {
        "module": "Disease Classification",
        "status": "ok",
        "model_loaded": False,
        "model_version": MODEL_VERSION,
    }
    try:
        classifier = get_classifier()
        info.update(
            model_loaded=classifier.is_ready(),
            model_version=classifier.model_version,
            class_order=classifier.class_order,
            temperature=classifier.temperature,
            tta_hflip=classifier.tta,
            uncertainty_threshold=classifier.uncertainty_threshold,
        )
    except Exception as exc:
        info.update(status="model_unavailable", detail=str(exc))
    return info
