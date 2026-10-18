"""
app/field_survey/router.py

The HTTP API for Module 1:
    POST /survey/register      validate + register a survey
    GET  /survey/{survey_id}   read a registered survey
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.field_survey import service

router = APIRouter(tags=["Module 1 — Field Survey"])


class SurveyRegisterRequest(BaseModel):
    """If someone sends {"survey_id": 5} (a number, not text) or leaves
    survey_id out entirely, FastAPI rejects it automatically (HTTP 422)."""
    survey_id: str
    crop: str = service.CROP
    field_width_m: float = Field(service.DEFAULT_FIELD_WIDTH_M, gt=0)
    field_height_m: float = Field(service.DEFAULT_FIELD_HEIGHT_M, gt=0)


@router.post("/survey/register")
def register_survey(request: SurveyRegisterRequest):
    try:
        return service.register_survey(
            request.survey_id,
            crop=request.crop,
            field_width_m=request.field_width_m,
            field_height_m=request.field_height_m,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/survey/{survey_id}")
def read_survey(survey_id: str):
    survey = service.get_survey(survey_id)
    if survey is None:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found")
    return survey
