"""
FILE PATH: SmartSpray/app/spatial_mapping/router.py

PURPOSE:
    Exposes Module 5 (Spatial Mapping) over HTTP.

    Endpoints:
        POST /spatial/calibrate          - ArUco calibration for one survey image
        POST /spatial/map                - map every plant in a survey to metres
        GET  /spatial/locations/{sid}    - read the stored plant_locations rows
        GET  /spatial/health             - module readiness

DEPENDENCIES:
    - fastapi: APIRouter, HTTPException
    - pydantic: request/response validation
    - app.spatial_mapping.calibration / mapper / database
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.spatial_mapping.mapper as spatial_mapper
from app.spatial_mapping.calibration import (
    DEFAULT_ARUCO_DICT,
    FIELD_CORNER_IDS,
    get_calibrator,
)
from app.spatial_mapping.database import (
    get_connection,
    get_locations_for_survey,
    save_calibration,
)

# Where survey images live (same convention as Modules 2, 3 and 4).
IMAGES_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "images"

router = APIRouter(
    prefix="/spatial",
    tags=["Module 5 - Spatial Mapping"],
)


class SpatialModel(BaseModel):
    """Base model for Module 5 pydantic models.

    Suppresses the pydantic ``model_version`` protected-namespace warning, so
    importing this router does not emit a UserWarning at collection time (the
    plant_locations column ``model_version`` clashes with pydantic's default
    ``model_`` protected namespace). Same fix as Module 4's SeverityModel.
    """

    model_config = {"protected_namespaces": ()}


class CalibrateRequest(SpatialModel):
    """ArUco calibration request for a single survey image."""

    survey_id: str
    image_filename: str
    field_width_m: float
    field_height_m: float
    aruco_dict: str = DEFAULT_ARUCO_DICT


class CalibrateResponse(SpatialModel):
    """Result of a successful ArUco calibration."""

    status: str
    calibration_id: str
    survey_id: str
    image_filename: str
    field_width_m: float
    field_height_m: float
    aruco_dict: str
    marker_count: int
    reprojection_error: float | None
    is_valid: bool
    notes: list[str]
    homography: list[list[float]] | None


class MapRequest(SpatialModel):
    """Spatial mapping request for a whole survey."""

    survey_id: str
    calibration_id: str | None = None

class LocationItem(SpatialModel):
    """One plant's stored location. field_x_m/field_y_m are None when the
    survey is uncalibrated (is_mapped False) - coordinates are never faked."""

    plant_id: int
    image_filename: str
    bbox_cx_px: float
    bbox_cy_px: float
    field_x_m: float | None
    field_y_m: float | None
    is_mapped: bool
    calibration_id: str | None
    mapping_mode: str
    model_version: str


class MapResponse(SpatialModel):
    """Result of mapping a whole survey."""

    status: str
    survey_id: str
    mapping_mode: str
    calibration_id: str | None
    total_plants: int
    mapped_plants: int
    unmapped_plants: int
    results: list[LocationItem]


class LocationsResponse(SpatialModel):
    """Stored plant_locations rows for a survey."""

    status: str
    survey_id: str
    total_locations: int
    locations: list[LocationItem]


class HealthResponse(SpatialModel):
    """Module 5 readiness."""

    module: str
    calibrator_ready: bool
    aruco_dict: str | None
    min_markers: int | None
    field_corner_ids: list[int]
    status: str
    note: str


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate(request: CalibrateRequest):
    """
    Calibrate a survey from one image containing the four corner ArUco markers.

    Detects the markers, solves the pixel->field homography and persists it to
    spatial_calibrations. Returns 400 when the markers are missing/unusable or
    the solved calibration breaches its reprojection-error tolerance.
    """
    survey_id = (request.survey_id or "").strip()
    image_filename = (request.image_filename or "").strip()

    if not survey_id:
        raise HTTPException(status_code=400, detail="survey_id must be a non-empty string")
    if not image_filename:
        raise HTTPException(status_code=400, detail="image_filename must be a non-empty string")
    if request.field_width_m <= 0 or request.field_height_m <= 0:
        raise HTTPException(
            status_code=400,
            detail="field_width_m and field_height_m must both be positive metres",
        )

    image_path = IMAGES_ROOT / survey_id / image_filename

    try:
        result = get_calibrator().calibrate_from_file(
            survey_id=survey_id,
            image_filename=image_filename,
            image_path=image_path,
            field_width_m=request.field_width_m,
            field_height_m=request.field_height_m,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not result.is_valid:
        raise HTTPException(
            status_code=400,
            detail=(
                "Calibration reprojection error %.3f px exceeds tolerance; "
                "refusing to store an inaccurate calibration. %s"
                % (result.reprojection_error, " ".join(result.notes))
            ).strip(),
        )

    conn = get_connection()
    try:
        save_calibration(
            conn=conn,
            calibration_id=result.calibration_id,
            survey_id=result.survey_id,
            image_filename=result.image_filename,
            field_width_m=result.field_width_m,
            field_height_m=result.field_height_m,
            aruco_dict=result.aruco_dict,
            marker_count=result.marker_count,
            homography_json=result.homography_json,
            reprojection_error=result.reprojection_error,
            is_valid=1 if result.is_valid else 0,
        )
    finally:
        conn.close()

    payload = result.as_dict()
    return CalibrateResponse(status="success", **payload)

@router.post("/map", response_model=MapResponse)
def map_survey_endpoint(request: MapRequest):
    """
    Map every detected plant in a survey from pixels to field metres.

    Uses the explicitly supplied calibration_id when given, otherwise the most
    recent VALID calibration for the survey. When no valid calibration exists
    the survey is mapped in 'uncalibrated' mode: rows are still written, but
    field_x_m/field_y_m stay NULL and is_mapped is False.
    """
    survey_id = (request.survey_id or "").strip()
    if not survey_id:
        raise HTTPException(status_code=400, detail="survey_id must be a non-empty string")

    calibration_id = (request.calibration_id or "").strip() or None

    try:
        records = spatial_mapper.map_survey(survey_id, calibration_id=calibration_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    summary = spatial_mapper.summarise(records)
    mapping_mode = (
        records[0]["mapping_mode"]
        if records
        else spatial_mapper.MAPPING_MODE_UNCALIBRATED
    )
    active_calibration_id = records[0]["calibration_id"] if records else None

    return MapResponse(
        status="success",
        survey_id=survey_id,
        mapping_mode=mapping_mode,
        calibration_id=active_calibration_id,
        total_plants=summary["total_plants"],
        mapped_plants=summary["mapped_plants"],
        unmapped_plants=summary["unmapped_plants"],
        results=[LocationItem(**record) for record in records],
    )


@router.get("/locations/{survey_id}", response_model=LocationsResponse)
def get_locations(survey_id: str):
    """Return the stored plant_locations rows for a survey (possibly empty)."""
    survey_id = (survey_id or "").strip()
    if not survey_id:
        raise HTTPException(status_code=400, detail="survey_id must be a non-empty string")

    conn = get_connection()
    try:
        rows = get_locations_for_survey(conn, survey_id)
    finally:
        conn.close()

    return LocationsResponse(
        status="success",
        survey_id=survey_id,
        total_locations=len(rows),
        locations=[LocationItem(**row) for row in rows],
    )


@router.get("/health", response_model=HealthResponse)
def spatial_health():
    """Health check for the spatial mapping module."""
    try:
        calibrator = get_calibrator()
        ready = True
        aruco_dict = calibrator.aruco_dict
        min_markers = calibrator.min_markers
    except Exception:
        ready = False
        aruco_dict = None
        min_markers = None

    return HealthResponse(
        module="Spatial Mapping",
        calibrator_ready=ready,
        aruco_dict=aruco_dict,
        min_markers=min_markers,
        field_corner_ids=list(FIELD_CORNER_IDS),
        status="ready" if ready else "calibrator_unavailable",
        note=(
            "ArUco marker homography (pixel -> field metres). Surveys without a "
            "valid calibration are stored as uncalibrated with NULL field "
            "coordinates; positions are never fabricated."
        ),
    )
