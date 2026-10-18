"""
FILE PATH: SmartSpray/app/spatial_mapping/mapper.py

PURPOSE:
    The coordinate-mapping engine for Module 5 - Spatial Mapping.

    It reads per-plant bounding boxes from Module 2 (detections, LEFT JOINed
    with Module 4's plant_severity), converts each bounding-box CENTRE from
    pixels to field metres using a stored homography, and persists the result
    into plant_locations.

    Two explicit modes, never mixed and never guessed:
        "calibrated"   - a VALID calibration exists for the survey. field_x_m
                         and field_y_m are computed with H and is_mapped = 1.
        "uncalibrated" - no valid calibration exists. field_x_m and field_y_m
                         are stored as NULL and is_mapped = 0. Coordinates are
                         NOT invented, interpolated or approximated.
        "simulation"   - reserved for a future offline/synthetic coordinate
                         source. Nothing in this module produces it today.

    This follows the project rule that measurements are reported, never
    fabricated. The repository's current survey images contain no ArUco
    markers (verified during the Module 5 inspection), so every survey maps in
    "uncalibrated" mode until corner markers are printed and a calibration
    image is captured.

DEPENDENCIES:
    - cv2 / numpy: projective transform
    - app.spatial_mapping.database: all persistence helpers
    - app.spatial_mapping.calibration: homography deserialisation
    - logging / typing (standard library)
"""

import logging

import cv2
import numpy as np

from app.spatial_mapping.calibration import homography_from_json
from app.spatial_mapping.database import (
    get_calibration,
    get_connection,
    get_detections_for_mapping,
    get_latest_calibration_for_survey,
    save_location,
)

# Named logger for this module.
logger = logging.getLogger("spatial_mapping.mapper")

# Version stamped onto every plant_locations row. Bump when the mapping
# convention (bbox centre, marker layout, mode names) changes.
MODEL_VERSION = "spatial-mapping-v1"

# Mapping modes recorded in plant_locations.mapping_mode.
MAPPING_MODE_CALIBRATED = "calibrated"
MAPPING_MODE_UNCALIBRATED = "uncalibrated"
MAPPING_MODE_SIMULATION = "simulation"

# ── Coordinate primitives ─────────────────────────────────────────────────────

def bbox_center_px(
    bbox_x1: float,
    bbox_y1: float,
    bbox_x2: float,
    bbox_y2: float,
) -> tuple:
    """
    Return the geometric centre of a pixel bounding box as (cx_px, cy_px).

    Module 5 deliberately maps the bounding-box CENTRE (a single, stable point)
    rather than a corner or the leaf centroid: the centre is already stored in
    plant_locations as bbox_cx_px / bbox_cy_px and does not depend on the
    severity mask being valid.
    """
    return (
        (float(bbox_x1) + float(bbox_x2)) / 2.0,
        (float(bbox_y1) + float(bbox_y2)) / 2.0,
    )


def pixel_to_field(
    homography: np.ndarray,
    x_px: float,
    y_px: float,
) -> tuple:
    """
    Apply a pixel -> field homography to a single point.

    Args:
        homography: 3x3 pixel->field matrix.
        x_px, y_px: pixel coordinates.

    Returns:
        (field_x_m, field_y_m).

    Raises:
        ValueError: if the homography is not a finite 3x3 matrix, or the point
            projects to a non-finite (point-at-infinity) location.
    """
    matrix = _validated_homography(homography)
    point = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
    projected = cv2.perspectiveTransform(point, matrix)
    field_x = float(projected[0, 0, 0])
    field_y = float(projected[0, 0, 1])
    if not (np.isfinite(field_x) and np.isfinite(field_y)):
        raise ValueError(
            "Pixel (%r, %r) projects to a non-finite field position; the "
            "homography is degenerate." % (x_px, y_px)
        )
    return (field_x, field_y)


def _validated_homography(homography: np.ndarray) -> np.ndarray:
    """Return the homography as a finite (3, 3) float64 array.

    Raises:
        ValueError: if the shape is wrong or any element is NaN/inf.
    """
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            "homography must be a 3x3 matrix, got shape %s" % (matrix.shape,)
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("homography contains non-finite values")
    return matrix


# ── Calibration resolution ────────────────────────────────────────────────────

def resolve_calibration(
    conn,
    survey_id: str,
    calibration_id: str | None = None,
) -> dict:
    """
    Decide which calibration (if any) applies to a survey.

    Args:
        conn: open SQLite connection.
        survey_id: the survey being mapped.
        calibration_id: explicit calibration to use, or None to auto-select the
            most recent valid calibration for the survey.

    Returns:
        The calibration row dict, or None when the survey is uncalibrated.

    Raises:
        ValueError: if an explicit calibration_id does not exist, belongs to a
            different survey, or is flagged invalid. An explicit request is
            never silently downgraded to uncalibrated mode.
    """
    if calibration_id:
        calibration = get_calibration(conn, calibration_id)
        if calibration is None:
            raise ValueError(
                "Calibration '%s' was not found" % calibration_id
            )
        if calibration["survey_id"] != survey_id:
            raise ValueError(
                "Calibration '%s' belongs to survey '%s', not '%s'"
                % (calibration_id, calibration["survey_id"], survey_id)
            )
        if not calibration["is_valid"]:
            raise ValueError(
                "Calibration '%s' is flagged invalid and cannot be used"
                % calibration_id
            )
        return calibration

    return get_latest_calibration_for_survey(conn, survey_id)

# ── Survey-level mapping ──────────────────────────────────────────────────────

def map_survey(survey_id: str, calibration_id: str | None = None) -> list:
    """
    Map every detected plant in a survey to field coordinates and persist them.

    Steps:
        1. Read detections (Module 2) LEFT JOINed with plant_severity (M4).
        2. Resolve the calibration to use (explicit id, else latest valid one).
        3. Convert each bounding-box centre from pixels to metres via H, or
           record the plant as unmapped when no calibration is available.
        4. Persist one plant_locations row per plant (UNIQUE on
           survey_id + plant_id, so re-running overwrites rather than duplicating).

    Args:
        survey_id: the survey to map.
        calibration_id: optional explicit calibration to use.

    Returns:
        list of per-plant dicts, each with plant_id, image_filename,
        bbox_cx_px, bbox_cy_px, field_x_m, field_y_m, is_mapped,
        calibration_id, mapping_mode, model_version.

    Raises:
        ValueError: if survey_id is empty, the survey has no detections, or an
            explicitly requested calibration is unusable.
    """
    if not isinstance(survey_id, str) or not survey_id.strip():
        raise ValueError("survey_id must be a non-empty string")
    survey_id = survey_id.strip()

    conn = get_connection()
    try:
        detections = get_detections_for_mapping(conn, survey_id)
        if not detections:
            raise ValueError(
                "No detections found for survey_id='%s'; run Module 2 "
                "(detection) before spatial mapping." % survey_id
            )

        logger.info(
            "Starting spatial mapping for survey '%s': %d plants (querying "
            "calibration)",
            survey_id,
            len(detections),
        )

        calibration = resolve_calibration(conn, survey_id, calibration_id)

        if calibration is not None:
            homography = homography_from_json(calibration["homography_json"])
            mapping_mode = MAPPING_MODE_CALIBRATED
            active_calibration_id = calibration["calibration_id"]
            field_width_m = float(calibration["field_width_m"])
            field_height_m = float(calibration["field_height_m"])
            logger.info(
                "Survey '%s' is calibrated (calibration_id=%s); converting "
                "pixels to metres.",
                survey_id,
                active_calibration_id,
            )
        else:
            homography = None
            mapping_mode = MAPPING_MODE_UNCALIBRATED
            active_calibration_id = None
            field_width_m = None
            field_height_m = None
            logger.warning(
                "Survey '%s' has no valid ArUco calibration. Storing NULL "
                "field coordinates (mapping_mode='%s') instead of inventing "
                "positions.",
                survey_id,
                mapping_mode,
            )

        records = []
        outside_bounds = 0

        for detection in detections:
            plant_id = int(detection["plant_id"])
            cx_px = float(detection["bbox_cx_px"])
            cy_px = float(detection["bbox_cy_px"])

            if homography is not None:
                field_x_m, field_y_m = pixel_to_field(homography, cx_px, cy_px)
                is_mapped = 1
                if not (
                    0.0 <= field_x_m <= field_width_m
                    and 0.0 <= field_y_m <= field_height_m
                ):
                    outside_bounds += 1
                    logger.warning(
                        "plant_id=%s maps outside the declared field "
                        "(%.3f, %.3f) m; keeping the measured value.",
                        plant_id,
                        field_x_m,
                        field_y_m,
                    )
            else:
                field_x_m = None
                field_y_m = None
                is_mapped = 0

            save_location(
                conn=conn,
                survey_id=survey_id,
                plant_id=plant_id,
                image_filename=detection["image_filename"],
                bbox_cx_px=cx_px,
                bbox_cy_px=cy_px,
                field_x_m=field_x_m,
                field_y_m=field_y_m,
                is_mapped=is_mapped,
                calibration_id=active_calibration_id,
                mapping_mode=mapping_mode,
                model_version=MODEL_VERSION,
            )

            records.append({
                "plant_id": plant_id,
                "image_filename": detection["image_filename"],
                "bbox_cx_px": cx_px,
                "bbox_cy_px": cy_px,
                "field_x_m": field_x_m,
                "field_y_m": field_y_m,
                "is_mapped": is_mapped,
                "calibration_id": active_calibration_id,
                "mapping_mode": mapping_mode,
                "model_version": MODEL_VERSION,
            })

        conn.commit()

        summary = summarise(records)
        logger.info(
            "Spatial mapping complete for '%s': %d mapped, %d unmapped, "
            "%d outside declared bounds, mode=%s",
            survey_id,
            summary["mapped_plants"],
            summary["unmapped_plants"],
            outside_bounds,
            mapping_mode,
        )
        return records
    finally:
        conn.close()


def summarise(records: list) -> dict:
    """Count mapped vs unmapped records. Shared by the router and pipeline."""
    mapped = sum(1 for record in records if record["is_mapped"])
    return {
        "total_plants": len(records),
        "mapped_plants": mapped,
        "unmapped_plants": len(records) - mapped,
    }
