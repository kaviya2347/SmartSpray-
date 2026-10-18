"""
app/spatial_mapping/__init__.py

Module 5 — Spatial Mapping.

Converts pixel-space plant bounding boxes to real-world field coordinates
using ArUco marker-based homography.
"""

from app.spatial_mapping.database import (
    ensure_spatial_tables,
    save_location,
    get_locations_for_survey,
    save_calibration,
    get_calibration,
    get_latest_calibration_for_survey,
    get_detections_for_mapping,
)
from app.spatial_mapping.calibration import (
    ArUcoCalibrator,
    CalibrationResult,
    MarkerDetection,
    build_field_corner_points,
    get_calibrator,
    homography_from_json,
    make_calibration_id,
)
from app.spatial_mapping.mapper import (
    MAPPING_MODE_CALIBRATED,
    MAPPING_MODE_SIMULATION,
    MAPPING_MODE_UNCALIBRATED,
    MODEL_VERSION,
    bbox_center_px,
    map_survey,
    pixel_to_field,
    summarise,
)

__all__ = [
    # database
    "ensure_spatial_tables",
    "save_location",
    "get_locations_for_survey",
    "save_calibration",
    "get_calibration",
    "get_latest_calibration_for_survey",
    "get_detections_for_mapping",
    # calibration
    "ArUcoCalibrator",
    "CalibrationResult",
    "MarkerDetection",
    "build_field_corner_points",
    "get_calibrator",
    "homography_from_json",
    "make_calibration_id",
    # mapper
    "MAPPING_MODE_CALIBRATED",
    "MAPPING_MODE_SIMULATION",
    "MAPPING_MODE_UNCALIBRATED",
    "MODEL_VERSION",
    "bbox_center_px",
    "map_survey",
    "pixel_to_field",
    "summarise",
]
