"""app/spatial_mapping/database.py

Data-access layer for Module 5 - Spatial Mapping.

Provides:
  - ensure_spatial_tables(): create tables if not exist
  - save_location(): persist per-plant field coordinates
  - get_locations_for_survey(): retrieve locations for a survey
  - save_calibration(): persist ArUco homography calibration
  - get_calibration(): retrieve calibration by ID
  - get_detections_for_mapping(): JOIN detections with plant_severity

This module contains NO ArUco detection, NO homography computation,
and NO coordinate transformation. Those belong in calibration.py and mapper.py.
"""

import json
import sqlite3

from app.db import get_connection


LOCATIONS_TABLE = "plant_locations"
CALIBRATIONS_TABLE = "spatial_calibrations"


def ensure_spatial_tables(conn: sqlite3.Connection) -> None:
    """Create plant_locations and spatial_calibrations tables if not exist."""
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {LOCATIONS_TABLE} (
        location_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        survey_id     TEXT    NOT NULL,
        plant_id      INTEGER NOT NULL,
        image_filename TEXT   NOT NULL,
        bbox_cx_px    REAL    NOT NULL,
        bbox_cy_px    REAL    NOT NULL,
        field_x_m     REAL,
        field_y_m     REAL,
        is_mapped     INTEGER NOT NULL DEFAULT 0,
        calibration_id TEXT,
        mapping_mode  TEXT    NOT NULL DEFAULT 'uncalibrated',
        model_version TEXT    NOT NULL,
        mapped_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (survey_id, plant_id)
    )""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {CALIBRATIONS_TABLE} (
        calibration_id    TEXT    PRIMARY KEY,
        survey_id         TEXT    NOT NULL,
        image_filename    TEXT    NOT NULL,
        field_width_m     REAL    NOT NULL,
        field_height_m    REAL    NOT NULL,
        aruco_dict        TEXT    NOT NULL,
        marker_count      INTEGER NOT NULL,
        homography_json   TEXT    NOT NULL,
        reprojection_error REAL,
        is_valid          INTEGER NOT NULL DEFAULT 1,
        calibrated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.commit()


def save_location(
    conn: sqlite3.Connection,
    survey_id: str,
    plant_id: int,
    image_filename: str,
    bbox_cx_px: float,
    bbox_cy_px: float,
    field_x_m: float | None,
    field_y_m: float | None,
    is_mapped: int,
    calibration_id: str | None,
    mapping_mode: str,
    model_version: str,
) -> None:
    """INSERT OR REPLACE into plant_locations."""
    conn.execute(f"""INSERT OR REPLACE INTO {LOCATIONS_TABLE} (
        survey_id, plant_id, image_filename,
        bbox_cx_px, bbox_cy_px,
        field_x_m, field_y_m,
        is_mapped, calibration_id, mapping_mode, model_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (survey_id, plant_id, image_filename, bbox_cx_px, bbox_cy_px,
         field_x_m, field_y_m, is_mapped, calibration_id, mapping_mode, model_version))
    conn.commit()


def get_locations_for_survey(conn: sqlite3.Connection, survey_id: str) -> list[dict]:
    """Return all location rows for a survey, ordered by plant_id."""
    cols = ["location_id", "survey_id", "plant_id", "image_filename",
            "bbox_cx_px", "bbox_cy_px", "field_x_m", "field_y_m",
            "is_mapped", "calibration_id", "mapping_mode", "model_version", "mapped_at"]
    rows = conn.execute(
        f"SELECT * FROM {LOCATIONS_TABLE} WHERE survey_id=? ORDER BY plant_id",
        (survey_id,)).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def save_calibration(
    conn: sqlite3.Connection,
    calibration_id: str,
    survey_id: str,
    image_filename: str,
    field_width_m: float,
    field_height_m: float,
    aruco_dict: str,
    marker_count: int,
    homography_json: str,
    reprojection_error: float | None,
    is_valid: int,
) -> None:
    """INSERT OR REPLACE into spatial_calibrations."""
    conn.execute(f"""INSERT OR REPLACE INTO {CALIBRATIONS_TABLE} (
        calibration_id, survey_id, image_filename,
        field_width_m, field_height_m,
        aruco_dict, marker_count,
        homography_json, reprojection_error, is_valid,
        calibrated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (calibration_id, survey_id, image_filename,
         field_width_m, field_height_m, aruco_dict,
         marker_count, homography_json,
         reprojection_error, is_valid))
    conn.commit()


def get_calibration(conn: sqlite3.Connection, calibration_id: str) -> dict | None:
    """Return calibration row by ID, or None if not found."""
    cols = ["calibration_id", "survey_id", "image_filename",
            "field_width_m", "field_height_m", "aruco_dict",
            "marker_count", "homography_json", "reprojection_error",
            "is_valid", "calibrated_at"]
    rows = conn.execute(
        f"SELECT * FROM {CALIBRATIONS_TABLE} WHERE calibration_id=?",
        (calibration_id,)).fetchall()
    if not rows:
        return None
    return dict(zip(cols, rows[0]))


def get_latest_calibration_for_survey(
    conn: sqlite3.Connection,
    survey_id: str,
) -> dict | None:
    """Return the most recent VALID calibration for a survey, or None.

    Only rows with is_valid = 1 are considered, so a calibration that breached
    its reprojection-error tolerance is never used to place plants. Ties are
    broken by calibration_id so the result is deterministic.
    """
    cols = ["calibration_id", "survey_id", "image_filename",
            "field_width_m", "field_height_m", "aruco_dict",
            "marker_count", "homography_json", "reprojection_error",
            "is_valid", "calibrated_at"]
    rows = conn.execute(
        f"SELECT * FROM {CALIBRATIONS_TABLE} "
        "WHERE survey_id = ? AND is_valid = 1 "
        "ORDER BY calibrated_at DESC, calibration_id DESC",
        (survey_id,)).fetchall()
    if not rows:
        return None
    return dict(zip(cols, rows[0]))


def get_detections_for_mapping(conn: sqlite3.Connection, survey_id: str) -> list[dict]:
    """JOIN detections with plant_severity for this survey.

    Returns per-plant data with computed bbox center.
    """
    rows = conn.execute(
        f"""SELECT d.plant_id, d.image_filename,
               d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
               s.severity_score, s.severity_category, s.is_valid
        FROM detections d
        LEFT JOIN plant_severity s
            ON d.survey_id = s.survey_id AND d.plant_id = s.plant_id
        WHERE d.survey_id = ?
        ORDER BY d.plant_id""",
        (survey_id,)).fetchall()
    out = []
    for row in rows:
        x1, y1, x2, y2 = row[2], row[3], row[4], row[5]
        out.append({
            "plant_id": row[0], "image_filename": row[1],
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "bbox_cx_px": (x1 + x2) / 2.0, "bbox_cy_px": (y1 + y2) / 2.0,
            "severity_score": row[6], "severity_category": row[7],
            "is_valid": row[8]})
    return out
