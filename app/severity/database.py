"""
FILE PATH: SmartSpray/app/severity/database.py

PURPOSE:
    The data-access layer for Module 4 — Disease Severity Estimation.

    This file is the ONLY place in Module 4 that talks to SQLite. It follows
    the same per-module ownership pattern as Modules 2 and 3:
        - Module 2 owns `detections`
        - Module 3 owns `classifications`
        - Module 4 owns `plant_severity`

    Responsibilities:
        - Ensure the plant_severity table exists (idempotent).
        - Persist one severity result per (survey_id, plant_id).
        - Retrieve saved severity results for a survey.
        - Retrieve joined detection + classification rows that Module 4 needs
          to reconstruct a plant crop (bbox) and know its disease label.

    This module contains NO image processing, NO ML, NO FastAPI, and NO
    business logic. It exposes SQLite helper functions only.

DEPENDENCIES:
    - app.db: shared SQLite connection factory
    - typing: standard library type hints
"""

import logging
import sqlite3
from typing import Any

from app.db import get_connection

# Named logger for this module.
logger = logging.getLogger("severity.database")

# Table name owned by this module.
SEVERITY_TABLE = "plant_severity"
def ensure_severity_table(conn: sqlite3.Connection) -> None:
    """
    Create the plant_severity table if it does not already exist.

    Idempotent: safe to call on every startup and inside every survey-level
    operation. Uses CREATE TABLE IF NOT EXISTS so existing data is never
    destroyed.

    Schema matches database/schema.sql exactly (Decision 2).

    Args:
        conn: open SQLite connection.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SEVERITY_TABLE} (
            severity_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id         TEXT    NOT NULL,
            plant_id          INTEGER NOT NULL,
            image_filename    TEXT    NOT NULL,
            disease_label     TEXT    NOT NULL,
            severity_score    REAL    NOT NULL,
            severity_category TEXT    NOT NULL,
            leaf_px           INTEGER NOT NULL,
            lesion_px         INTEGER NOT NULL,
            is_valid          INTEGER NOT NULL DEFAULT 1,
            model_version     TEXT    NOT NULL,
            estimated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE (survey_id, plant_id)
        )
        """
    )
    conn.commit()
def save_severity(
    conn: sqlite3.Connection,
    survey_id: str,
    plant_id: int,
    severity_score: float,
    severity_category: str,
    leaf_px: int,
    lesion_px: int,
    is_valid: bool,
    model_version: str,
) -> None:
    """
    Insert or replace one severity result for a single plant.

    Uses INSERT OR REPLACE keyed on (survey_id, plant_id), so re-running
    severity estimation for a survey overwrites previous rows (no duplicate
    plant rows). image_filename and disease_label are resolved from the
    detections/classifications tables so the stored row stays consistent
    with the survey's detection and classification data.

    Args:
        conn: open SQLite connection.
        survey_id: The survey this plant belongs to.
        plant_id: The plant identifier (from Module 2's detections).
        severity_score: Measured severity in [0.0, 1.0].
        severity_category: "low" | "medium" | "high".
        leaf_px: Number of leaf pixels in the crop.
        lesion_px: Number of lesion pixels in the leaf area.
        is_valid: Whether the estimate is considered valid.
        model_version: Version string of the estimator.
    """
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {SEVERITY_TABLE} (
            survey_id,
            plant_id,
            image_filename,
            disease_label,
            severity_score,
            severity_category,
            leaf_px,
            lesion_px,
            is_valid,
            model_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            survey_id,
            plant_id,
            _resolve_image_filename(conn, survey_id, plant_id),
            _resolve_disease_label(conn, survey_id, plant_id),
            severity_score,
            severity_category,
            leaf_px,
            lesion_px,
            int(bool(is_valid)),
            model_version,
        ),
    )
    conn.commit()
    logger.debug(
        f"Saved severity for {survey_id}/plant {plant_id}: "
        f"score={severity_score:.4f} category={severity_category} "
        f"(leaf={leaf_px}, lesion={lesion_px}, valid={is_valid})"
    )
def get_severity_for_survey(
    conn: sqlite3.Connection,
    survey_id: str,
) -> list[dict[str, Any]]:
    """
    Return all saved severity results for a survey, ordered by plant_id.

    Args:
        conn: open SQLite connection.
        survey_id: The survey to fetch.

    Returns:
        List of row dicts with keys:
          severity_id, survey_id, plant_id, image_filename, disease_label,
          severity_score, severity_category, leaf_px, lesion_px, is_valid,
          model_version, estimated_at.
        Empty list if the survey has no severity rows.
    """
    cursor = conn.execute(
        f"""
        SELECT severity_id,
               survey_id,
               plant_id,
               image_filename,
               disease_label,
               severity_score,
               severity_category,
               leaf_px,
               lesion_px,
               is_valid,
               model_version,
               estimated_at
        FROM {SEVERITY_TABLE}
        WHERE survey_id = ?
        ORDER BY plant_id ASC
        """,
        (survey_id,),
    )
    rows = cursor.fetchall()
    return [
        {
            "severity_id": r[0],
            "survey_id": r[1],
            "plant_id": r[2],
            "image_filename": r[3],
            "disease_label": r[4],
            "severity_score": r[5],
            "severity_category": r[6],
            "leaf_px": r[7],
            "lesion_px": r[8],
            "is_valid": bool(r[9]),
            "model_version": r[10],
            "estimated_at": r[11],
        }
        for r in rows
    ]
def get_detections_for_severity(
    conn: sqlite3.Connection,
    survey_id: str,
) -> list[dict[str, Any]]:
    """
    Return joined detection + classification rows needed by severity estimation.

    Joins `detections` (Module 2) with `classifications` (Module 3) on
    (survey_id, plant_id), so each row carries the plant's bounding box and
    its disease label/confidence. This is the authoritative feed for the
    service layer.

    Args:
        conn: open SQLite connection.
        survey_id: The survey to fetch.



    Returns:
        list of dicts with keys:
            plant_id, image_filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            disease_label, classification_confidence..
        Empty list if the survey has no classified detections..
    """



    cursor = conn.execute(
        """
        SELECT d.plant_id,
               d.image_filename,
               d.bbox_x1,
               d.bbox_y1,
               d.bbox_x2,
               d.bbox_y2,
               c.label       AS disease_label,
               c.confidence  AS classification_confidence
        FROM   detections d
        JOIN   classifications c
               ON  d.survey_id = c.survey_id
               AND d.plant_id  = c.plant_id
        WHERE  d.survey_id = ?
        ORDER  BY d.plant_id ASC
        """,
        (survey_id,),
    )
    rows = cursor.fetchall()
    return [
        {
            "plant_id": r[0],
            "image_filename": r[1],
            "bbox_x1": r[2],
            "bbox_y1": r[3],
            "bbox_x2": r[4],
            "bbox_y2": r[5],
            "disease_label": r[6],
            "classification_confidence": r[7],
        }
        for r in rows
    ]
# ── Internal helpers (kept private) ───────────────────────────────────────────

def _resolve_image_filename(
    conn: sqlite3.Connection,
    survey_id: str,
    plant_id: int,
) -> str:
    """
    Look up the source image filename for a survey/plant from detections.
    Used by save_severity to populate image_filename consistently.
    """
    row = conn.execute(
        """
        SELECT image_filename FROM detections
        WHERE survey_id = ? AND plant_id = ?
        """,
        (survey_id, plant_id),
    ).fetchone()
    return row[0] if row else ""


def _resolve_disease_label(
    conn: sqlite3.Connection,
    survey_id: str,
    plant_id: int,
) -> str:
    """
    Look up the disease label for a survey/plant from classifications.
    Used by save_severity to populate disease_label consistently..
    """
    row = conn.execute(
        """
        SELECT label FROM classifications
        WHERE survey_id = ? AND plant_id = ?
        """,
        (survey_id, plant_id),
    ).fetchone()
    return row[0] if row else "unknown"