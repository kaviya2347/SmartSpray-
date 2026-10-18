"""
app/classification/database.py

Database helpers for Module 3 — Disease Classification.
"""

import json
import sqlite3

from app.classification.prediction import ClassificationResult


def ensure_classifications_table(conn: sqlite3.Connection) -> None:
    """
    Create the classifications table if it does not already exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id         TEXT    NOT NULL,
            image_filename    TEXT    NOT NULL,
            plant_id          INTEGER NOT NULL,
            bbox_x1           REAL    NOT NULL,
            bbox_y1           REAL    NOT NULL,
            bbox_x2           REAL    NOT NULL,
            bbox_y2           REAL    NOT NULL,
            label             TEXT    NOT NULL,
            confidence        REAL    NOT NULL,
            all_probs_json    TEXT    NOT NULL,
            model_version     TEXT    NOT NULL,
            is_uncertain      INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT    DEFAULT (datetime('now')),
            UNIQUE(survey_id, plant_id)
        )
        """
    )
    # Lightweight migration for databases created before is_uncertain existed.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(classifications)")}
    if "is_uncertain" not in columns:
        conn.execute(
            "ALTER TABLE classifications "
            "ADD COLUMN is_uncertain INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()


def fetch_detections_for_survey(
    conn: sqlite3.Connection,
    survey_id: str,
) -> list[dict]:
    """
    Fetch plant detections produced by Module 2 for a survey.
    """
    cursor = conn.execute(
        """
        SELECT image_filename,
               plant_id,
               bbox_x1,
               bbox_y1,
               bbox_x2,
               bbox_y2
        FROM detections
        WHERE survey_id = ?
        ORDER BY plant_id ASC
        """,
        (survey_id,),
    )

    rows = cursor.fetchall()

    return [
        {
            "image_filename": row[0],
            "plant_id": row[1],
            "bbox_x1": row[2],
            "bbox_y1": row[3],
            "bbox_x2": row[4],
            "bbox_y2": row[5],
        }
        for row in rows
    ]


def delete_classifications_for_survey(
    conn: sqlite3.Connection,
    survey_id: str,
) -> None:
    """
    Delete previously saved classifications for a survey.
    """
    conn.execute(
        """
        DELETE FROM classifications
        WHERE survey_id = ?
        """,
        (survey_id,),
    )
    conn.commit()


def save_classification(
    conn: sqlite3.Connection,
    result: ClassificationResult,
) -> None:
    """
    Save one ClassificationResult to the database.
    """
    all_probs_json = json.dumps(result.all_probs)

    conn.execute(
        """
        INSERT OR REPLACE INTO classifications (
            survey_id,
            image_filename,
            plant_id,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            label,
            confidence,
            all_probs_json,
            model_version,
            is_uncertain
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.survey_id,
            result.image_filename,
            result.plant_id,
            result.bbox[0],
            result.bbox[1],
            result.bbox[2],
            result.bbox[3],
            result.label,
            result.confidence,
            all_probs_json,
            result.model_version,
            int(bool(getattr(result, "uncertain", False))),
        ),
    )

    conn.commit()


def fetch_classifications_for_survey(
    conn: sqlite3.Connection,
    survey_id: str,
) -> list[dict]:
    """
    Fetch all saved classifications for a survey.
    """
    cursor = conn.execute(
        """
        SELECT survey_id,
               image_filename,
               plant_id,
               bbox_x1,
               bbox_y1,
               bbox_x2,
               bbox_y2,
               label,
               confidence,
               all_probs_json,
               model_version,
               created_at,
               is_uncertain
        FROM classifications
        WHERE survey_id = ?
        ORDER BY plant_id ASC
        """,
        (survey_id,),
    )

    rows = cursor.fetchall()

    return [
        {
            "survey_id": row[0],
            "image_filename": row[1],
            "plant_id": row[2],
            "bbox": [
                row[3],
                row[4],
                row[5],
                row[6],
            ],
            "label": row[7],
            "confidence": row[8],
            "all_probs": json.loads(row[9]),
            "model_version": row[10],
            "created_at": row[11],
            "uncertain": bool(row[12]),
        }
        for row in rows
    ]