"""
FILE PATH: SmartSpray/app/detection/router.py

PURPOSE:
    Exposes Module 2's functionality as an HTTP API endpoint.
    Receives a survey_id, processes every image in that survey's folder,
    and returns structured detection results.

    Also writes detections to the SQLite database so downstream modules
    can query them without needing to re-run the model.

INPUT (HTTP Request Body):
    {"survey_id": "survey_001"}

OUTPUT (HTTP Response):
    {
        "status": "success",
        "survey_id": "survey_001",
        "images_processed": 5,
        "plants_detected": 63,
        "detections": [
            {
                "plant_id": 1,
                "image": "IMG_001.jpg",
                "bbox": [120, 80, 260, 210],
                "confidence": 0.97
            },
            ...
        ]
    }

DEPENDENCIES:
    - fastapi: for APIRouter, HTTPException, and request/response handling
    - pydantic: for request body validation (DetectRequest)
    - app.db: for SQLite database connection
    - app.detection.detector: for PlantDetector inference engine
    - app.detection.identity_resolver: for assigning plant_ids
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db import get_connection
from app.detection.detector import MODEL_VERSION, get_detector
from app.detection.identity_resolver import resolve_survey_detections

logger = logging.getLogger("detection.router")

# ── FastAPI router ────────────────────────────────────────────────────────────
# This router is registered in app/main.py with include_router().
router = APIRouter(tags=["Module 2 — Plant Detection"])

# ── Image directory ───────────────────────────────────────────────────────────
# Module 1 places survey images here (anchored at the repository root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_ROOT = REPO_ROOT / "data" / "images"
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ── Database setup ────────────────────────────────────────────────────────────
def _ensure_detections_table(conn: sqlite3.Connection) -> None:
    """
    Create the detections table if it doesn't exist.

    This is called once per request. SQLite's "IF NOT EXISTS" makes it safe
    to call multiple times without creating duplicates or errors.

    We define the table here in router.py (rather than schema.sql) because:
    - Module 2 owns this table
    - Module 1's files must not be modified
    - Each module creates only the tables it needs
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id      TEXT    NOT NULL,
            image_filename TEXT    NOT NULL,
            plant_id       INTEGER NOT NULL,
            bbox_x1        REAL    NOT NULL,
            bbox_y1        REAL    NOT NULL,
            bbox_x2        REAL    NOT NULL,
            bbox_y2        REAL    NOT NULL,
            confidence     REAL    NOT NULL,
            model_version  TEXT    NOT NULL,
            created_at     TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _save_detections(
    conn: sqlite3.Connection,
    survey_id: str,
    all_detections: list[dict],
    model_version: str = MODEL_VERSION,
) -> None:
    """
    Persist every detection to the detections table.

    Why persist to the database?
        Module 3 (Disease Classification) needs to retrieve plant detections
        without re-running the detector. Storing them in SQLite means Module 3
        can simply query:
            SELECT * FROM detections WHERE survey_id = ?

    Args:
        conn: SQLite connection.
        survey_id: The survey these detections belong to.
        all_detections: The flat list from resolve_survey_detections().
    """
    # Delete any previous detections for this survey before inserting new ones.
    # This allows the same survey to be re-processed (e.g., after retraining the model)
    # without creating duplicate rows.
    conn.execute("DELETE FROM detections WHERE survey_id = ?", (survey_id,))

    rows = [
        (
            survey_id,
            d["image"],
            d["plant_id"],
            d["bbox"][0],  # bbox_x1
            d["bbox"][1],  # bbox_y1
            d["bbox"][2],  # bbox_x2
            d["bbox"][3],  # bbox_y2
            d["confidence"],
            model_version,
        )
        for d in all_detections
    ]

    conn.executemany(
        """
        INSERT INTO detections
            (survey_id, image_filename, plant_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, model_version)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )
    conn.commit()
    logger.info(f"Saved {len(rows)} detections for survey '{survey_id}' to database.")


# ── Request / Response models ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """
    Input schema for POST /detect.

    survey_id: The survey identifier used in Module 1.
               Must match a folder name under data/images/.
               Example: "survey_001"
    """
    survey_id: str
    # "yolo" (default, trained model), "auto" (YOLO, else OpenCV fallback) or
    # "classical" (OpenCV fallback, for the synthetic demo field).
    detector: str | None = None


class DetectionItem(BaseModel):
    """
    A single plant detection in the API response.

    plant_id:   Sequential ID within this survey, starting from 1.
    image:      Filename of the image this detection came from.
    bbox:       [x1, y1, x2, y2] in pixels.
    confidence: Model confidence (0.0 to 1.0).
    """
    plant_id: int
    image: str
    bbox: list[int]
    confidence: float


class DetectResponse(BaseModel):
    """
    Full response for POST /detect.

    status:           "success" or "error"
    model_version:    Which detector produced the boxes.
    survey_id:        Echo of the input survey_id.
    images_processed: How many image files were found and processed.
    plants_detected:  Total number of plant bounding boxes across all images.
    detections:       The flat list of all detected plants.
    """
    model_config = {"protected_namespaces": ()}

    status: str
    survey_id: str
    model_version: str = MODEL_VERSION
    images_processed: int
    plants_detected: int
    detections: list[DetectionItem]


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectResponse)
def detect_plants(request: DetectRequest) -> DetectResponse:
    """
    POST /detect

    Processes all images in a survey folder, detects tomato plants in each,
    assigns unique plant IDs, persists results, and returns the full detection list.

    Steps performed:
        1. Validate that the survey folder exists and contains images.
        2. Load the detector (model loads from disk only on first call).
        3. Run inference on each image.
        4. Assign plant IDs via the identity resolver.
        5. Save all detections to the SQLite database.
        6. Return the structured JSON response.

    Error conditions (all return appropriate HTTP error codes):
        - Survey folder does not exist → 404
        - Survey folder contains no valid images → 400
        - Model weights (best.pt) not found → 503 Service Unavailable
    """
    survey_id = request.survey_id.strip()
    if not survey_id or "/" in survey_id or "\\" in survey_id or survey_id.startswith("."):
        raise HTTPException(status_code=400,
                            detail="survey_id must be a plain folder name")

    # ── Step 1: Validate survey folder ───────────────────────────────────────
    survey_folder = IMAGES_ROOT / survey_id
    if not survey_folder.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Survey folder not found: '{survey_folder}'. "
                f"Create it and place survey images inside before running detection."
            )
        )

    image_files = sorted([
        f for f in survey_folder.iterdir()
        if f.suffix.lower() in VALID_IMAGE_SUFFIXES
    ])

    if not image_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No valid images found in '{survey_folder}'. "
                f"Add .jpg or .png files and try again."
            )
        )

    logger.info(f"Starting detection for survey '{survey_id}': {len(image_files)} images")

    # ── Step 2: Load the detector ─────────────────────────────────────────────
    # get_detector() returns the singleton — model is loaded once.
    try:
        detector = get_detector(request.detector)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        # best.pt doesn't exist yet — training hasn't been run.
        raise HTTPException(
            status_code=503,
            detail=(
                f"Plant detection model is not available: {e}. "
                f"Run the training notebook (notebooks/m2_detection_trials.ipynb) first."
            )
        )

    # ── Step 3: Run inference on every image ──────────────────────────────────
    per_image_results = []  # [(filename, [RawDetection, ...]), ...]

    for img_path in image_files:
        try:
            raw_detections = detector.detect(img_path)
            per_image_results.append((img_path.name, raw_detections))
            logger.info(f"  {img_path.name}: {len(raw_detections)} plants found")
        except Exception as e:
            # Log the error for the specific image but continue processing others.
            # A single corrupted image should not fail the whole survey.
            logger.warning(f"  Skipped {img_path.name}: {e}")
            per_image_results.append((img_path.name, []))

    # ── Step 4: Assign plant IDs ──────────────────────────────────────────────
    resolved = resolve_survey_detections(per_image_results)
    # resolved = {
    #     "total_plants": N,
    #     "detections_by_image": {...},
    #     "all_detections": [{"plant_id", "image", "bbox", "confidence"}, ...]
    # }

    # ── Step 5: Save to database ──────────────────────────────────────────────
    conn = get_connection()
    try:
        _ensure_detections_table(conn)
        _save_detections(conn, survey_id, resolved["all_detections"],
                         model_version=detector.model_version)
    finally:
        conn.close()

    # ── Step 6: Build and return the response ─────────────────────────────────
    detection_items = [
        DetectionItem(
            plant_id=d["plant_id"],
            image=d["image"],
            bbox=d["bbox"],
            confidence=d["confidence"]
        )
        for d in resolved["all_detections"]
    ]

    logger.info(
        f"Detection complete for '{survey_id}': "
        f"{len(image_files)} images, {resolved['total_plants']} plants total."
    )

    return DetectResponse(
        status="success",
        survey_id=survey_id,
        model_version=detector.model_version,
        images_processed=len(image_files),
        plants_detected=resolved["total_plants"],
        detections=detection_items
    )


# ── Health check for this module ──────────────────────────────────────────────

@router.get("/detect/health")
def detection_health() -> dict:
    """
    GET /detect/health

    Returns the status of the detection model.
    Useful for checking whether best.pt is loaded before submitting a survey.
    """
    try:
        detector = get_detector()
        return {
            "module": "Plant Detection",
            "model_loaded": detector.is_ready(),
            "model_version": detector.model_version,
            "status": "ready"
        }
    except FileNotFoundError:
        return {
            "module": "Plant Detection",
            "model_loaded": False,
            "model_version": None,
            "status": "model_not_found — run training notebook first"
        }