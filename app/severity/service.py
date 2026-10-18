"""
app/severity/service.py

Service layer for Module 4 — Disease Severity Estimation.

Orchestrates the severity estimation workflow for a complete survey:
reads per-plant detections and disease classifications from the database,
loads the source images, crops each plant bounding box, and calls the
OpenCV-based severity estimator.

ACCURACY LIMITATION:
severity_score is an OpenCV heuristic estimate of lesion-area proportion.
No ground-truth severity annotations exist. No real-world severity accuracy
claim can be made. is_valid=False results are passed through as-is; they
must not be silently dropped.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from app.severity.database import (
    get_detections_for_severity,
    save_severity,
    get_connection,
)
from app.severity.estimator import get_estimator, SeverityResult

logger = logging.getLogger("severity.service")

# Where survey images are stored on disk.
# Convention: data/images/<survey_id>/<image_filename>
# Same path used by Module 1 (field_survey), Module 2 (detection), and
# Module 3 (classification) services.
IMAGES_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "images"


def _load_and_crop(
    image_path: Path,
    bbox_x1: float,
    bbox_y1: float,
    bbox_x2: float,
    bbox_y2: float,
) -> np.ndarray:
    """
    Load a survey image and crop a single plant using its bounding box.

    Uses IDENTICAL crop logic to Module 3's classification service
    (_load_and_crop in app/classification/service.py): the bounding-box
    rounding, boundary clamping, and slice are byte-for-byte equivalent.
    The only difference is that the result is returned as an RGB array
    (the severity estimator expects RGB, whereas the classifier expects BGR
    and converts internally).

    Args:
        image_path: Full path to the survey image on disk.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2:
            Bounding box coordinates in pixels.

    Returns:
        The cropped region as an RGB numpy array (uint8) of shape (H, W, 3).

    Raises:
        FileNotFoundError: if the image file does not exist.
        ValueError: if the crop region is empty or invalid after clamping.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the full survey image (BGR, as cv2.imread always returns BGR).
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert bounding box to integer pixel coordinates (identical to Module 3).
    x1 = int(round(bbox_x1))
    y1 = int(round(bbox_y1))
    x2 = int(round(bbox_x2))
    y2 = int(round(bbox_y2))

    # Clamp to image boundaries (identical to Module 3).
    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Ensure the crop region has positive dimensions (identical to Module 3).
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box after clamping: "
            f"[{x1}, {y1}, {x2}, {y2}] "
            f"for image of size {width}x{height}"
        )

    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    return crop_rgb


def estimate_survey(survey_id: str) -> list[SeverityResult]:
    """
    Estimate severity for every classified plant in a survey.

    Workflow:
      1. Query database for all detections+classifications for survey_id.
         (Uses get_detections_for_severity which JOINs detections +
         classifications ON (survey_id, plant_id).)
      2. If no rows: raise ValueError.
      3. For each row: crop (BGR then BGR->RGB), call estimator, collect.
      4. Return list[SeverityResult].

    Error handling per plant:
      - FileNotFoundError (image missing): log WARNING, skip, continue.
      - ValueError (zero-area crop): log WARNING, skip, continue.
      - Any other exception: log ERROR with plant_id, skip, continue.

    Do NOT add a healthy short-circuit; the estimator handles healthy
    internally. Do NOT filter out is_valid=False results; return them.

    Log:
      - INFO: "Starting severity estimation for survey '{survey_id}': N plants
        (querying DB)" then "Starting severity estimation for survey
        '{survey_id}': N plants" after counting.
      - DEBUG per plant: "plant_id={id}: label={label} score={score:.4f}
        cat={cat} valid={valid}"
      - INFO: completion summary with processed/valid/invalid/skipped counts.
    """
    logger.info(
        "Starting severity estimation for survey '%s': 0 plants (querying DB)",
        survey_id,
    )

    conn = get_connection()
    try:
        detections = get_detections_for_severity(conn, survey_id)
    finally:
        conn.close()

    if not detections:
        raise ValueError(
            f"No classified detections found for survey_id='{survey_id}'"
        )

    n_plants = len(detections)
    logger.info(
        "Starting severity estimation for survey '%s': %d plants",
        survey_id,
        n_plants,
    )

    results: list[SeverityResult] = []
    skipped = 0

    for det in detections:
        plant_id = det["plant_id"]
        label = det["disease_label"]
        image_filename = det["image_filename"]

        image_path = IMAGES_ROOT / survey_id / image_filename

        try:
            crop_rgb = _load_and_crop(
                image_path,
                det["bbox_x1"],
                det["bbox_y1"],
                det["bbox_x2"],
                det["bbox_y2"],
            )
        except FileNotFoundError:
            logger.warning(
                "plant_id=%s: image not found (%s), skipping",
                plant_id,
                image_path,
            )
            skipped += 1
            continue
        except ValueError as exc:
            logger.warning(
                "plant_id=%s: invalid crop (%s), skipping",
                plant_id,
                exc,
            )
            skipped += 1
            continue

        try:
            result = get_estimator().estimate(
                crop_rgb,
                label,
                plant_id,
                survey_id,
            )
        except Exception as exc:
            logger.error(
                "plant_id=%s: estimator raised unexpected exception: %s",
                plant_id,
                exc,
                exc_info=True,
            )
            skipped += 1
            continue

        logger.debug(
            "plant_id=%s: label=%s score=%.4f cat=%s valid=%s",
            plant_id,
            label,
            result.severity_score,
            result.severity_category,
            result.is_valid,
        )
        results.append(result)

    n_valid = sum(1 for r in results if r.is_valid)
    n_invalid = sum(1 for r in results if not r.is_valid)

    logger.info(
        "Severity estimation complete for '%s': %d processed, %d valid, "
        "%d invalid, %d skipped",
        survey_id,
        n_plants,
        n_valid,
        n_invalid,
        skipped,
    )

    return results

