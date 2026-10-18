"""
FILE PATH: SmartSpray/app/classification/service.py

PURPOSE:
    Orchestration layer for Module 3 — Disease Classification.

    This file contains the business workflow that ties together:
        - database.py  (read detections, save results)
        - classifier.py (run ML inference on crops)
        - prediction.py (data contracts)
        - Filesystem    (load and crop images)

    This file contains NO:
        - FastAPI endpoints
        - SQL queries or CREATE TABLE statements
        - TensorFlow model loading
        - Duplicated logic from other modules

    Single public function: classify_survey(survey_id) -> dict

DEPENDENCIES:
    - cv2 (opencv-python): for loading and cropping images
    - app.db: for database connections
    - app.classification.prediction: for CropInput, ClassificationResult
    - app.classification.classifier: for the singleton DiseaseClassifier
    - app.classification.database: for all data access functions
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from app.db import get_connection
from app.classification.prediction import CropInput, ClassificationResult
from app.classification.classifier import get_classifier
from app.classification.database import (
    ensure_classifications_table,
    fetch_detections_for_survey,
    delete_classifications_for_survey,
    save_classification,
)

# Set up a named logger for this module.
logger = logging.getLogger("classification.service")

# Where survey images are stored on disk.
# Convention: data/images/<survey_id>/<image_filename>
# Same path used by Module 1 (field_survey/service.py) and Module 2 (detection/router.py).
# Anchored at the repository root so the service works from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_ROOT = REPO_ROOT / "data" / "images"

# Grad-CAM overlays (optional) are written here, one folder per survey.
OUTPUTS_ROOT = REPO_ROOT / "data" / "outputs"

# Image formats accepted when loading survey images.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_image_path(
    survey_id: str,
    image_filename: str,
) -> Path:
    """
    Build the full path to a survey image on disk.

    Args:
        survey_id:      The survey identifier, e.g. "survey_001".
        image_filename: The image file name, e.g. "IMG_001.jpg".

    Returns:
        Path object pointing to the image file.

    Raises:
        ValueError: if the file extension is not a supported image format.
    """
    img_path = IMAGES_ROOT / survey_id / image_filename

    ext = img_path.suffix.lower()

    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}' for '{image_filename}'. "
            f"Supported: {VALID_EXTENSIONS}"
        )

    return img_path


def _load_and_crop(
    image_path: Path,
    bbox_x1: float,
    bbox_y1: float,
    bbox_x2: float,
    bbox_y2: float,
) -> np.ndarray:
    """
    Load a survey image and crop a single plant using its bounding box.

    Args:
        image_path: Full path to the survey image on disk.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2:
            Bounding box coordinates in pixels.

    Returns:
        The cropped region as a BGR numpy array (uint8).

    Raises:
        FileNotFoundError: if the image file does not exist.
        ValueError: if the crop region is empty or invalid.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load the full survey image.
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Failed to load image: {image_path}"
        )

    # Convert bounding box to integer pixel coordinates.
    x1 = int(round(bbox_x1))
    y1 = int(round(bbox_y1))
    x2 = int(round(bbox_x2))
    y2 = int(round(bbox_y2))

    # Clamp to image boundaries.
    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Ensure the crop region has positive dimensions.
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box after clamping: "
            f"[{x1}, {y1}, {x2}, {y2}] "
            f"for image of size {width}x{height}"
        )

    crop = image[y1:y2, x1:x2]

    return crop


def _build_classification_result(
    survey_id: str,
    image_filename: str,
    plant_id: int,
    bbox: list[float],
    prediction: dict,
) -> ClassificationResult:
    """
    Wrap a raw classifier prediction dict into a ClassificationResult dataclass.

    Args:
        survey_id: The survey identifier.
        image_filename: The source image file name.
        plant_id: The unique plant ID.
        bbox: The bounding box [x1, y1, x2, y2].
        prediction:
            The dict returned by DiseaseClassifier.predict():
            {"label": str, "confidence": float, "all_probs": dict}.

    Returns:
        A populated ClassificationResult instance.
    """

    return ClassificationResult(
        survey_id=survey_id,
        image_filename=image_filename,
        plant_id=plant_id,
        bbox=bbox,
        label=prediction["label"],
        confidence=prediction["confidence"],
        all_probs=prediction["all_probs"],
        uncertain=bool(prediction.get("uncertain", False)),
    )


def _is_file_backed(conn) -> bool:
    """True for a normal on-disk SQLite connection, False for ':memory:'."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        return bool(row and row[2])
    except Exception:
        return False


def _save_gradcam(classifier, crop_bgr: np.ndarray, survey_id: str, plant_id: int) -> str:
    """Write a Grad-CAM overlay for one plant and return its repo-relative path."""
    from app.classification.explain import grad_cam, overlay_heatmap

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    heat, _, _ = grad_cam(classifier.model, crop_rgb)
    overlay = overlay_heatmap(crop_rgb, heat)
    out_dir = OUTPUTS_ROOT / survey_id / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"plant_{plant_id:03d}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    try:
        return out_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(out_path)


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_survey(survey_id: str, save_gradcam: bool = False) -> dict:
    """
    Run disease classification on every detected plant in a survey.

    This is the single entry point for Module 3 orchestration.

    The workflow is:
        1. Open a database connection.
        2. Ensure the classifications table exists.
        3. Fetch all plant detections for the survey.
        4. For each detection:
           a. Load the original survey image.
           b. Crop the plant region.
           c. Validate the crop.
           d. Run the classifier.
           e. Build a ClassificationResult.
           f. Save the result to the database.
        5. Return a structured summary.

    Args:
        survey_id: The survey identifier.
        save_gradcam: also write a Grad-CAM overlay per plant to
            data/outputs/<survey_id>/gradcam/ (explainability).

    Returns:
        A dict containing:
            - status
            - survey_id
            - total_plants
            - classified
            - failed
            - uncertain
            - results

    Raises:
        ValueError:
            if the survey has no plant detections.
    """

    logger.info(
        f"Starting disease classification for survey '{survey_id}'."
    )

    # ── Step 1: Open database connection ──────────────────────────────────────
    conn = get_connection()

    # The connection is closed at the end when it is a normal on-disk
    # database. An injected in-memory connection (tests) is left open so the
    # caller can inspect the saved rows.
    try:
        return _classify_with_connection(conn, survey_id, save_gradcam)
    finally:
        if _is_file_backed(conn):
            conn.close()


def _classify_with_connection(conn, survey_id: str, save_gradcam: bool) -> dict:

    # ── Step 2: Ensure the classifications (and upstream detections) tables exist
    from app.detection.router import _ensure_detections_table

    _ensure_detections_table(conn)
    ensure_classifications_table(conn)

    # ── Step 3: Fetch detections ──────────────────────────────────────────────
    detections = fetch_detections_for_survey(
        conn,
        survey_id,
    )

    if not detections:
        raise ValueError(
            f"No plant detections found for survey '{survey_id}'. "
            f"Run Module 2 (POST /detect) first to detect plants."
        )

    logger.info(
        f"Found {len(detections)} plants to classify "
        f"for survey '{survey_id}'."
    )

    # ── Step 4: Clear previous classification results ─────────────────────────
    # This allows re-running classification without duplicate rows.
    delete_classifications_for_survey(
        conn,
        survey_id,
    )

    # ── Step 5: Classify each plant ───────────────────────────────────────────
    # Load the classifier singleton.
    classifier = get_classifier()

    results = []
    failures = 0

    for det in detections:

        plant_id = det["plant_id"]
        image_filename = det["image_filename"]

        bbox = [
            det["bbox_x1"],
            det["bbox_y1"],
            det["bbox_x2"],
            det["bbox_y2"],
        ]

        try:

            # ── Step 5a: Resolve image path ───────────────────────────────────
            img_path = _resolve_image_path(
                survey_id,
                image_filename,
            )

            # ── Step 5b: Load and crop plant ──────────────────────────────────
            crop_bgr = _load_and_crop(
                img_path,
                det["bbox_x1"],
                det["bbox_y1"],
                det["bbox_x2"],
                det["bbox_y2"],
            )

            # ── Step 5c: Validate crop size ───────────────────────────────────
            if (
                crop_bgr.shape[0] < 32
                or crop_bgr.shape[1] < 32
            ):
                raise ValueError(
                    f"Crop too small for classification: "
                    f"{crop_bgr.shape[1]}x{crop_bgr.shape[0]} "
                    f"(minimum 32x32)"
                )

            # ── Step 5d: Run classifier inference ────────────────────────────
            prediction = classifier.predict(crop_bgr)

            # ── Step 5e: Build result dataclass ───────────────────────────────
            classification_result = _build_classification_result(
                survey_id=survey_id,
                image_filename=image_filename,
                plant_id=plant_id,
                bbox=bbox,
                prediction=prediction,
            )

            # ── Step 5f: Save result to database ──────────────────────────────
            save_classification(
                conn,
                classification_result,
            )

            # Add successful result to response.
            item = {
                "plant_id": plant_id,
                "image_filename": image_filename,
                "bbox": bbox,
                "label": classification_result.label,
                "confidence": classification_result.confidence,
                "all_probs": classification_result.all_probs,
                "uncertain": classification_result.uncertain,
            }
            if save_gradcam:
                try:
                    item["gradcam_path"] = _save_gradcam(
                        classifier, crop_bgr, survey_id, plant_id)
                except Exception as exc:  # explainability must never block M3
                    logger.warning(f"  Plant {plant_id}: Grad-CAM skipped — {exc}")
            results.append(item)

            logger.info(
                f"  Plant {plant_id}: "
                f"{classification_result.label} "
                f"(confidence: "
                f"{classification_result.confidence:.4f})"
            )

        except FileNotFoundError as e:

            failures += 1

            logger.warning(
                f"  Plant {plant_id}: Skipped — {e}"
            )

            results.append(
                {
                    "plant_id": plant_id,
                    "image_filename": image_filename,
                    "bbox": bbox,
                    "label": "error",
                    "confidence": 0.0,
                    "all_probs": {},
                    "error": str(e),
                }
            )

        except ValueError as e:

            failures += 1

            logger.warning(
                f"  Plant {plant_id}: Skipped — {e}"
            )

            results.append(
                {
                    "plant_id": plant_id,
                    "image_filename": image_filename,
                    "bbox": bbox,
                    "label": "error",
                    "confidence": 0.0,
                    "all_probs": {},
                    "error": str(e),
                }
            )

        except Exception as e:

            failures += 1

            logger.error(
                f"  Plant {plant_id}: "
                f"Unexpected error during classification: {e}",
                exc_info=True,
            )

            results.append(
                {
                    "plant_id": plant_id,
                    "image_filename": image_filename,
                    "bbox": bbox,
                    "label": "error",
                    "confidence": 0.0,
                    "all_probs": {},
                    "error": str(e),
                }
            )

    # ── Step 6: Build summary ─────────────────────────────────────────────────
    total_plants = len(detections)
    classified = total_plants - failures

    status = (
        "success"
        if failures == 0
        else "partial"
    )

    summary = {
        "status": status,
        "survey_id": survey_id,
        "total_plants": total_plants,
        "classified": classified,
        "failed": failures,
        "uncertain": sum(1 for r in results if r.get("uncertain")),
        "model_version": getattr(classifier, "model_version", None)
        if isinstance(getattr(classifier, "model_version", None), str) else None,
        "results": results,
    }

    logger.info(
        f"Classification complete for '{survey_id}': "
        f"{classified}/{total_plants} plants classified "
        f"({failures} failures)."
    )

    return summary