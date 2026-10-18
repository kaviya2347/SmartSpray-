"""
app/field_survey/service.py

The business logic for Module 1: validating survey images and registering a
survey.

What Module 1 does:
    1. Refuse to register the same survey twice.
    2. Count the images in data/images/<survey_id>/.
    3. Quality-check every image (readable? blurry? — variance of Laplacian,
       as specified in the design) and store one row per image in
       survey_images.
    4. Save one row to the surveys table.
"""

from datetime import date, datetime
from pathlib import Path

import cv2

from app.db import get_connection

# --- Defaults (can be overridden per request) ---
CROP = "Tomato"
DEFAULT_FIELD_WIDTH_M = 10.0
DEFAULT_FIELD_HEIGHT_M = 10.0
FIELD_SIZE = f"{DEFAULT_FIELD_WIDTH_M:g}m x {DEFAULT_FIELD_HEIGHT_M:g}m"

# Variance-of-Laplacian below this value => the frame is flagged as blurry.
# (Pech-Pacheco et al. 2000; 100 is the usual starting point for 8-bit images.)
BLUR_THRESHOLD = 100.0

# parent.parent.parent walks: service.py -> field_survey/ -> app/ -> SmartSpray/
IMAGES_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "images"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _ensure_images_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS survey_images (
            survey_id      TEXT    NOT NULL,
            image_filename TEXT    NOT NULL,
            width_px       INTEGER,
            height_px      INTEGER,
            blur_score     REAL,
            is_readable    INTEGER NOT NULL DEFAULT 0,
            is_blurry      INTEGER NOT NULL DEFAULT 0,
            checked_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (survey_id, image_filename)
        )
        """
    )


def list_images(survey_id: str) -> list[Path]:
    """Return the image files in data/images/<survey_id>/ (sorted)."""
    survey_folder = IMAGES_ROOT / survey_id
    if not survey_folder.exists():
        raise FileNotFoundError(
            f"No folder found at data/images/{survey_id}/. "
            f"Create it and add 5-10 images before registering this survey."
        )
    return sorted(
        f for f in survey_folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def count_images(survey_id: str) -> int:
    """Looks inside data/images/<survey_id>/ and counts the image files there."""
    image_files = list_images(survey_id)
    if len(image_files) == 0:
        raise ValueError(f"data/images/{survey_id}/ exists but has no images in it.")
    return len(image_files)


def assess_image(path: Path, blur_threshold: float = BLUR_THRESHOLD) -> dict:
    """Readability + sharpness check for one image (never raises)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"image_filename": path.name, "is_readable": False,
                "is_blurry": False, "blur_score": None,
                "width_px": None, "height_px": None}
    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return {"image_filename": path.name, "is_readable": True,
            "is_blurry": score < blur_threshold, "blur_score": round(score, 2),
            "width_px": int(image.shape[1]), "height_px": int(image.shape[0])}


def survey_exists(survey_id: str) -> bool:
    """Checks if this survey_id is already in the database."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT 1 FROM surveys WHERE survey_id = ?", (survey_id,)
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def register_survey(
    survey_id: str,
    crop: str = CROP,
    field_width_m: float = DEFAULT_FIELD_WIDTH_M,
    field_height_m: float = DEFAULT_FIELD_HEIGHT_M,
    blur_threshold: float = BLUR_THRESHOLD,
) -> dict:
    """
    The main entry point for Module 1.
    Returns a dict the API turns into JSON.
    """
    survey_id = (survey_id or "").strip()
    if not survey_id:
        raise ValueError("survey_id must be a non-empty string")
    if "/" in survey_id or "\\" in survey_id or survey_id.startswith("."):
        raise ValueError("survey_id must be a plain folder name")
    if field_width_m <= 0 or field_height_m <= 0:
        raise ValueError("field dimensions must be positive metres")
    if survey_exists(survey_id):
        raise ValueError(f"Survey '{survey_id}' is already registered.")

    image_files = list_images(survey_id)
    if not image_files:
        raise ValueError(f"data/images/{survey_id}/ exists but has no images in it.")
    num_images = len(image_files)
    quality = [assess_image(p, blur_threshold) for p in image_files]

    capture_date = date.today().isoformat()
    created_at = datetime.now().isoformat(timespec="seconds")
    field_size = f"{field_width_m:g}m x {field_height_m:g}m"

    connection = get_connection()
    try:
        _ensure_images_table(connection)
        connection.execute(
            """
            INSERT INTO surveys (survey_id, capture_date, crop, field_size, num_images, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (survey_id, capture_date, crop, field_size, num_images, created_at),
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO survey_images
                (survey_id, image_filename, width_px, height_px, blur_score,
                 is_readable, is_blurry)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(survey_id, q["image_filename"], q["width_px"], q["height_px"],
              q["blur_score"], int(q["is_readable"]), int(q["is_blurry"]))
             for q in quality],
        )
        connection.commit()
    finally:
        connection.close()

    readable = [q for q in quality if q["is_readable"]]
    blurry = [q["image_filename"] for q in quality if q["is_blurry"]]
    unreadable = [q["image_filename"] for q in quality if not q["is_readable"]]

    return {
        "status": "success",
        "survey_id": survey_id,
        "crop": crop,
        "field_size": field_size,
        "num_images": num_images,
        "readable_images": len(readable),
        "sharp_images": len(readable) - len(blurry),
        "blurry_images": blurry,
        "unreadable_images": unreadable,
        "blur_threshold": blur_threshold,
    }


def get_survey(survey_id: str) -> dict | None:
    """Return the stored survey row (or None)."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT survey_id, capture_date, crop, field_size, num_images, created_at "
            "FROM surveys WHERE survey_id = ?", (survey_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    keys = ["survey_id", "capture_date", "crop", "field_size", "num_images", "created_at"]
    return dict(zip(keys, row))


def parse_field_size(field_size: str) -> tuple[float, float]:
    """'10m x 12.5m' -> (10.0, 12.5); falls back to the defaults."""
    try:
        w, h = [float(part.strip().rstrip("mM ")) for part in field_size.lower().split("x")]
        return w, h
    except Exception:
        return DEFAULT_FIELD_WIDTH_M, DEFAULT_FIELD_HEIGHT_M
