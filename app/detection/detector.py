"""
FILE PATH: SmartSpray/app/detection/detector.py

PURPOSE:
    This file is the core inference engine for Module 2.
    It loads the trained YOLOv8 model exactly ONCE when the FastAPI server starts,
    then runs detection on individual images on demand.

    It is intentionally kept to a single responsibility: receive an image path,
    return a list of RawDetection objects. Nothing else.

INPUT:
    A filesystem path (string or Path) to a single image file (.jpg or .png).

OUTPUT:
    A list of RawDetection dataclass objects, each containing:
        - bbox: [x1, y1, x2, y2] in integer pixel coordinates
        - confidence: float between 0.0 and 1.0

FALLBACK:
    If the YOLO weights are missing, get_detector(mode="auto") can fall back to
    ClassicalPlantDetector (OpenCV vegetation segmentation). The fallback is
    OFF by default (so a missing model still returns HTTP 503) and is enabled
    with mode="classical"/"auto" or the environment variable
    SMARTSPRAY_DETECTOR=auto|classical. It is intended for the synthetic demo
    field and is clearly labelled in every result (model_version).

DEPENDENCIES:
    - ultralytics: the official YOLOv8 Python library by Ultralytics Inc.
      (imported lazily so the API still starts without it)
    - pathlib: Python standard library, for safe file path handling.
    - dataclasses: Python standard library, for clean structured data.
    - logging: Python standard library, for printing status messages.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Set up a named logger for this module.
# Messages will appear in the uvicorn console prefixed with "detection.detector".
logger = logging.getLogger("detection.detector")

# ── Constants ────────────────────────────────────────────────────────────────

# This is where the trained model weights live (scripts/m2_train_detector.py or
# notebook Cell 6). Anchored at the repository root, so the working directory
# does not matter.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "models" / "yolov8_plant_detector" / "best.pt"

# Minimum confidence score required to accept a detection.
# 0.25 is the YOLOv8 default. Raise this (e.g. to 0.40) if you get many false positives.
# Lower it (e.g. to 0.15) if plants are being missed.
CONFIDENCE_THRESHOLD = 0.25

# Version string stored alongside every detection so you know which model made the prediction.
MODEL_VERSION = "yolov8n-tomato-plant-v1"
FALLBACK_MODEL_VERSION = "opencv-vegetation-fallback-v1"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RawDetection:
    """
    A single plant detection from a single image.

    Attributes:
        bbox        - Bounding box as [x1, y1, x2, y2] in pixel coordinates.
                      (x1, y1) = top-left corner, (x2, y2) = bottom-right corner.
        confidence  - How certain the model is that this is a tomato plant (0.0 to 1.0).

    Why a dataclass instead of a plain dict?
        - Type hints prevent common bugs (e.g. accidentally putting confidence inside bbox)
        - Easier to add fields later without breaking existing code
        - IDE autocomplete works
    """
    bbox: list[int]       # [x1, y1, x2, y2]
    confidence: float


# ── Main detector class ───────────────────────────────────────────────────────

class PlantDetector:
    """
    Loads a YOLOv8 model once and exposes a detect() method.

    Design decisions:
        - The model is loaded in __init__(), meaning it's loaded once when the
          first PlantDetector instance is created, and stays in memory.
        - We use a module-level _instance variable below to ensure only one
          PlantDetector is created across the entire FastAPI process.
        - detect() is deliberately synchronous. FastAPI can run it in a thread
          pool if async is needed later, but at FYP scale (5-10 images per survey)
          this is not necessary.
    """

    def __init__(self, weights_path: Path | None = None):
        """
        Load the YOLOv8 model from disk into memory.

        Args:
            weights_path: Path to the best.pt file.

        Raises:
            FileNotFoundError: if best.pt does not exist at the given path.
            RuntimeError: if the model file is found but cannot be loaded.
        """
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'.\n"
                f"You must train the model first using notebooks/m2_detection_trials.ipynb "
                f"and ensure Cell 6 (copy best.pt) has been run."
            )

        logger.info(f"Loading YOLOv8 model from: {weights_path}")

        try:
            from ultralytics import YOLO  # lazy import (heavy dependency)

            # ultralytics.YOLO() loads the weights file and prepares the model
            # for inference. This call takes a few seconds — it happens once.
            self._model = YOLO(str(weights_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from '{weights_path}': {e}"
            ) from e

        self._weights_path = weights_path
        logger.info("Model loaded successfully. Ready for inference.")

    def detect(self, image_path: str | Path) -> list[RawDetection]:
        """
        Run YOLOv8 inference on a single image.

        Args:
            image_path: Path to the image file. Can be a string or a Path object.

        Returns:
            A list of RawDetection objects.
            Returns an empty list if no plants are detected (not an error).

        Raises:
            FileNotFoundError: if the image file does not exist.
            ValueError: if the file extension is not a supported image format.
        """
        image_path = Path(image_path)

        # ── Validate input ──────────────────────────────────────────────────
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported image format '{image_path.suffix}'. "
                f"Supported: {SUPPORTED_SUFFIXES}"
            )

        logger.debug(f"Running inference on: {image_path.name}")

        # ── Run inference ───────────────────────────────────────────────────
        # self._model() runs the full YOLOv8 forward pass.
        # conf=CONFIDENCE_THRESHOLD: discard any box below this confidence level.
        # verbose=False: suppress the per-image print output from ultralytics.
        results = self._model(
            str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # results is always a list, one element per image.
        # Since we passed one image path, results[0] is our result.
        result = results[0]

        # ── Extract bounding boxes ──────────────────────────────────────────
        detections = []

        # result.boxes contains all detections for this image.
        # If zero plants were detected, result.boxes is empty and this loop
        # doesn't execute — we return an empty list, not an error.
        for box in result.boxes:

            # box.xyxy[0] is a tensor of shape [4]: [x1, y1, x2, y2]
            # .tolist() converts the tensor to a plain Python list of floats.
            # int() rounds each coordinate to a whole pixel number.
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # box.conf[0] is a 1-element tensor containing the confidence score.
            # .item() converts it to a plain Python float.
            confidence = round(float(box.conf[0].item()), 4)

            detections.append(RawDetection(
                bbox=[x1, y1, x2, y2],
                confidence=confidence
            ))

        logger.debug(f"  → {len(detections)} plants detected in {image_path.name}")
        return detections

    def is_ready(self) -> bool:
        """
        Return True if the model is loaded and ready.
        Used by the /health endpoint and the test suite.
        """
        return self._model is not None

    @property
    def model_version(self) -> str:
        """Return the version string for this model."""
        return MODEL_VERSION


# ── Classical fallback detector ──────────────────────────────────────────────

class ClassicalPlantDetector:
    """
    Training-free plant detector: segments vegetation (green / yellow-green /
    brown leaf tissue) against soil in HSV + excess-green space and returns one
    bounding box per connected plant blob.

    It is a FALLBACK for demos when the YOLO weights are unavailable. It works
    well on the synthetic demo field (plants on a soil background with clear
    spacing) and on sparse, top-down nursery images; it is NOT a replacement
    for the trained YOLOv8 model on dense real fields.
    """

    def __init__(self, min_area_frac: float = 0.0004, max_area_frac: float = 0.25,
                 pad_frac: float = 0.08, square: bool = True, kernel_frac: float = 0.01):
        self.min_area_frac = min_area_frac
        # Morphology kernel as a fraction of the shorter image side. Use a
        # smaller value when plants are small relative to the frame (e.g. a
        # whole-field overview with wide margins) so small leaves survive.
        self.kernel_frac = kernel_frac
        self.max_area_frac = max_area_frac
        self.pad_frac = pad_frac
        # Square boxes keep the leaf's aspect ratio when Module 3 resizes the
        # crop to 224 x 224.
        self.square = square

    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (22, 25, 30), (95, 255, 255))
        b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
        exg = (2 * g - r - b) > 20                     # excess-green index
        mask = (green > 0) | exg
        mask = mask.astype(np.uint8) * 255
        k = max(3, int(round(min(bgr.shape[:2]) * self.kernel_frac)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # remove speckle
        big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k, 2 * k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big)          # fill lesions / gaps
        # A second opening with the same large kernel cuts the thin "necks"
        # that closing creates between neighbouring plants.
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, big)

    def detect_array(self, bgr: np.ndarray) -> list[RawDetection]:
        h, w = bgr.shape[:2]
        mask = self._mask(bgr)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / float(h * w)
            if frac < self.min_area_frac or frac > self.max_area_frac:
                continue
            cxb, cyb = x + bw / 2.0, y + bh / 2.0
            half_w = bw * (1 + 2 * self.pad_frac) / 2.0
            half_h = bh * (1 + 2 * self.pad_frac) / 2.0
            if self.square:
                half_w = half_h = max(half_w, half_h)
            x1, y1 = max(0, int(round(cxb - half_w))), max(0, int(round(cyb - half_h)))
            x2 = min(w - 1, int(round(cxb + half_w)))
            y2 = min(h - 1, int(round(cyb + half_h)))
            fill = area / float(max(bw * bh, 1))
            conf = round(float(min(0.99, 0.5 + 0.5 * fill)), 4)
            dets.append(RawDetection(bbox=[int(x1), int(y1), int(x2), int(y2)],
                                     confidence=conf))
        return dets

    def detect(self, image_path: str | Path) -> list[RawDetection]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported image format '{image_path.suffix}'.")
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return self.detect_array(bgr)

    def is_ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return FALLBACK_MODEL_VERSION


# ── Module-level singleton ────────────────────────────────────────────────────
#
# We want only one PlantDetector in memory no matter how many API requests come in.
# This variable holds the single shared instance.
# It starts as None and is initialised the first time get_detector() is called.
#
_instance: Optional[PlantDetector] = None
_fallback_instance: Optional[ClassicalPlantDetector] = None


def get_detector(mode: Optional[str] = None):
    """
    Return the shared detector.

    mode:
        "yolo"      (default) trained YOLOv8 model; FileNotFoundError if missing
        "auto"      YOLOv8 if the weights exist, otherwise the OpenCV fallback
        "classical" always the OpenCV fallback
        None        read SMARTSPRAY_DETECTOR (default "yolo")

    Raises:
        FileNotFoundError: if best.pt doesn't exist and mode is "yolo".
    """
    global _instance, _fallback_instance
    mode = (mode or os.environ.get("SMARTSPRAY_DETECTOR", "yolo")).lower()
    if mode not in {"yolo", "auto", "classical"}:
        raise ValueError(f"Unknown detector mode '{mode}'")

    if mode == "classical":
        if _fallback_instance is None:
            _fallback_instance = ClassicalPlantDetector()
        return _fallback_instance

    if _instance is None:
        try:
            _instance = PlantDetector(DEFAULT_WEIGHTS_PATH)
        except FileNotFoundError:
            if mode == "auto":
                logger.warning("YOLO weights missing — using the OpenCV fallback detector.")
                if _fallback_instance is None:
                    _fallback_instance = ClassicalPlantDetector()
                return _fallback_instance
            raise
    return _instance
