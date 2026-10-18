"""
FILE PATH: SmartSpray/app/classification/prediction.py

PURPOSE:
    Defines the data structures (dataclasses) used throughout Module 3.
    This file contains NO machine learning code, NO database code, NO HTTP logic,
    and NO business logic. It exists solely to define the data contracts between
    the other layers of the module.

    Every other file in app/classification/ imports from here.

DATA CONTRACTS:
    CropInput
        The service layer creates one CropInput per detected plant.
        It bundles the cropped image pixels with the metadata needed to
        identify which plant the crop came from.

    ClassificationResult
        The classifier returns one ClassificationResult per CropInput.
        It contains the predicted label, confidence score, full probability
        vector, and enough metadata to persist the result to the database.

DEPENDENCIES:
    - dataclasses: Python standard library
    - numpy: for the image array type hint
    - typing: Python standard library, for type hints
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

# The canonical class names in the order the model was trained on.
# This MUST match the class_order.json saved alongside best.keras.
# Order: ["early_blight", "healthy", "late_blight", "septoria_leaf_spot"]
CLASS_NAMES: list[str] = [
    "early_blight",
    "healthy",
    "late_blight",
    "septoria_leaf_spot",
]

# Version string stored alongside every classification result.
# Update this when the model is retrained with a different architecture or dataset.
# v2 = retrained on a de-duplicated, seeded, stratified 70/15/15 split
#      (scripts/m3_train_classifier.py, manifest models/classification/split_manifest.json).
MODEL_VERSION: str = "efficientnetb0-tomato-disease-v2-701515"

# Predictions whose (calibrated) confidence is below this value are flagged
# "uncertain" and are never sprayed automatically by Module 6.
UNCERTAINTY_THRESHOLD: float = 0.60


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CropInput:
    """
    A single plant crop ready for disease classification.

    The service layer creates one of these for every plant detection returned
    by Module 2. The classifier receives this and returns a ClassificationResult.

    Attributes:
        survey_id:      The survey this plant belongs to (e.g. "survey_001").
        image_filename: The original image file this plant was detected in.
        plant_id:       The unique plant ID assigned by Module 2's identity resolver.
        bbox:           The bounding box [x1, y1, x2, y2] in pixel coordinates.
        pixels:         The cropped image as an RGB numpy array (uint8).
                        Expected to be 224x224 (resized by the service layer).
    """
    survey_id: str
    image_filename: str
    plant_id: int
    bbox: list[int]
    pixels: np.ndarray


@dataclass
class ClassificationResult:
    """
    The output of classifying a single plant crop.

    Returned by the classifier and passed to the database layer for persistence.
    Also used by the router to build the HTTP response.

    Attributes:
        survey_id:      The survey this classification belongs to.
        image_filename: The original image file this plant was detected in.
        plant_id:       The unique plant ID assigned by Module 2.
        bbox:           The bounding box [x1, y1, x2, y2] in pixel coordinates.
        label:          The predicted disease class (one of CLASS_NAMES).
        confidence:     The model's confidence in the predicted class (0.0 to 1.0).
        all_probs:      Full probability vector as {class_name: probability}.
                        All values are between 0.0 and 1.0 and sum to 1.0.
        model_version:  Identifies which model version produced this prediction.
        uncertain:      True when confidence < UNCERTAINTY_THRESHOLD.
    """
    survey_id: str
    image_filename: str
    plant_id: int
    bbox: list[int]
    label: str
    confidence: float
    all_probs: dict[str, float]
    model_version: str = MODEL_VERSION
    uncertain: bool = False
