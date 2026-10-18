"""
FILE PATH: SmartSpray/app/severity/estimator.py

PURPOSE:
    The core computer-vision engine for Module 4 — Disease Severity Estimation.

    This module estimates, from a plant crop image ALONE, the proportion of
    the visible leaf area that shows disease symptoms. It is a deterministic
    OpenCV heuristic estimator. It is NOT a trained machine-learning model
    and makes NO real-world accuracy claims.

    Pipeline:
        1. Leaf segmentation (HSV green/yellow-green masking + morphology)
        2. Disease-specific lesion detection (per-disease HSV/Lab thresholds)
        3. Morphological cleanup restricted to the leaf mask
        4. severity_score = lesion_px / leaf_px   (0.0 .. 1.0)
        5. severity_category derived from the score using SEVERITY_BANDS

    The severity SCORE is measured from the image. The severity CATEGORY is
    always derived from the score via the approved bands. No caller may
    supply a severity value or category; severity is never inferred purely
    from the disease label.

DEPENDENCIES:
    - numpy
    - cv2 (opencv-python)
    - dataclasses / typing / logging (standard library)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Named logger for this module.
logger = logging.getLogger("severity.estimator")

# Version identifier for the OpenCV severity estimator (not an ML model).
MODEL_VERSION = "opencv-severity-v1"

# Minimum leaf fill ratio: leaf_px / bbox_pixels. Below this the bounding
# box is considered a poor/partial detection and the result is invalid.
MIN_LEAF_FILL_RATIO = 0.20

# Severity category strings (approved band names from the master prompt).
CATEGORY_LOW = "low"
CATEGORY_MEDIUM = "medium"
CATEGORY_HIGH = "high"
@dataclass
class LeafSegmentationParams:
    """
    Parameters for the leaf-segmentation stage (pipeline Step 2).

    Attributes:
        hsv_lower: HSV lower bound (H, S, V) for green/yellow-green leaf tissue.
        hsv_upper: HSV upper bound (H, S, V) for green/yellow-green leaf tissue.
        close_kernel_size: kernel size for morphological closing; fills small
            holes inside the leaf (veins, slight colour variation).
        fill_kernel_size: kernel size (odd) for the large elliptical closing
            that expands the green leaf component into diseased tissue so that
            leaf_px represents total leaf area, not just the green area.
        min_leaf_fill_ratio: leaf_px / bbox_pixels threshold below which the
            detection is treated as a poor/partial bounding box.
    """
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    close_kernel_size: int = 7
    fill_kernel_size: int = 21
    min_leaf_fill_ratio: float = MIN_LEAF_FILL_RATIO


@dataclass
class DiseaseThresholdProfile:
    """
    Disease-specific lesion-detection parameters (pipeline Step 3).

    Attributes:
        disease: the Module 3 disease label this profile targets.
        hsv_lesion_lower: HSV lower bound for diseased tissue.
        hsv_lesion_upper: HSV upper bound for diseased tissue.
        lab_use: whether to also apply a Lab-space lesion mask.
        lab_lesion_lower: Lab lower bound for diseased tissue.
        lab_lesion_upper: Lab upper bound for diseased tissue.
        open_kernel_size: morphological opening kernel (removes noise).
        close_kernel_size: morphological closing kernel (fills lesion holes).
        min_component_area_px: minimum connected-component area to count.
        max_component_area_ratio: max lesion component as a fraction of
            leaf_px; larger components are treated as background leakage.
    """
    disease: str
    hsv_lesion_lower: tuple[int, int, int]
    hsv_lesion_upper: tuple[int, int, int]
    lab_use: bool = False
    lab_lesion_lower: tuple[int, int, int] = (0, 0, 0)
    lab_lesion_upper: tuple[int, int, int] = (255, 255, 255)
    open_kernel_size: int = 3
    close_kernel_size: int = 5
    min_component_area_px: int = 0
    max_component_area_ratio: float = 1.0


@dataclass
class SeverityResult:
    """
    The output of estimating severity for ONE plant crop.

    Mirrors the plant_severity table columns so the object can be passed to
    app.severity.database.save_severity() with no schema mismatch.

    Attributes:
        plant_id: plant identifier from Module 2.
        survey_id: survey this plant belongs to.
        disease_label: echo of the Module 3 disease label.
        severity_score: float in [0.0, 1.0], measured lesion_px / leaf_px.
        severity_category: "low" | "medium" | "high" derived from the score.
        leaf_px: number of detected leaf pixels.
        lesion_px: number of detected lesion pixels within the leaf.
        is_valid: False when leaf segmentation failed or leaf fill is low.
        model_version: version string of this estimator.
    """
    plant_id: int
    survey_id: str
    disease_label: str
    severity_score: float
    severity_category: str
    leaf_px: int
    lesion_px: int
    is_valid: bool
    model_version: str = MODEL_VERSION
# Severity band boundaries from the original SmartSpray architecture:
# <20% → low, 20-50% → medium, >50% → high.
# These thresholds define decision categories for Module 6 (Treatment Decision Engine).
# They are engineering design parameters derived from the original system specification,
# NOT empirically calibrated agronomic severity thresholds from plant pathology literature.
# severity_score = lesion_px / leaf_px is a heuristic estimate; these bands
# are applied to that estimate to produce a categorical output for Module 6.
#
# Boundary behaviour is deterministic: the FIRST band where
# lower <= score < upper is selected.
SEVERITY_BANDS: list[tuple[float, float, str]] = [
    (0.00, 0.20, CATEGORY_LOW),
    (0.20, 0.50, CATEGORY_MEDIUM),
    (0.50, 1.01, CATEGORY_HIGH),
]


# DENOMINATOR NOTE (current implementation):
# leaf_px is computed from filled_leaf_body, which is the green-tissue HSV
# largest-connected-component (the "green leaf component") expanded by a large
# elliptical morphological closing (fill_kernel_size=21, MORPH_ELLIPSE). This
# expansion fills in non-green diseased tissue (brown lesions, dark water-soaked
# areas) that is SURROUNDED by green tissue, so leaf_px represents total leaf
# area (healthy + diseased) rather than just the remaining green area.
#
# LIMITATION: when the green HSV mask finds NO leaf pixels at all
# (green_component empty → filled_leaf_body empty), the morphological closing
# cannot create leaf area from nothing. On such images leaf_px=0, the severity
# score is 0.0, and is_valid may be False (below min_leaf_fill_ratio). This is
# a complete leaf-segmentation failure, NOT denominator bias. In the Stage 3A
# real-image diagnostic this was the dominant cause of late_blight zero-lesion
# results: 25 of 26 zero-lesion late_blight images had leaf_px=0 (no green
# component detected at all), which the filled-leaf-body expansion cannot fix.
# The remaining 1 of 26 had a detected leaf (fill_ratio 0.2743) but the
# late_blight lesion profile did not fire on its tissue.
#
# severity_score = lesion_px / leaf_px where leaf_px is the filled-leaf-body
# pixel count. No severity ground truth exists. No real-world severity accuracy
# is claimed.



# Disease-specific lesion profiles. Rationale per disease (master prompt §5.2).
DEFAULT_PROFILES: dict[str, DiseaseThresholdProfile] = {
    "early_blight": DiseaseThresholdProfile(
        disease="early_blight",
        # Concentric dark-brown rings on green leaf tissue: brown/orange-brown
        # hue, low-mid saturation, low-mid value.
        hsv_lesion_lower=(10, 40, 40),
        hsv_lesion_upper=(40, 255, 255),
        lab_use=True,
        # Lab: low L* (dark), positive a* (reddish-brown) surface.
        lab_lesion_lower=(0, 115, 0),
        lab_lesion_upper=(110, 255, 255),
        open_kernel_size=3,               # remove tiny noise spots
        close_kernel_size=5,              # merge fragmented ring segments
        min_component_area_px=9,          # at least a 3x3 spot
        max_component_area_ratio=1.0,     # no size cap for dark lesions
    ),
    "late_blight": DiseaseThresholdProfile(
        disease="late_blight",
        # Large coalescing dark greenish-gray/black lesions: very low value,
        # low saturation, blue-gray cast.
        hsv_lesion_lower=(0, 0, 0),
        hsv_lesion_upper=(180, 255, 90),
        lab_use=False,
        close_kernel_size=11,             # coalesce fragmented blobs
        min_component_area_px=20,
        max_component_area_ratio=1.0,
    ),
    "septoria_leaf_spot": DiseaseThresholdProfile(
        disease="septoria_leaf_spot",
        # Small tan/white centers + dark margins: bright low-sat centers.
        hsv_lesion_lower=(10, 0, 90),
        hsv_lesion_upper=(40, 255, 255),
        lab_use=True,
        lab_lesion_lower=(25, 90, 0),     # mid-high L*, reddish/neutral
        lab_lesion_upper=(255, 255, 255),
        open_kernel_size=2,               # open small spots
        close_kernel_size=5,
        min_component_area_px=12,         # spots ~4x4..16x16 px
        max_component_area_ratio=0.2,     # reject components >20% of leaf
    ),
}

# Canonical Module 3 class order (matches class_order.json).
KNOWN_LABELS: tuple[str, ...] = (
    "early_blight",
    "healthy",
    "late_blight",
    "septoria_leaf_spot",
)
class SeverityEstimator:
    """
    Deterministic OpenCV severity estimator.

    Holds configuration (leaf params, disease profiles, bands), NOT per-image
    mutable state. Each estimate() call is independent and deterministic.

    Singleton-wrapped via get_estimator() at the bottom of this module.
    """

    def __init__(
        self,
        profiles: Optional[dict[str, DiseaseThresholdProfile]] = None,
        bands: Optional[list[tuple[float, float, str]]] = None,
        leaf_params: Optional[LeafSegmentationParams] = None,
    ) -> None:
        """
        Initialise the estimator with configuration.

        Args:
            profiles: disease -> DiseaseThresholdProfile map. Defaults to
                DEFAULT_PROFILES.
            bands: score bands. Defaults to SEVERITY_BANDS.
            leaf_params: leaf segmentation parameters. Defaults to a
                green/yellow-green HSV range tuned for tomato leaf tissue.
        """
        self._profiles = DEFAULT_PROFILES if profiles is None else profiles
        self._bands = SEVERITY_BANDS if bands is None else bands
        if leaf_params is None:
            self._leaf_params = LeafSegmentationParams(
                # Green/yellow-green tomato leaf tissue; excludes brown soil
                # (H ~10-20), dark mulch (V < 50), blue/white sky (H > 100).
                hsv_lower=(25, 40, 40),
                hsv_upper=(90, 255, 255),
                close_kernel_size=7,
                min_leaf_fill_ratio=MIN_LEAF_FILL_RATIO,
            )
        else:
            self._leaf_params = leaf_params

    def _score_to_category(self, score: float) -> str:
        """
        Map a severity score to a category using the approved bands.

        Deterministic: returns the FIRST band where
        band.lower <= score < band.upper. If no band matches (should not
        happen for clamped scores in [0,1]), logs a warning and returns
        "low".

        Args:
            score: severity score in [0.0, 1.0].

        Returns:
            "low" | "medium" | "high".
        """
        for lower, upper, category in self._bands:
            if lower <= score < upper:
                return category
        logger.warning("No band matched score %s; defaulting to %s",
                       score, CATEGORY_LOW)
        return CATEGORY_LOW
    def _segment_leaf(self, crop: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Segment the leaf region from the background using HSV colour.

        Steps:
            1. Convert RGB -> HSV.
            2. Threshold on the green/yellow-green leaf range.
            3. Morphological closing fills holes inside the leaf.
            4. Keep the largest connected component (main leaf body) to
               exclude neighbouring plants/leaves inside the bbox.
            5. Expand the green component with a large elliptical closing
               (fill_kernel_size) to recover non-green diseased tissue inside
               the leaf body.

        # filled_leaf_body: green_component expanded by large morphological closing.
        # This fills in non-green diseased tissue (brown lesions, dark water-soaked
        # areas) so that leaf_px represents total leaf area (healthy + diseased),
        # not just the remaining healthy green area. Without this, severity_score
        # is inflated on diseased leaves because leaf_px undercounts the denominator.
        # This is a heuristic approximation; it assumes the leaf body is spatially
        # contiguous and that the large-kernel closing fills internal diseased regions
        # without extending substantially beyond the leaf boundary.

        Returns (filled_leaf_body, leaf_px) where leaf_px is the number of
        nonzero pixels in filled_leaf_body. Failed segmentation yields an empty
        mask and 0 pixels (caller handles validity).
        """
        if crop is None or crop.size == 0:
            return np.zeros((0, 0), dtype=bool), 0

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        lower = np.array(self._leaf_params.hsv_lower, dtype=np.uint8)
        upper = np.array(self._leaf_params.hsv_upper, dtype=np.uint8)
        leaf_mask = cv2.inRange(hsv, lower, upper) > 0

        kernel = np.ones(
            (self._leaf_params.close_kernel_size,
             self._leaf_params.close_kernel_size),
            dtype=np.uint8,
        )
        leaf_mask = cv2.morphologyEx(
            leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

        # Keep the largest foreground component as the green spatial seed.
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            leaf_mask.astype(np.uint8), connectivity=8
        )
        if n_components > 1:
            largest_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            green_component = (labels == largest_idx)
        else:
            green_component = leaf_mask

        # Expand the green seed into non-green diseased tissue with a large
        # elliptical closing (see docstring note above).
        fill_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._leaf_params.fill_kernel_size,
             self._leaf_params.fill_kernel_size),
        )
        filled_leaf_body = cv2.morphologyEx(
            green_component.astype(np.uint8), cv2.MORPH_CLOSE, fill_kernel
        ).astype(bool)

        leaf_px = int(cv2.countNonZero(filled_leaf_body.astype(np.uint8)))
        return filled_leaf_body, leaf_px
    def _segment_lesions(
        self,
        crop: np.ndarray,
        leaf_mask: np.ndarray,
        profile: DiseaseThresholdProfile,
    ) -> tuple[np.ndarray, int]:
        """
        Detect diseased pixels within the leaf mask for a disease profile.

        Steps:
            1. Build an HSV lesion mask using the profile range.
            2. Optionally build a Lab lesion mask and OR with HSV.
            3. Morphological opening removes speckle noise.
            4. Morphological closing fills holes inside lesion regions.
            5. Connected-component size filter (esp. for septoria spots).
            6. Intersect with the leaf mask so background is never counted.

        Returns (lesion_mask, lesion_px).
        """
        if leaf_mask is None or leaf_mask.sum() == 0:
            return leaf_mask, 0

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        lower = np.array(profile.hsv_lesion_lower, dtype=np.uint8)
        upper = np.array(profile.hsv_lesion_upper, dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower, upper) > 0

        lab_mask = np.zeros_like(hsv_mask)
        if profile.lab_use:
            lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
            lab_lower = np.array(profile.lab_lesion_lower, dtype=np.uint8)
            lab_upper = np.array(profile.lab_lesion_upper, dtype=np.uint8)
            lab_mask = cv2.inRange(lab, lab_lower, lab_upper) > 0

        lesion = hsv_mask | lab_mask

        open_k = np.ones(
            (profile.open_kernel_size, profile.open_kernel_size), dtype=np.uint8
        )
        lesion = cv2.morphologyEx(
            lesion.astype(np.uint8), cv2.MORPH_OPEN, open_k
        ).astype(bool)

        close_k = np.ones(
            (profile.close_kernel_size, profile.close_kernel_size), dtype=np.uint8
        )
        lesion = cv2.morphologyEx(
            lesion.astype(np.uint8), cv2.MORPH_CLOSE, close_k
        ).astype(bool)

        # Final intersection with the leaf mask.
        lesion = lesion & leaf_mask

        # Connected-component size filtering (septoria spots are small).
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            lesion.astype(np.uint8), connectivity=8
        )
        if n_components > 1:
            keep = np.zeros_like(lesion)
            leaf_area = max(int(leaf_mask.sum()), 1)
            max_allowed = max(
                profile.min_component_area_px,
                int(leaf_area * profile.max_component_area_ratio),
            )
            for i in range(1, n_components):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if profile.min_component_area_px <= area <= max_allowed:
                    keep[labels == i] = True
            lesion = keep

        lesion_px = int(lesion.sum())
        return lesion, lesion_px
    def estimate(
        self,
        crop_rgb: np.ndarray,
        disease_label: str,
        plant_id: int,
        survey_id: str,
    ) -> SeverityResult:
        """
        Estimate severity for one plant crop.

        The severity SCORE is computed from the image pixels (lesion_px /
        leaf_px). The severity CATEGORY is always derived from the score via
        SEVERITY_BANDS. No caller supplies a severity value or category.

        Args:
            crop_rgb: RGB numpy array (uint8) of the plant crop.
            disease_label: Module 3 disease label (selects the profile).
            plant_id: plant identifier from Module 2.
            survey_id: survey identifier.

        Returns:
            SeverityResult. Never raises for image-content problems; failures
            produce is_valid=False with score 0.0 / category "low".
        """
        try:
            if crop_rgb is None or crop_rgb.size == 0:
                logger.warning("Empty crop (plant %s)", plant_id)
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "empty image"
                )
            if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
                logger.warning("Expected 3-channel RGB, got shape %s",
                               getattr(crop_rgb, "shape", None))
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "bad channels"
                )
            h, w = crop_rgb.shape[:2]
            if h < 16 or w < 16:
                logger.warning("Crop too small (%dx%d)", w, h)
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "tiny crop"
                )

            # Healthy: by definition no lesion area; skip lesion detection.
            if disease_label == "healthy":
                return SeverityResult(
                    plant_id=plant_id,
                    survey_id=survey_id,
                    disease_label=disease_label,
                    severity_score=0.0,
                    severity_category=CATEGORY_LOW,
                    leaf_px=0,
                    lesion_px=0,
                    is_valid=True,
                    model_version=MODEL_VERSION,
                )

            profile = self._profiles.get(disease_label)
            if profile is None:
                logger.warning("No lesion profile for label %r; "
                               "marking invalid.", disease_label)
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "unknown label"
                )

            leaf_mask, leaf_px = self._segment_leaf(crop_rgb)
            if leaf_px == 0 or leaf_mask is None or not leaf_mask.any():
                logger.warning("No leaf tissue detected (plant %s)", plant_id)
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "no leaf tissue"
                )

            fill_ratio = leaf_px / float(
                max(crop_rgb.shape[0] * crop_rgb.shape[1], 1)
            )
            if fill_ratio < self._leaf_params.min_leaf_fill_ratio:
                logger.warning(
                    "Leaf fill %.3f below min %.3f (plant %s)",
                    fill_ratio, self._leaf_params.min_leaf_fill_ratio, plant_id,
                )
                return self._invalid_result(
                    survey_id, plant_id, disease_label, "low leaf fill"
                )

            _, lesion_px = self._segment_lesions(crop_rgb, leaf_mask, profile)

            score = lesion_px / max(leaf_px, 1)
            score = min(score, 1.0)  # clamp
            score = round(float(score), 4)
            category = self._score_to_category(score)

            return SeverityResult(
                plant_id=plant_id,
                survey_id=survey_id,
                disease_label=disease_label,
                severity_score=score,
                severity_category=category,
                leaf_px=leaf_px,
                lesion_px=lesion_px,
                is_valid=True,
                model_version=MODEL_VERSION,
            )

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Severity estimation failed for plant %s: %s",
                         plant_id, exc, exc_info=True)
            return self._invalid_result(
                survey_id, plant_id, disease_label, f"exception: {exc}"
            )
    def _invalid_result(
        self,
        survey_id: str,
        plant_id: int,
        disease_label: str,
        reason: str,
    ) -> SeverityResult:
        """Build a safe fallback SeverityResult (score 0, category low, invalid)."""
        logger.warning("Severity invalid for plant %s: %s", plant_id, reason)
        return SeverityResult(
            plant_id=plant_id,
            survey_id=survey_id,
            disease_label=disease_label,
            severity_score=0.0,
            severity_category=CATEGORY_LOW,
            leaf_px=0,
            lesion_px=0,
            is_valid=False,
            model_version=MODEL_VERSION,
        )


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: Optional[SeverityEstimator] = None


def get_estimator() -> SeverityEstimator:
    """
    Return the shared SeverityEstimator instance, creating it if needed.

    Lazy singleton (same pattern as app/detection/detector.get_detector and
    app/classification/classifier.get_classifier). Holds only configuration,
    so concurrent/independent estimate() calls are safe.
    """
    global _instance
    if _instance is None:
        _instance = SeverityEstimator()
    return _instance
