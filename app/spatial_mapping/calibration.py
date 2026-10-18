"""
FILE PATH: SmartSpray/app/spatial_mapping/calibration.py

PURPOSE:
    The ArUco + homography calibration engine for Module 5 - Spatial Mapping.

    It turns a survey image that contains four ArUco markers placed at the four
    corners of the field into a 3x3 homography matrix H that maps PIXEL
    coordinates to FIELD coordinates in metres.

    Pipeline (mirrors the Module 5 box in database/schema.sql):
        1. Build the ArUco dictionary (default DICT_4X4_50).
        2. Detect markers -> per-marker pixel centres.
        3. Match detected marker IDs to their known field-corner positions.
        4. cv2.findHomography() -> H (pixel -> field) + reprojection error.

MARKER LAYOUT ASSUMPTION (must be honoured when the markers are placed):
    Exactly four markers lie flat on the ground, one at each field corner,
    using these marker IDs:
        ID 0 -> (0, 0)                              origin corner
        ID 1 -> (field_width_m, 0)
        ID 2 -> (field_width_m, field_height_m)
        ID 3 -> (0, field_height_m)
    The correspondence point used for each marker is its CENTRE (the mean of
    its four detected corners). Marker centres are therefore assumed to sit
    exactly on the physical field corners.

KNOWN LIMITATION (verified by inspection - see
rules/severity/stage3_correction_fraction_report.md):
    The imagery currently in data/images/ contains NO ArUco markers, and there
    are no GPS / camera-calibration records anywhere in the repository. This
    module implements the specified ArUco path faithfully, but until survey
    images are re-acquired with markers placed at the field corners it cannot
    produce a real calibration from the existing data. mapper.py therefore
    falls back to an explicit "uncalibrated" mode that stores a NULL field
    position instead of inventing coordinates.

DEPENDENCIES:
    - cv2 (opencv-python). IMPORTANT: OpenCV >= 5.0 REMOVED the old functional
      cv2.aruco.detectMarkers() + cv2.aruco.DetectorParameters_create() API.
      This module uses the cv2.aruco.ArucoDetector class (the >= 5.0 API).
    - numpy
    - json / logging / dataclasses / datetime / pathlib / typing (stdlib)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Named logger for this module.
logger = logging.getLogger("spatial_mapping.calibration")

# Default ArUco dictionary. 4x4 markers are large enough to be detected from a
# drone frame while still fitting comfortably on a printed corner tile.
DEFAULT_ARUCO_DICT = "DICT_4X4_50"

# A homography needs at least 4 point correspondences.
MIN_MARKERS = 4

# RANSAC reprojection threshold (pixels) used when more than 4 markers exist.
DEFAULT_RANSAC_THRESHOLD_PX = 5.0

# Calibrations whose mean reprojection error exceeds this are flagged
# is_valid=False. 4-marker solves are exact (error ~0), so this only bites on
# multi-marker solves with a mis-placed marker.
MAX_REPROJECTION_ERROR_PX = 25.0

# Marker IDs reserved for the four field corners, in opposite-corner order.
FIELD_CORNER_IDS = (0, 1, 2, 3)

# OpenCV >= 5.0 API probe. If this is falsy the module can still be imported
# (so the rest of the app boots) but detection will raise a clear error.
_ARUCO_DETECTOR_AVAILABLE = bool(
    getattr(getattr(cv2, "aruco", None), "ArucoDetector", None)
)

def make_calibration_id(survey_id: str, when: Optional[datetime] = None) -> str:
    """
    Build a unique, sortable calibration identifier.

    Format: cal_<survey_id>_<YYYYmmddTHHMMSSZ>  (UTC)
    Example: cal_survey_001_20260714T101530Z
    """
    stamp = when if when is not None else datetime.now(timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    return "cal_%s_%s" % (survey_id, stamp.strftime("%Y%m%dT%H%M%SZ"))


def build_field_corner_points(
    field_width_m: float,
    field_height_m: float,
) -> dict:
    """
    Return the known FIELD (metre) position of each corner marker centre.

    Args:
        field_width_m: physical field width (X extent) in metres, > 0.
        field_height_m: physical field height (Y extent) in metres, > 0.

    Returns:
        {marker_id: (x_m, y_m)} for IDs 0..3 (see MARKER LAYOUT ASSUMPTION).

    Raises:
        ValueError: if either dimension is not a positive number.
    """
    width = float(field_width_m)
    height = float(field_height_m)
    if not (width > 0.0):
        raise ValueError("field_width_m must be a positive number of metres")
    if not (height > 0.0):
        raise ValueError("field_height_m must be a positive number of metres")
    return {
        0: (0.0, 0.0),
        1: (width, 0.0),
        2: (width, height),
        3: (0.0, height),
    }


def homography_from_json(payload: str) -> np.ndarray:
    """
    Rebuild a 3x3 float64 homography from the JSON stored in
    spatial_calibrations.homography_json.

    Raises:
        ValueError: if the payload is not a 3x3 numeric matrix.
    """
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("homography_json is not valid JSON: %s" % exc) from exc
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            "homography_json must decode to a 3x3 matrix, got shape %s"
            % (matrix.shape,)
        )
    return matrix


def _resolve_dictionary(aruco_dict: str):
    """
    Resolve an ArUco dictionary NAME (e.g. "DICT_4X4_50") to a cv2 dictionary
    object using the OpenCV >= 5.0 API.

    Raises:
        RuntimeError: if this OpenCV build has no ArUco support.
        ValueError: if the name is empty or not a known predefined dictionary.
    """
    if not _ARUCO_DETECTOR_AVAILABLE:
        raise RuntimeError(
            "This OpenCV build has no cv2.aruco.ArucoDetector (OpenCV >= 5.0 "
            "API). ArUco calibration is unavailable. Found cv2 version %s."
            % cv2.__version__
        )
    if not isinstance(aruco_dict, str) or not aruco_dict.strip():
        raise ValueError(
            "aruco_dict must be a non-empty string, e.g. 'DICT_4X4_50'"
        )
    name = aruco_dict.strip()
    dict_id = getattr(cv2.aruco, name, None)
    if not isinstance(dict_id, int):
        raise ValueError(
            "Unknown ArUco dictionary '%s'. Use a predefined cv2.aruco name "
            "such as 'DICT_4X4_50'." % name
        )
    return cv2.aruco.getPredefinedDictionary(dict_id)

@dataclass
class MarkerDetection:
    """One detected ArUco marker in image (pixel) space.

    Attributes:
        marker_id: decoded ArUco marker id.
        corners: (4, 2) float64 array of corner pixels, ArUco order
            (top-left, top-right, bottom-right, bottom-left).
        center: (x_px, y_px) mean of the four corners.
    """

    marker_id: int
    corners: np.ndarray
    center: tuple

    def as_dict(self) -> dict:
        """Return a JSON-friendly view of this detection."""
        return {
            "marker_id": int(self.marker_id),
            "center_px": [float(self.center[0]), float(self.center[1])],
            "corners_px": self.corners.tolist(),
        }


@dataclass
class CalibrationResult:
    """Outcome of calibrating one survey image.

    Attributes:
        homography: 3x3 float64 pixel->field matrix, or None if unsolved.
        homography_json: JSON serialisation stored in the database.
        reprojection_error: mean pixel error of the fitted correspondences.
        is_valid: True when the solve succeeded AND the reprojection error is
            within the configured tolerance.
        notes: human-readable remarks (skipped markers, tolerance breaches).
    """

    calibration_id: str
    survey_id: str
    image_filename: str
    field_width_m: float
    field_height_m: float
    aruco_dict: str
    marker_count: int
    homography: Optional[np.ndarray]
    homography_json: str
    reprojection_error: Optional[float]
    is_valid: bool
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a JSON-friendly view of this calibration."""
        return {
            "calibration_id": self.calibration_id,
            "survey_id": self.survey_id,
            "image_filename": self.image_filename,
            "field_width_m": self.field_width_m,
            "field_height_m": self.field_height_m,
            "aruco_dict": self.aruco_dict,
            "marker_count": self.marker_count,
            "homography": (
                self.homography.tolist()
                if self.homography is not None
                else None
            ),
            "reprojection_error": self.reprojection_error,
            "is_valid": self.is_valid,
            "notes": list(self.notes),
        }

class ArUcoCalibrator:
    """Detects corner ArUco markers and solves the pixel->field homography.

    This class is stateless apart from its OpenCV dictionary/detector, so a
    single instance can safely serve every survey.
    """

    def __init__(
        self,
        aruco_dict: str = DEFAULT_ARUCO_DICT,
        min_markers: int = MIN_MARKERS,
        max_reprojection_error_px: float = MAX_REPROJECTION_ERROR_PX,
        ransac_threshold_px: float = DEFAULT_RANSAC_THRESHOLD_PX,
    ) -> None:
        """
        Args:
            aruco_dict: predefined cv2.aruco dictionary name.
            min_markers: minimum usable corner markers (>= 4).
            max_reprojection_error_px: tolerance for is_valid.
            ransac_threshold_px: RANSAC inlier threshold when > 4 markers.
        """
        self.aruco_dict = aruco_dict
        self.min_markers = max(MIN_MARKERS, int(min_markers))
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.ransac_threshold_px = float(ransac_threshold_px)
        self._dictionary = _resolve_dictionary(aruco_dict)
        self._detector = cv2.aruco.ArucoDetector(
            self._dictionary, cv2.aruco.DetectorParameters()
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert an input image (grey / BGR / BGRA) to single-channel grey."""
        if image is None:
            raise ValueError("image must not be None")
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("image must not be empty")
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] == 3:
            return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.ndim == 3 and array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        raise ValueError(
            "Unsupported image shape %s; expected (H, W), (H, W, 3) or (H, W, 4)"
            % (array.shape,)
        )

    def detect_markers(self, image: np.ndarray) -> list:
        """
        Detect every ArUco marker in an image.

        Args:
            image: grey, BGR or BGRA numpy array.

        Returns:
            MarkerDetection list sorted by marker_id (OpenCV does NOT return
            markers in id order, so sorting makes results deterministic).
        """
        gray = self._to_gray(image)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        detections = []
        if ids is None:
            return detections
        for corner_set, marker_id in zip(corners, ids.ravel()):
            points = np.asarray(corner_set, dtype=np.float64).reshape(4, 2)
            center = (
                float(points[:, 0].mean()),
                float(points[:, 1].mean()),
            )
            detections.append(
                MarkerDetection(int(marker_id), points, center)
            )
        detections.sort(key=lambda d: d.marker_id)
        return detections

    def compute_homography(
        self,
        pixel_points: list,
        field_points: list,
    ) -> tuple:
        """
        Solve H such that field_point ~= H @ pixel_point.

        Uses a direct 4-point solve when exactly 4 correspondences are given
        (exact and deterministic) and RANSAC when more are available.

        Args:
            pixel_points: (x_px, y_px) pairs.
            field_points: matching (x_m, y_m) pairs, same length/order.

        Returns:
            (homography (3, 3) float64, mean reprojection error in pixels).

        Raises:
            ValueError: fewer than 4 correspondences, mismatched lengths, or a
                degenerate layout from which no homography can be estimated.
        """
        src = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(field_points, dtype=np.float64).reshape(-1, 2)
        if src.shape != dst.shape:
            raise ValueError(
                "pixel_points %s and field_points %s must have the same shape"
                % (src.shape, dst.shape)
            )
        count = src.shape[0]
        if count < MIN_MARKERS:
            raise ValueError(
                "At least %d correspondences are required to solve a "
                "homography, got %d" % (MIN_MARKERS, count)
            )
        if count == MIN_MARKERS:
            homography, _mask = cv2.findHomography(src, dst, 0)
        else:
            homography, _mask = cv2.findHomography(
                src, dst, cv2.RANSAC, self.ransac_threshold_px
            )
        if homography is None:
            raise ValueError(
                "cv2.findHomography returned no solution; the marker layout "
                "is degenerate (e.g. all markers collinear)"
            )
        error = self._reprojection_error(homography, src, dst)
        return homography, error

    @staticmethod
    def _reprojection_error(
        homography: np.ndarray,
        pixel_points: np.ndarray,
        field_points: np.ndarray,
    ) -> float:
        """
        Mean pixel distance between the detected marker centres and where the
        known field points project back into the image.

        The field points are projected with the INVERSE homography (field ->
        pixel) because the fitted matrix maps pixel -> field.
        """
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Homography is singular and cannot be inverted: %s" % exc
            ) from exc
        field_pts = field_points.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(field_pts, inverse).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - pixel_points, axis=1)))

    def calibrate(
        self,
        survey_id: str,
        image_filename: str,
        image: np.ndarray,
        field_width_m: float,
        field_height_m: float,
    ) -> CalibrationResult:
        """
        Calibrate a survey from an in-memory image containing corner markers.

        Args:
            survey_id: survey the calibration belongs to.
            image_filename: name of the calibration image (provenance).
            image: grey/BGR/BGRA image holding the corner ArUco markers.
            field_width_m: physical field width in metres (> 0).
            field_height_m: physical field height in metres (> 0).

        Returns:
            CalibrationResult with the solved homography.

        Raises:
            ValueError: if survey_id is empty, the field size is invalid, or
                fewer than self.min_markers usable corner markers were found.
        """
        if not isinstance(survey_id, str) or not survey_id.strip():
            raise ValueError("survey_id must be a non-empty string")
        survey_id = survey_id.strip()

        corner_map = build_field_corner_points(field_width_m, field_height_m)
        detections = self.detect_markers(image)

        usable = [d for d in detections if d.marker_id in corner_map]
        notes = []
        ignored = len(detections) - len(usable)
        if ignored:
            notes.append(
                "Ignored %d detected marker(s) whose ids are outside the "
                "field-corner set %s." % (ignored, list(FIELD_CORNER_IDS))
            )

        if len(usable) < self.min_markers:
            raise ValueError(
                "Need at least %d ArUco markers with ids %s printed at the "
                "field corners, but found %d usable marker(s) (%d detected in "
                "total) in image '%s'. Refusing to invent field coordinates."
                % (
                    self.min_markers,
                    list(FIELD_CORNER_IDS),
                    len(usable),
                    len(detections),
                    image_filename,
                )
            )

        pixel_points = [d.center for d in usable]
        field_points = [corner_map[d.marker_id] for d in usable]
        homography, error = self.compute_homography(pixel_points, field_points)

        valid = error <= self.max_reprojection_error_px
        if not valid:
            notes.append(
                "Reprojection error %.3f px exceeds the %.3f px tolerance; "
                "calibration flagged invalid."
                % (error, self.max_reprojection_error_px)
            )

        logger.info(
            "Calibrated survey '%s' from '%s': %d markers, "
            "reprojection error %.3f px, valid=%s",
            survey_id,
            image_filename,
            len(usable),
            error,
            valid,
        )

        return CalibrationResult(
            calibration_id=make_calibration_id(survey_id),
            survey_id=survey_id,
            image_filename=image_filename,
            field_width_m=float(field_width_m),
            field_height_m=float(field_height_m),
            aruco_dict=self.aruco_dict,
            marker_count=len(usable),
            homography=homography,
            homography_json=json.dumps(homography.tolist()),
            reprojection_error=error,
            is_valid=valid,
            notes=notes,
        )

    def calibrate_from_file(
        self,
        survey_id: str,
        image_filename: str,
        image_path,
        field_width_m: float,
        field_height_m: float,
    ) -> CalibrationResult:
        """
        Same as calibrate(), but loads the calibration image from disk.

        Raises:
            FileNotFoundError: if the image file does not exist.
            ValueError: if the file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError("Calibration image not found: %s" % path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Failed to load calibration image: %s" % path)
        return self.calibrate(
            survey_id=survey_id,
            image_filename=image_filename,
            image=image,
            field_width_m=field_width_m,
            field_height_m=field_height_m,
        )

# ── Singleton accessor ────────────────────────────────────────────────────────

_calibrator: Optional[ArUcoCalibrator] = None


def get_calibrator() -> ArUcoCalibrator:
    """
    Return the process-wide ArUcoCalibrator, creating it on first use.

    Mirrors the singleton pattern used by Module 2 (get_detector), Module 3
    (get_classifier) and Module 4 (get_estimator), so the OpenCV dictionary and
    detector are built once per process rather than per request.
    """
    global _calibrator
    if _calibrator is None:
        _calibrator = ArUcoCalibrator()
        logger.info(
            "ArUcoCalibrator initialised (dict=%s, min_markers=%d)",
            _calibrator.aruco_dict,
            _calibrator.min_markers,
        )
    return _calibrator
