"""
FILE PATH: SmartSpray/tests/test_detection.py

PURPOSE:
    Automated test suite for Module 2 — Plant Detection.

    Tests are organized into four groups:
        1. Unit tests for detector.py (model loading and inference)
        2. Unit tests for identity_resolver.py (ID assignment logic)
        3. API tests for router.py (HTTP endpoint behaviour)
        4. Integration tests (detector + resolver + database together)

    Tests that require best.pt (the trained model) are automatically skipped
    if the model has not been trained yet. This allows you to run tests at any
    stage of development without errors.

HOW TO RUN:
    Terminal: your SmartSpray virtual environment
    Folder:   SmartSpray/ (repo root — NOT the tests/ subfolder)

    Run all Module 2 tests:
        pytest tests/test_detection.py -v

    Run only tests that don't need the model (safe before training):
        pytest tests/test_detection.py -v -m "not requires_model"

    Run with detailed output including print statements:
        pytest tests/test_detection.py -v -s

EXPECTED OUTPUT (all tests passing with model):
    tests/test_detection.py::test_raw_detection_dataclass PASSED
    tests/test_detection.py::test_assign_plant_ids_empty PASSED
    tests/test_detection.py::test_assign_plant_ids_sorting PASSED
    tests/test_detection.py::test_assign_plant_ids_custom_start PASSED
    tests/test_detection.py::test_resolve_survey_detections_global_ids PASSED
    tests/test_detection.py::test_resolve_survey_detections_empty PASSED
    tests/test_detection.py::test_detector_raises_on_missing_weights PASSED
    tests/test_detection.py::test_detector_raises_on_missing_image [requires_model] PASSED
    tests/test_detection.py::test_detector_inference_returns_list [requires_model] PASSED
    tests/test_detection.py::test_api_missing_survey PASSED
    tests/test_detection.py::test_api_empty_folder PASSED
    tests/test_detection.py::test_api_model_not_found PASSED
    tests/test_detection.py::test_api_valid_survey [requires_model] PASSED
    tests/test_detection.py::test_database_persistence [requires_model] PASSED
"""

import shutil
import sqlite3
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.detection.detector import RawDetection
from app.detection.identity_resolver import assign_plant_ids, resolve_survey_detections
from app.main import app

# ── Test client ───────────────────────────────────────────────────────────────
client = TestClient(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "yolov8_plant_detector" / "best.pt"
TEST_SURVEY_ID = "test_survey_detection_001"
TEST_IMAGES_DIR = REPO_ROOT / "data" / "images" / TEST_SURVEY_ID

# ── Marker: tests that need the real trained model ────────────────────────────
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="best.pt not found — run the training notebook first (Cell 6)"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def setup_test_survey():
    """
    Create a temporary survey folder with synthetic test images before any test runs.
    Delete it after all tests in this module are complete.
    """
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create 3 small synthetic images (64×64 green rectangles).
    # These are valid images that can be opened by OpenCV/PIL even without
    # being real field photos.
    try:
        import cv2

        for i in range(1, 4):
            img = np.full((64, 64, 3), (34, 139, 34), dtype=np.uint8)  # forest green
            cv2.imwrite(str(TEST_IMAGES_DIR / f"test_plant_{i:03d}.jpg"), img)
    except ImportError:
        # If cv2 is not installed yet, create minimal valid JPEG files
        from PIL import Image
        for i in range(1, 4):
            img = Image.new("RGB", (64, 64), color=(34, 139, 34))
            img.save(str(TEST_IMAGES_DIR / f"test_plant_{i:03d}.jpg"))

    yield  # tests run here

    # Teardown: remove test images and every DB row the tests created
    shutil.rmtree(TEST_IMAGES_DIR, ignore_errors=True)
    from app.db import get_connection
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
            if "survey_id" in cols:
                conn.execute(f"DELETE FROM {table} WHERE survey_id = ?", (TEST_SURVEY_ID,))
        conn.commit()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Unit tests for RawDetection dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestRawDetection:

    def test_raw_detection_dataclass(self):
        """RawDetection stores bbox and confidence correctly."""
        d = RawDetection(bbox=[10, 20, 100, 200], confidence=0.95)
        assert d.bbox == [10, 20, 100, 200]
        assert d.confidence == 0.95

    def test_raw_detection_default_values(self):
        """Creating a RawDetection without arguments should raise TypeError."""
        with pytest.raises(TypeError):
            RawDetection()  # bbox and confidence are required


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Unit tests for identity_resolver.py
# ══════════════════════════════════════════════════════════════════════════════

class TestAssignPlantIds:

    def test_assign_plant_ids_empty(self):
        """Empty input produces empty output without raising."""
        result = assign_plant_ids([])
        assert result == []

    def test_assign_plant_ids_single(self):
        """A single detection gets plant_id=1."""
        dets = [RawDetection(bbox=[10, 10, 100, 100], confidence=0.90)]
        result = assign_plant_ids(dets)
        assert len(result) == 1
        assert result[0]["plant_id"] == 1
        assert result[0]["bbox"] == [10, 10, 100, 100]
        assert result[0]["confidence"] == 0.90

    def test_assign_plant_ids_sorting_top_to_bottom(self):
        """
        Detections should be sorted top-to-bottom.
        The detection with the smaller y_center should get a lower plant_id.
        """
        # Detection A: center at y=200 (lower in image)
        # Detection B: center at y=50  (higher in image — should be plant_id=1)
        det_a = RawDetection(bbox=[50, 150, 150, 250], confidence=0.90)   # y_center=200
        det_b = RawDetection(bbox=[50, 10, 150, 90], confidence=0.85)     # y_center=50

        result = assign_plant_ids([det_a, det_b])
        assert result[0]["plant_id"] == 1
        assert result[0]["bbox"] == det_b.bbox, "Top detection should be plant_id=1"
        assert result[1]["plant_id"] == 2
        assert result[1]["bbox"] == det_a.bbox

    def test_assign_plant_ids_sorting_left_to_right_same_row(self):
        """
        Two detections at the same vertical level should be sorted left-to-right.
        """
        # Both at y≈50 (same row bucket), but A is to the right of B
        det_a = RawDetection(bbox=[300, 10, 400, 90], confidence=0.92)  # x_center=350
        det_b = RawDetection(bbox=[50, 15, 150, 85], confidence=0.88)   # x_center=100

        result = assign_plant_ids([det_a, det_b])
        # B (left) should come first
        assert result[0]["bbox"] == det_b.bbox, "Left detection should be plant_id=1"
        assert result[1]["bbox"] == det_a.bbox

    def test_assign_plant_ids_tilted_high_resolution_rows(self):
        """
        Rows of 180-px plants drifting down by 20 px per plant (camera tilt)
        must still be numbered row by row, left to right.
        """
        dets = []
        for row in range(2):
            for col in range(5):
                x = 50 + col * 220
                y = 100 + row * 400 + col * 20
                dets.append(RawDetection(bbox=[x, y, x + 180, y + 180], confidence=0.9))
        result = assign_plant_ids(list(reversed(dets)))
        xs_row1 = [r["bbox"][0] for r in result[:5]]
        assert xs_row1 == sorted(xs_row1)
        assert all(r["bbox"][1] < 300 for r in result[:5])
        assert all(r["bbox"][1] > 400 for r in result[5:])

    def test_assign_plant_ids_custom_start(self):
        """start_id parameter correctly offsets plant IDs."""
        dets = [
            RawDetection(bbox=[10, 10, 50, 50], confidence=0.9),
            RawDetection(bbox=[100, 10, 150, 50], confidence=0.8),
        ]
        result = assign_plant_ids(dets, start_id=10)
        assert result[0]["plant_id"] == 10
        assert result[1]["plant_id"] == 11

    def test_assign_plant_ids_all_required_keys(self):
        """Each dict in the output must contain plant_id, bbox, and confidence."""
        dets = [RawDetection(bbox=[0, 0, 10, 10], confidence=0.75)]
        result = assign_plant_ids(dets)
        assert "plant_id" in result[0]
        assert "bbox" in result[0]
        assert "confidence" in result[0]


class TestResolveSurveyDetections:

    def test_resolve_survey_detections_global_ids(self):
        """
        Plant IDs must be globally unique across all images in the survey.
        Image 1 has 3 plants → IDs 1,2,3.
        Image 2 has 2 plants → IDs 4,5 (not 1,2 again).
        """
        img1_dets = [
            RawDetection(bbox=[10, 10, 50, 50], confidence=0.9),
            RawDetection(bbox=[60, 10, 100, 50], confidence=0.85),
            RawDetection(bbox=[120, 10, 160, 50], confidence=0.80),
        ]
        img2_dets = [
            RawDetection(bbox=[10, 200, 50, 240], confidence=0.92),
            RawDetection(bbox=[60, 200, 100, 240], confidence=0.88),
        ]

        result = resolve_survey_detections([
            ("image1.jpg", img1_dets),
            ("image2.jpg", img2_dets),
        ])

        assert result["total_plants"] == 5
        all_ids = [d["plant_id"] for d in result["all_detections"]]
        assert sorted(all_ids) == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {sorted(all_ids)}"
        assert len(set(all_ids)) == 5, "All plant_ids must be unique"

    def test_resolve_survey_detections_empty(self):
        """Empty input produces zero plants."""
        result = resolve_survey_detections([])
        assert result["total_plants"] == 0
        assert result["all_detections"] == []

    def test_resolve_survey_detections_image_key(self):
        """Each detection in all_detections must include the source image filename."""
        dets = [RawDetection(bbox=[10, 10, 50, 50], confidence=0.9)]
        result = resolve_survey_detections([("my_photo.jpg", dets)])
        assert result["all_detections"][0]["image"] == "my_photo.jpg"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Unit tests for PlantDetector (model loading)
# ══════════════════════════════════════════════════════════════════════════════

class TestPlantDetector:

    def test_detector_raises_on_missing_weights(self):
        """
        PlantDetector must raise FileNotFoundError with a clear message
        when best.pt does not exist.
        """
        from app.detection.detector import PlantDetector
        fake_path = Path("models/does_not_exist/best.pt")
        with pytest.raises(FileNotFoundError) as exc_info:
            PlantDetector(weights_path=fake_path)
        assert "not found" in str(exc_info.value).lower()

    @requires_model
    def test_detector_loads_successfully(self):
        """PlantDetector loads without error when best.pt exists."""
        from app.detection.detector import PlantDetector
        detector = PlantDetector(weights_path=MODEL_PATH)
        assert detector.is_ready() is True

    @requires_model
    def test_detector_raises_on_missing_image(self):
        """detect() raises FileNotFoundError when the image path doesn't exist."""
        from app.detection.detector import PlantDetector
        detector = PlantDetector(weights_path=MODEL_PATH)
        with pytest.raises(FileNotFoundError):
            detector.detect("data/images/nonexistent.jpg")

    @requires_model
    def test_detector_raises_on_unsupported_format(self):
        """detect() raises ValueError for unsupported file extensions."""
        from app.detection.detector import PlantDetector
        # Create a temporary .txt file
        tmp = REPO_ROOT / "tmp_test_file.txt"
        tmp.write_text("fake")
        try:
            detector = PlantDetector(weights_path=MODEL_PATH)
            with pytest.raises(ValueError):
                detector.detect(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    @requires_model
    def test_detector_inference_returns_list(self):
        """
        detect() returns a list. An image with no plants returns an empty list,
        not an error (our synthetic green images won't contain tomato plants).
        """
        from app.detection.detector import PlantDetector
        test_image = TEST_IMAGES_DIR / "test_plant_001.jpg"
        detector = PlantDetector(weights_path=MODEL_PATH)
        result = detector.detect(test_image)
        assert isinstance(result, list)
        # Each element must be a RawDetection
        for item in result:
            assert isinstance(item, RawDetection)
            assert len(item.bbox) == 4
            assert 0.0 <= item.confidence <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — API endpoint tests for router.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectEndpoint:

    def test_api_missing_survey(self):
        """
        POST /detect with a survey_id that has no folder should return 404.
        """
        response = client.post("/detect", json={"survey_id": "survey_does_not_exist_xyz"})
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_api_empty_folder(self, tmp_path, monkeypatch):
        """
        POST /detect with a survey folder that exists but contains no images
        should return 400.
        """
        # Point IMAGES_ROOT at a temp directory containing an empty survey folder
        empty_survey = tmp_path / "empty_survey_abc"
        empty_survey.mkdir()

        import app.detection.router as dr
        monkeypatch.setattr(dr, "IMAGES_ROOT", tmp_path)

        response = client.post("/detect", json={"survey_id": "empty_survey_abc"})
        assert response.status_code == 400
        assert "no valid images" in response.json()["detail"].lower()

    def test_api_model_not_found(self, monkeypatch):
        """
        POST /detect should return 503 Service Unavailable
        when best.pt doesn't exist.
        """
        import app.detection.detector as det_module

        # Force the singleton to be None so get_detector() tries to reload
        monkeypatch.setattr(det_module, "_instance", None)

        # Make the default weights path point to a nonexistent file
        monkeypatch.setattr(det_module, "DEFAULT_WEIGHTS_PATH", Path("models/nonexistent/best.pt"))

        response = client.post("/detect", json={"survey_id": TEST_SURVEY_ID})
        assert response.status_code == 503
        assert "model" in response.json()["detail"].lower()

    @requires_model
    def test_api_valid_survey(self):
        """
        POST /detect with a valid survey should return 200 and the expected structure.

        Note: our synthetic green images won't contain real tomato plants,
        so plants_detected may be 0. We're testing the API structure, not the model.
        """
        response = client.post("/detect", json={"survey_id": TEST_SURVEY_ID})
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "success"
        assert body["survey_id"] == TEST_SURVEY_ID
        assert body["images_processed"] == 3       # we created 3 test images
        assert isinstance(body["plants_detected"], int)
        assert isinstance(body["detections"], list)

        # Each detection must have the required keys
        for det in body["detections"]:
            assert "plant_id" in det
            assert "image" in det
            assert "bbox" in det
            assert "confidence" in det
            assert len(det["bbox"]) == 4

    @requires_model
    def test_api_detection_health(self):
        """GET /detect/health should return model status."""
        response = client.get("/detect/health")
        assert response.status_code == 200
        body = response.json()
        assert body["module"] == "Plant Detection"
        assert "status" in body

    @requires_model
    def test_database_persistence(self):
        """
        After POST /detect, detections must exist in the SQLite database.
        """
        from app.db import DB_PATH

        # Trigger detection
        response = client.post("/detect", json={"survey_id": TEST_SURVEY_ID})
        assert response.status_code == 200

        # Check the database directly
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE survey_id = ?",
            (TEST_SURVEY_ID,)
        ).fetchone()
        conn.close()

        detected_count = response.json()["plants_detected"]
        assert rows[0] == detected_count, (
            f"API reported {detected_count} plants but DB has {rows[0]} rows"
        )

    @requires_model
    def test_pipeline_integration(self):
        """
        run_pipeline() should correctly invoke Stages 1 and 2.
        Requires the test survey to be registered in the surveys table.
        """
        from app.db import get_connection
        from app.pipeline import run_pipeline

        # Register the test survey in Module 1's table so Stage 1 passes
        conn = get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO surveys (survey_id, capture_date, crop, field_size, num_images) "
            "VALUES (?, ?, ?, ?, ?)",
            (TEST_SURVEY_ID, "2026-01-01", "Tomato", "10m x 10m", 3)
        )
        conn.commit()
        conn.close()

        result = run_pipeline(TEST_SURVEY_ID)

        assert "stage_1_survey" in result
        assert result["stage_1_survey"]["status"] == "found"

        assert "stage_2_detection" in result
        assert result["stage_2_detection"]["status"] in ("success", "skipped")

        # All nine stages are implemented now. The synthetic green squares
        # contain no tomato plants, so later stages legitimately report
        # "skipped" (no input) — they must never be missing or crash the run.
        ok = ("success", "skipped")
        for key in ("stage_3_classification", "stage_4_severity",
                    "stage_5_spatial_mapping", "stage_6_treatment_decision",
                    "stage_7_resource_optimization", "stage_8_mission_planning",
                    "stage_9_drone_commands"):
            assert key in result
            assert result[key]["status"] in ok, (key, result[key])



# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — OpenCV fallback detector (opt-in, used for the synthetic demo)
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackDetector:

    def _field(self):
        import cv2
        img = np.full((400, 600, 3), (60, 90, 130), dtype=np.uint8)   # soil (BGR)
        centres = [(100, 100), (300, 100), (500, 100), (200, 300), (400, 300)]
        for cx, cy in centres:
            cv2.circle(img, (cx, cy), 40, (40, 160, 50), -1)            # green plant
        return img, centres

    def test_detects_each_plant_once(self):
        from app.detection.detector import ClassicalPlantDetector
        img, centres = self._field()
        dets = ClassicalPlantDetector().detect_array(img)
        assert len(dets) == len(centres)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1 and y2 > y1
            assert abs((x2 - x1) - (y2 - y1)) <= 2          # square crops
            assert 0.0 < d.confidence <= 1.0

    def test_bare_soil_gives_no_detections(self):
        from app.detection.detector import ClassicalPlantDetector
        soil = np.full((300, 300, 3), (60, 90, 130), dtype=np.uint8)
        assert ClassicalPlantDetector().detect_array(soil) == []

    def test_get_detector_modes(self, monkeypatch):
        import app.detection.detector as det_module
        monkeypatch.setattr(det_module, "_instance", None)
        monkeypatch.setattr(det_module, "DEFAULT_WEIGHTS_PATH", Path("models/nonexistent/best.pt"))
        assert det_module.get_detector("classical").model_version == det_module.FALLBACK_MODEL_VERSION
        assert det_module.get_detector("auto").model_version == det_module.FALLBACK_MODEL_VERSION
        with pytest.raises(FileNotFoundError):
            det_module.get_detector("yolo")
        with pytest.raises(ValueError):
            det_module.get_detector("magic")

    def test_api_classical_mode_on_synthetic_images(self, monkeypatch):
        import app.detection.detector as det_module
        monkeypatch.setattr(det_module, "_instance", None)
        response = client.post("/detect", json={"survey_id": TEST_SURVEY_ID,
                                                "detector": "classical"})
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == det_module.FALLBACK_MODEL_VERSION
        assert body["images_processed"] == 3
