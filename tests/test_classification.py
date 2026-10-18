"""
FILE PATH: SmartSpray/tests/test_classification.py

PURPOSE:
    Automated test suite for Module 3 — Disease Classification.

    Tests are organized into five groups:
        1. Unit tests for prediction.py (dataclasses)
        2. Unit tests for classifier.py (model loading and inference)
        3. Unit tests for database.py (SQLite helpers)
        4. Unit tests for service.py (orchestration)
        5. API tests for router.py (HTTP endpoint behaviour)

    Tests that require the actual trained model (best.keras) are automatically
    skipped if the model has not been trained yet. Most tests mock the model
    to avoid this dependency.

HOW TO RUN:
    Terminal: your SmartSpray virtual environment
    Folder:   SmartSpray/ (repo root — NOT the tests/ subfolder)

    Run all Module 3 tests:
        pytest tests/test_classification.py -v

    Run only tests that don't need the model (safe before training):
        pytest tests/test_classification.py -v -m "not requires_model"

    Run with detailed output:
        pytest tests/test_classification.py -v -s

EXPECTED OUTPUT (all tests passing with model):
    tests/test_classification.py::TestPredictionDataclasses::test_crop_input_construction PASSED
    tests/test_classification.py::TestPredictionDataclasses::test_classification_result_construction PASSED
    tests/test_classification.py::TestDiseaseClassifier::test_classifier_raises_on_missing_weights PASSED
    tests/test_classification.py::TestDiseaseClassifier::test_classifier_raises_on_missing_class_order PASSED
    tests/test_classification.py::TestDiseaseClassifier::test_classifier_predict_structure [requires_model] PASSED
    tests/test_classification.py::TestDatabaseHelpers::test_ensure_classifications_table_creates PASSED
    tests/test_classification.py::TestDatabaseHelpers::test_save_and_fetch_classifications PASSED
    tests/test_classification.py::TestDatabaseHelpers::test_delete_classifications_for_survey PASSED
    tests/test_classification.py::TestClassifyService::test_classify_survey_happy_path PASSED
    tests/test_classification.py::TestClassifyEndpoint::test_api_model_not_found PASSED
    tests/test_classification.py::TestClassifyEndpoint::test_api_invalid_body PASSED
    tests/test_classification.py::TestClassifyEndpoint::test_api_health PASSED
"""

import json
import sqlite3
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.classification.prediction import (
    CropInput,
    ClassificationResult,
    CLASS_NAMES,
    MODEL_VERSION,
)
from app.classification.database import (
    ensure_classifications_table,
    fetch_detections_for_survey,
    delete_classifications_for_survey,
    save_classification,
    fetch_classifications_for_survey,
)
from app.main import app

# ── Test client ───────────────────────────────────────────────────────────────
client = TestClient(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "classification" / "best.keras"
CLASS_ORDER_PATH = REPO_ROOT / "models" / "classification" / "class_order.json"

# ── Marker: tests that need the real trained model ────────────────────────────
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="best.keras not found — run the training notebook first (Cell 13)"
)

# ── Test survey constants ─────────────────────────────────────────────────────
TEST_SURVEY_ID = "test_survey_classification_001"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_survey():
    """
    Create a temporary survey image directory with test images before tests.
    Delete it after all tests in this module are complete.
    """
    test_images_dir = REPO_ROOT / "data" / "images" / TEST_SURVEY_ID
    test_images_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 synthetic test images (64x64 green rectangles).
    for i in range(1, 3):
        img = np.full((64, 64, 3), (34, 139, 34), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(test_images_dir / f"test_plant_{i:03d}.jpg"), img)

    yield

    # Teardown: remove test images and directory.
    import shutil
    shutil.rmtree(test_images_dir, ignore_errors=True)


@pytest.fixture
def in_memory_db():
    """
    Create an in-memory SQLite database with the detections table
    (Module 2's schema) and the classifications table (Module 3's schema).
    Yields the connection; the caller is responsible for closing it.
    """
    conn = sqlite3.connect(":memory:")

    # Create Module 2's detections table (exact schema from app/detection/router.py).
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

    # Create Module 3's classifications table.
    ensure_classifications_table(conn)

    yield conn
    conn.close()


@pytest.fixture
def sample_result() -> ClassificationResult:
    """
    Return a ClassificationResult with known values for database tests.
    """
    return ClassificationResult(
        survey_id="test_survey_001",
        image_filename="IMG_001.jpg",
        plant_id=1,
        bbox=[10.0, 20.0, 100.0, 200.0],
        label="early_blight",
        confidence=0.9543,
        all_probs={
            "early_blight": 0.9543,
            "healthy": 0.0210,
            "late_blight": 0.0150,
            "septoria_leaf_spot": 0.0097,
        },
        model_version="test-model-v1",
    )


@pytest.fixture
def seeded_in_memory_db(in_memory_db):
    """
    An in-memory database pre-seeded with detection rows for TEST_SURVEY_ID.
    """
    conn = in_memory_db

    # Insert synthetic detections matching TEST_SURVEY_ID.
    detections_data = [
        (TEST_SURVEY_ID, "test_plant_001.jpg", 1, 10.0, 10.0, 50.0, 50.0, 0.95, "test-model"),
        (TEST_SURVEY_ID, "test_plant_002.jpg", 2, 10.0, 10.0, 50.0, 50.0, 0.92, "test-model"),
    ]
    conn.executemany(
        """
        INSERT INTO detections
            (survey_id, image_filename, plant_id,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             confidence, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        detections_data,
    )
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Unit tests for prediction.py (dataclasses)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionDataclasses:

    def test_crop_input_construction(self):
        """CropInput stores all fields correctly."""
        pixels = np.ones((224, 224, 3), dtype=np.uint8) * 128
        crop = CropInput(
            survey_id="survey_001",
            image_filename="IMG_001.jpg",
            plant_id=1,
            bbox=[10, 20, 100, 200],
            pixels=pixels,
        )
        assert crop.survey_id == "survey_001"
        assert crop.image_filename == "IMG_001.jpg"
        assert crop.plant_id == 1
        assert crop.bbox == [10, 20, 100, 200]
        assert crop.pixels.shape == (224, 224, 3)
        assert np.allclose(crop.pixels, 128)

    def test_classification_result_construction(self):
        """ClassificationResult stores all fields and defaults model_version."""
        result = ClassificationResult(
            survey_id="survey_001",
            image_filename="IMG_001.jpg",
            plant_id=1,
            bbox=[10.0, 20.0, 100.0, 200.0],
            label="early_blight",
            confidence=0.9543,
            all_probs={
                "early_blight": 0.9543,
                "healthy": 0.0210,
                "late_blight": 0.0150,
                "septoria_leaf_spot": 0.0097,
            },
        )
        assert result.survey_id == "survey_001"
        assert result.label == "early_blight"
        assert result.confidence == 0.9543
        assert result.model_version == MODEL_VERSION  # Uses the default

    def test_classification_result_custom_model_version(self):
        """model_version can be overridden."""
        result = ClassificationResult(
            survey_id="survey_001",
            image_filename="IMG_001.jpg",
            plant_id=1,
            bbox=[10.0, 20.0, 100.0, 200.0],
            label="healthy",
            confidence=0.9876,
            all_probs={"early_blight": 0.01, "healthy": 0.99, "late_blight": 0.0, "septoria_leaf_spot": 0.0},
            model_version="custom-v2",
        )
        assert result.model_version == "custom-v2"

    def test_classification_result_missing_all_probs_key(self):
        """Missing or extra keys in all_probs should not raise — it's a dict."""
        result = ClassificationResult(
            survey_id="survey_001",
            image_filename="IMG_001.jpg",
            plant_id=1,
            bbox=[0.0, 0.0, 10.0, 10.0],
            label="late_blight",
            confidence=0.85,
            all_probs={"late_blight": 0.85, "healthy": 0.15},
        )
        assert "late_blight" in result.all_probs
        assert result.all_probs["late_blight"] == 0.85


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Unit tests for classifier.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDiseaseClassifier:

    def test_classifier_raises_on_missing_weights(self):
        """
        DiseaseClassifier must raise FileNotFoundError with a clear message
        when best.keras does not exist.
        """
        from app.classification.classifier import DiseaseClassifier

        fake_path = Path("models/does_not_exist/best.keras")
        with pytest.raises(FileNotFoundError) as exc_info:
            DiseaseClassifier(weights_path=fake_path)
        assert "not found" in str(exc_info.value).lower()

    def test_classifier_raises_on_missing_class_order(self, tmp_path, monkeypatch):
        """
        DiseaseClassifier must raise FileNotFoundError when class_order.json
        does not exist, even if the weights file is present.
        """
        from app.classification.classifier import DiseaseClassifier, CLASS_ORDER_PATH

        # Point CLASS_ORDER_PATH at a non-existent file.
        fake_order_path = tmp_path / "nonexistent_class_order.json"
        monkeypatch.setattr(
            "app.classification.classifier.CLASS_ORDER_PATH", fake_order_path
        )

        # Create a dummy weights file so the first check passes.
        dummy_weights = tmp_path / "best.keras"
        dummy_weights.write_text("fake model data")

        with pytest.raises(FileNotFoundError) as exc_info:
            DiseaseClassifier(weights_path=dummy_weights)
        assert "class order" in str(exc_info.value).lower()

    @requires_model
    def test_classifier_loads_successfully(self):
        """
        DiseaseClassifier loads without error when best.keras and
        class_order.json exist and are valid.
        """
        from app.classification.classifier import DiseaseClassifier
        classifier = DiseaseClassifier(weights_path=MODEL_PATH)
        assert classifier.is_ready() is True
        assert classifier.class_order == CLASS_NAMES

    def test_classifier_predict_structure_mocked(self, tmp_path, monkeypatch):
        """
        DiseaseClassifier.predict() returns the correct dict structure
        when the model is a mock.
        """
        from app.classification.classifier import DiseaseClassifier

        # Create a valid class_order.json in a temporary directory.
        temp_class_order = tmp_path / "class_order.json"
        with open(temp_class_order, "w") as f:
            json.dump({"class_order": CLASS_NAMES}, f)
        monkeypatch.setattr(
            "app.classification.classifier.CLASS_ORDER_PATH", temp_class_order
        )

        # Create a fake .keras file (minimal content, just for exists() check).
        fake_weights = tmp_path / "best.keras"
        fake_weights.write_text("fake model")

        # Build a fake Keras model that returns known probabilities.
        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([[0.90, 0.03, 0.05, 0.02]])

        # Mock tf.keras.models.load_model to return the fake model.
        with patch("app.classification.classifier.tf.keras.models.load_model",
                   return_value=fake_model):
            classifier = DiseaseClassifier(weights_path=fake_weights)

            # Create a synthetic BGR image.
            crop_bgr = np.ones((224, 224, 3), dtype=np.uint8) * 128

            result = classifier.predict(crop_bgr)

        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "all_probs" in result
        assert result["label"] == "early_blight"
        assert result["confidence"] == 0.9
        assert abs(sum(result["all_probs"].values()) - 1.0) < 0.01
        assert result["all_probs"]["early_blight"] == 0.9


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Unit tests for database.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseHelpers:

    def test_ensure_classifications_table_creates(self, in_memory_db):
        """
        ensure_classifications_table() creates the table on first call
        and does not error on subsequent calls.
        """
        conn = in_memory_db

        # Table should already exist (created by the fixture).
        # Verify by querying its schema.
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'"
        )
        assert cursor.fetchone() is not None

        # Calling it again should not raise.
        ensure_classifications_table(conn)

    def test_save_and_fetch_classifications(self, in_memory_db, sample_result):
        """
        A ClassificationResult can be saved and then retrieved with
        matching field values.
        """
        conn = in_memory_db

        # Save the result.
        save_classification(conn, sample_result)

        # Fetch it back.
        results = fetch_classifications_for_survey(conn, sample_result.survey_id)

        assert len(results) == 1
        fetched = results[0]
        assert fetched["survey_id"] == sample_result.survey_id  # Not in output but verified
        assert fetched["plant_id"] == sample_result.plant_id
        assert fetched["label"] == sample_result.label
        assert fetched["confidence"] == sample_result.confidence
        assert fetched["bbox"] == sample_result.bbox
        assert fetched["all_probs"] == sample_result.all_probs
        assert fetched["model_version"] == sample_result.model_version

    def test_delete_classifications_for_survey(self, in_memory_db, sample_result):
        """
        delete_classifications_for_survey() removes rows for the specified survey
        but keeps rows for other surveys.
        """
        conn = in_memory_db

        # Create results for two surveys.
        result_a = sample_result
        result_b = ClassificationResult(
            survey_id="other_survey",
            image_filename="other.jpg",
            plant_id=1,
            bbox=[0.0, 0.0, 10.0, 10.0],
            label="healthy",
            confidence=0.99,
            all_probs={"early_blight": 0.01, "healthy": 0.99, "late_blight": 0.0, "septoria_leaf_spot": 0.0},
        )

        save_classification(conn, result_a)
        save_classification(conn, result_b)

        # Delete only result_a's survey.
        delete_classifications_for_survey(conn, result_a.survey_id)

        # Verify result_a is gone.
        results_a = fetch_classifications_for_survey(conn, result_a.survey_id)
        assert len(results_a) == 0

        # Verify result_b is still present.
        results_b = fetch_classifications_for_survey(conn, result_b.survey_id)
        assert len(results_b) == 1
        assert results_b[0]["label"] == "healthy"

    def test_fetch_detections_for_survey(self, seeded_in_memory_db):
        """
        fetch_detections_for_survey() returns the correct number of detections
        with the expected fields.
        """
        conn = seeded_in_memory_db

        detections = fetch_detections_for_survey(conn, TEST_SURVEY_ID)
        assert len(detections) == 2
        assert detections[0]["plant_id"] == 1
        assert detections[1]["plant_id"] == 2
        assert "image_filename" in detections[0]
        assert "bbox_x1" in detections[0]

    def test_fetch_detections_for_survey_empty(self, in_memory_db):
        """
        fetch_detections_for_survey() returns an empty list when no detections
        exist for the given survey.
        """
        conn = in_memory_db
        detections = fetch_detections_for_survey(conn, "nonexistent_survey")
        assert detections == []


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Unit tests for service.py (orchestration)
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyService:

    def test_classify_survey_happy_path(self, monkeypatch, seeded_in_memory_db, tmp_path):
        """
        classify_survey() should correctly orchestrate the full flow:
        read detections -> load images -> classify -> save results.
        This test mocks the classifier, database connection, and image loading.
        """
        from app.classification.service import classify_survey

        # ── Step 1: Mock the database connection ──────────────────────────────
        # Replace app.db.get_connection() to return our seeded in-memory DB.
        monkeypatch.setattr(
            "app.classification.service.get_connection",
            lambda: seeded_in_memory_db,
        )

        # Ensure the images root points to the real data/images directory
        # (the setup fixture already created test images there).
        monkeypatch.setattr(
            "app.classification.service.IMAGES_ROOT",
            REPO_ROOT / "data" / "images",
        )

        # ── Step 2: Mock the classifier ──────────────────────────────────────
        # Replace get_classifier() with a mock that returns known predictions.
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = {
            "label": "early_blight",
            "confidence": 0.9543,
            "all_probs": {
                "early_blight": 0.9543,
                "healthy": 0.0210,
                "late_blight": 0.0150,
                "septoria_leaf_spot": 0.0097,
            },
        }
        monkeypatch.setattr(
            "app.classification.service.get_classifier",
            lambda: mock_classifier,
        )

        # ── Step 3: Run classification ───────────────────────────────────────
        result = classify_survey(TEST_SURVEY_ID)

        # ── Step 4: Verify the result structure ──────────────────────────────
        assert result["status"] == "success"
        assert result["survey_id"] == TEST_SURVEY_ID
        assert result["total_plants"] == 2
        assert result["classified"] == 2
        assert result["failed"] == 0

        # Verify per-plant results.
        assert len(result["results"]) == 2

        # Plant 1.
        assert result["results"][0]["plant_id"] == 1
        assert result["results"][0]["label"] == "early_blight"
        assert result["results"][0]["confidence"] == 0.9543
        assert "error" not in result["results"][0]

        # Plant 2.
        assert result["results"][1]["plant_id"] == 2
        assert result["results"][1]["label"] == "early_blight"

        # ── Step 5: Verify results were persisted in the database ────────────
        saved = fetch_classifications_for_survey(seeded_in_memory_db, TEST_SURVEY_ID)
        assert len(saved) == 2
        assert saved[0]["label"] == "early_blight"

    def test_classify_survey_no_detections(self, monkeypatch, in_memory_db):
        """
        classify_survey() must raise ValueError when the survey has no
        plant detections in the database.
        """
        from app.classification.service import classify_survey

        monkeypatch.setattr(
            "app.classification.service.get_connection",
            lambda: in_memory_db,
        )

        with pytest.raises(ValueError) as exc_info:
            classify_survey("survey_with_no_detections")
        assert "no plant detections" in str(exc_info.value).lower()

    def test_classify_survey_missing_image(
        self,
        monkeypatch,
        seeded_in_memory_db,
        tmp_path,
    ):
        """
        classify_survey() should handle missing image files gracefully by
        marking the plant as failed and continuing with the next plant.
        """
        from app.classification.service import classify_survey

        monkeypatch.setattr(
            "app.classification.service.get_connection",
            lambda: seeded_in_memory_db,
        )

        # Point IMAGES_ROOT at a directory that doesn't have test images.
        empty_dir = tmp_path
        monkeypatch.setattr(
            "app.classification.service.IMAGES_ROOT",
            empty_dir,
        )

        mock_classifier = MagicMock()
        monkeypatch.setattr(
            "app.classification.service.get_classifier",
            lambda: mock_classifier,
        )

        result = classify_survey(TEST_SURVEY_ID)

        # Both plants should fail because the images don't exist.
        assert result["status"] == "partial"
        assert result["classified"] == 0
        assert result["failed"] == 2
        for r in result["results"]:
            assert "error" in r
            assert "not found" in r["error"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — API endpoint tests for router.py
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyEndpoint:

    def test_api_missing_survey(self, monkeypatch):
        """
        POST /classify with a survey_id that has no detections should return 400.
        """
        import app.classification.service as svc

        def mock_classify(survey_id):
            raise ValueError(
                f"No plant detections found for survey '{survey_id}'."
            )

        monkeypatch.setattr(svc, "classify_survey", mock_classify)

        response = client.post(
            "/classify",
            json={"survey_id": "survey_with_no_detections"},
        )
        assert response.status_code == 400
        assert "no plant detections" in response.json()["detail"].lower()

    def test_api_model_not_found(self, monkeypatch):
        """
        POST /classify should return 503 Service Unavailable
        when best.keras doesn't exist.
        """
        import app.classification.service as svc

        def mock_classify(survey_id):
            raise FileNotFoundError("Model weights not found")

        monkeypatch.setattr(svc, "classify_survey", mock_classify)

        response = client.post(
            "/classify",
            json={"survey_id": "test_survey_001"},
        )
        assert response.status_code == 503
        assert "model" in response.json()["detail"].lower()

    def test_api_invalid_body(self):
        """
        POST /classify with missing survey_id should return 422.
        """
        response = client.post("/classify", json={})
        assert response.status_code == 422

    def test_api_invalid_body_type(self):
        """
        POST /classify with a non-string survey_id should return 422.
        """
        response = client.post("/classify", json={"survey_id": 123})
        assert response.status_code == 422

    def test_api_health(self):
        """
        GET /classify/health should return a dict with module info.
        """
        response = client.get("/classify/health")
        assert response.status_code == 200
        body = response.json()
        assert body["module"] == "Disease Classification"
        assert "status" in body
        assert "model_loaded" in body
        assert "model_version" in body

    def test_api_successful_classification(self, monkeypatch, seeded_in_memory_db):
        """
        POST /classify with a valid survey should return 200
        and the expected response structure.
        """
        import app.classification.service as svc

        def mock_classify(survey_id: str) -> dict:
            return {
                "status": "success",
                "survey_id": survey_id,
                "total_plants": 2,
                "classified": 2,
                "failed": 0,
                "results": [
                    {
                        "plant_id": 1,
                        "image_filename": "test_plant_001.jpg",
                        "bbox": [10.0, 10.0, 50.0, 50.0],
                        "label": "early_blight",
                        "confidence": 0.9543,
                        "all_probs": {
                            "early_blight": 0.9543,
                            "healthy": 0.0210,
                            "late_blight": 0.0150,
                            "septoria_leaf_spot": 0.0097,
                        },
                    },
                    {
                        "plant_id": 2,
                        "image_filename": "test_plant_002.jpg",
                        "bbox": [60.0, 60.0, 110.0, 110.0],
                        "label": "healthy",
                        "confidence": 0.8832,
                        "all_probs": {
                            "early_blight": 0.0510,
                            "healthy": 0.8832,
                            "late_blight": 0.0420,
                            "septoria_leaf_spot": 0.0238,
                        },
                    },
                ],
            }

        monkeypatch.setattr(svc, "classify_survey", mock_classify)

        response = client.post(
            "/classify",
            json={"survey_id": "test_survey_001"},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "success"
        assert body["survey_id"] == "test_survey_001"
        assert body["total_plants"] == 2
        assert body["classified"] == 2
        assert body["failed"] == 0
        assert len(body["results"]) == 2

        # Verify the structure of each result.
        for det in body["results"]:
            assert "plant_id" in det
            assert "image_filename" in det
            assert "bbox" in det
            assert "label" in det
            assert "confidence" in det
            assert "all_probs" in det
            assert len(det["bbox"]) == 4

        assert body["results"][0]["label"] == "early_blight"
        assert body["results"][1]["label"] == "healthy"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — 70/15/15 split manifest, calibration and uncertainty (v2 rebuild)
# ══════════════════════════════════════════════════════════════════════════════

MANIFEST_PATH = REPO_ROOT / "models" / "classification" / "split_manifest.json"
INFERENCE_CONFIG_PATH = REPO_ROOT / "models" / "classification" / "inference_config.json"
METRICS_PATH = REPO_ROOT / "models" / "classification" / "eval" / "metrics.json"

requires_manifest = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="split_manifest.json not found — run scripts/m3_train_classifier.py --stage split",
)


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


@requires_manifest
class TestSplitManifest:

    def test_seed_method_and_class_order(self, manifest):
        assert manifest["random_seed"] == 42
        assert "stratify" in manifest["split_method"]
        assert manifest["class_order"] == CLASS_NAMES

    def test_every_image_in_exactly_one_split(self, manifest):
        paths = [r["path"] for r in manifest["images"]]
        assert len(paths) == len(set(paths)) == manifest["unique_images"]
        assert {r["split"] for r in manifest["images"]} == {"train", "validation", "test"}

    def test_zero_overlap_by_path_and_content_hash(self, manifest):
        by_split = {}
        for r in manifest["images"]:
            by_split.setdefault(r["split"], {"p": set(), "h": set()})
            by_split[r["split"]]["p"].add(r["path"])
            by_split[r["split"]]["h"].add(r["sha256"])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not by_split[a]["p"] & by_split[b]["p"]
            assert not by_split[a]["h"] & by_split[b]["h"]
        assert all(v == 0 for k, v in manifest["leakage_checks"].items() if "overlap" in k)

    def test_split_is_70_15_15_and_stratified(self, manifest):
        n = manifest["unique_images"]
        totals = manifest["split_totals"]
        assert abs(totals["train"] / n - 0.70) < 0.005
        assert abs(totals["validation"] / n - 0.15) < 0.005
        assert abs(totals["test"] / n - 0.15) < 0.005
        for cls in CLASS_NAMES:
            per = {s: manifest["split_counts"][s][cls] for s in totals}
            cls_total = sum(per.values())
            assert abs(per["train"] / cls_total - 0.70) < 0.01
            assert abs(per["test"] - per["validation"]) <= 1

    def test_duplicates_were_excluded(self, manifest):
        dups = manifest["excluded"]["exact_duplicates"]
        kept = {r["path"] for r in manifest["images"]}
        assert all(d["path"] not in kept and d["duplicate_of"] in kept for d in dups)

    def test_labels_match_folders(self, manifest):
        for r in manifest["images"]:
            assert r["path"].split("/")[0] == r["label"]
            assert CLASS_NAMES[r["label_index"]] == r["label"]


class TestCalibrationAndUncertainty:

    def _classifier(self, tmp_path, monkeypatch, probs, cfg=None):
        from app.classification.classifier import DiseaseClassifier

        order = tmp_path / "class_order.json"
        order.write_text(json.dumps({"class_order": CLASS_NAMES}))
        monkeypatch.setattr("app.classification.classifier.CLASS_ORDER_PATH", order)
        weights = tmp_path / "best.keras"
        weights.write_text("fake")
        if cfg is not None:
            (tmp_path / "inference_config.json").write_text(json.dumps(cfg))
        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([probs])
        with patch("app.classification.classifier.tf.keras.models.load_model",
                   return_value=fake_model):
            return DiseaseClassifier(weights_path=weights), fake_model

    def test_low_confidence_is_flagged_uncertain(self, tmp_path, monkeypatch):
        clf, _ = self._classifier(tmp_path, monkeypatch, [0.40, 0.35, 0.15, 0.10])
        out = clf.predict(np.full((50, 80, 3), 90, dtype=np.uint8))
        assert out["label"] == "early_blight"
        assert out["uncertain"] is True

    def test_temperature_changes_confidence_not_label(self, tmp_path, monkeypatch):
        cfg = {"class_order": CLASS_NAMES, "temperature": 2.0, "tta_hflip": False,
               "uncertainty_threshold": 0.6}
        clf, _ = self._classifier(tmp_path, monkeypatch, [0.90, 0.05, 0.03, 0.02], cfg)
        out = clf.predict(np.full((224, 224, 3), 90, dtype=np.uint8))
        assert out["label"] == "early_blight"
        assert out["raw_confidence"] == pytest.approx(0.9)
        assert out["confidence"] < 0.9
        assert abs(sum(out["all_probs"].values()) - 1.0) < 0.01
        assert clf.temperature == 2.0

    def test_tta_runs_a_flipped_pass(self, tmp_path, monkeypatch):
        cfg = {"class_order": CLASS_NAMES, "temperature": 1.0, "tta_hflip": True}
        clf, fake = self._classifier(tmp_path, monkeypatch, [0.7, 0.1, 0.1, 0.1], cfg)
        clf.predict(np.full((224, 224, 3), 90, dtype=np.uint8))
        assert fake.predict.call_count == 2

    def test_mismatched_inference_config_is_ignored(self, tmp_path, monkeypatch):
        cfg = {"class_order": ["a", "b", "c", "d"], "temperature": 5.0}
        clf, _ = self._classifier(tmp_path, monkeypatch, [0.9, 0.05, 0.03, 0.02], cfg)
        assert clf.temperature == 1.0

    def test_uncertain_flag_is_persisted(self, in_memory_db, sample_result):
        sample_result.uncertain = True
        save_classification(in_memory_db, sample_result)
        rows = fetch_classifications_for_survey(in_memory_db, sample_result.survey_id)
        assert rows[0]["uncertain"] is True

    def test_legacy_table_is_migrated(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY, "
            "survey_id TEXT NOT NULL, image_filename TEXT NOT NULL, plant_id INTEGER NOT NULL, "
            "bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL, bbox_x2 REAL NOT NULL, "
            "bbox_y2 REAL NOT NULL, label TEXT NOT NULL, confidence REAL NOT NULL, "
            "all_probs_json TEXT NOT NULL, model_version TEXT NOT NULL, created_at TEXT, "
            "UNIQUE(survey_id, plant_id))")
        ensure_classifications_table(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(classifications)")}
        assert "is_uncertain" in cols
        conn.close()


@requires_model
class TestProductionModel:

    def test_output_shape_and_inference_on_real_test_image(self):
        from app.classification.classifier import DiseaseClassifier
        import cv2

        clf = DiseaseClassifier(weights_path=MODEL_PATH)
        assert clf.model.output_shape[-1] == len(CLASS_NAMES)
        dataset = REPO_ROOT / "data" / "datasets" / "plantvillage_tomato"
        if not (MANIFEST_PATH.exists() and dataset.exists()):
            pytest.skip("dataset not available locally")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        rows = [r for r in manifest["images"] if r["split"] == "test"][:8]
        for r in rows:
            out = clf.predict(cv2.imread(str(dataset / r["path"])))
            assert out["label"] in CLASS_NAMES
            assert 0.0 <= out["confidence"] <= 1.0

    @pytest.mark.skipif(not METRICS_PATH.exists(), reason="no evaluation metrics yet")
    def test_promoted_model_is_the_evaluated_one(self):
        import hashlib

        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        record_path = METRICS_PATH.parent / "promotion_record.json"
        if not record_path.exists():
            pytest.skip("model not promoted yet")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        assert digest == record["new_model_sha256"] == metrics["evaluated_model_sha256"]
