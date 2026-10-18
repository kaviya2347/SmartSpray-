"""
FILE PATH: SmartSpray/tests/test_severity.py

PURPOSE:
    Automated test suite for Module 4 — Disease Severity Estimation (Stage 3).

    Covers:
        1. Unit tests for app/severity/estimator.py:
             - severity math (lesion_px / leaf_px) on synthetic images
             - severity bands (0.20 / 0.50 boundaries are deterministic)
             - healthy-plant short-circuit
             - invalid/edge-case images (empty, black, tiny, grayscale)
             - disease-profile resolution
             - singleton behaviour
             - determinism
        2. Database persistence contract (integrates with
           app/severity/database.save_severity via an in-memory DB).

    These tests use ONLY synthetic images. There is no severity-labeled
    ground-truth dataset, so NO real-world severity accuracy claim is made.

HOW TO RUN (repo root, inside .venv):
    pytest tests/test_severity.py -v
"""

import sqlite3

import numpy as np
import pytest

from app.severity.estimator import (
    SeverityEstimator,
    get_estimator,
    SeverityResult,
    DEFAULT_PROFILES,
    SEVERITY_BANDS,
    CATEGORY_LOW,
    CATEGORY_MEDIUM,
    CATEGORY_HIGH,
    DiseaseThresholdProfile,
)
from app.severity.database import (
    ensure_severity_table,
    save_severity,
    get_severity_for_survey,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_leaf_image():
    """200x200 filled with leaf-green (BGR = 34,139,34)."""
    return np.full((200, 200, 3), (34, 139, 34), dtype=np.uint8)


@pytest.fixture
def early_blight_lesion_image():
    """200x200: left half dark-brown (BGR ~40,40,10), right half leaf-green."""
    img = np.full((200, 200, 3), (34, 139, 34), dtype=np.uint8)
    img[:, :100] = (40, 40, 10)
    return img


@pytest.fixture
def black_image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def tiny_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image():
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def in_memory_db():
    """In-memory SQLite with detections + classifications + plant_severity."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE detections (detection_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
        " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
        " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, confidence REAL NOT NULL,"
        " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')))"
    )
    conn.execute(
        "CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
        " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
        " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, label TEXT NOT NULL,"
        " confidence REAL NOT NULL, all_probs_json TEXT NOT NULL,"
        " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')),"
        " UNIQUE (survey_id, plant_id))"
    )
    conn.execute(
        "INSERT INTO detections"
        " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,model_version)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("s1", "i1.jpg", 1, 10.0, 20.0, 80.0, 90.0, 0.95, "m2"),
    )
    conn.execute(
        "INSERT INTO classifications"
        " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,label,confidence,all_probs_json,model_version)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("s1", "i1.jpg", 1, 10.0, 20.0, 80.0, 90.0, "early_blight", 0.9, "{}", "m3"),
    )
    conn.commit()
    ensure_severity_table(conn)
    yield conn
    conn.close()


# ── Unit tests: severity math and bands ───────────────────────────────────────

class TestSeverityBands:

    def test_low_boundaries(self):
        est = SeverityEstimator()
        assert est._score_to_category(0.00) == CATEGORY_LOW
        assert est._score_to_category(0.14) == CATEGORY_LOW

    def test_medium_boundaries(self):
        est = SeverityEstimator()
        assert est._score_to_category(0.20) == CATEGORY_MEDIUM
        assert est._score_to_category(0.49) == CATEGORY_MEDIUM

    def test_band_boundary_0_20_is_low_medium_boundary(self):
        est = SeverityEstimator()
        assert est._score_to_category(0.19) == CATEGORY_LOW
        assert est._score_to_category(0.20) == CATEGORY_MEDIUM
        assert est._score_to_category(0.1999) == CATEGORY_LOW

    def test_high_boundaries(self):
        est = SeverityEstimator()
        assert est._score_to_category(0.50) == CATEGORY_HIGH
        assert est._score_to_category(1.0) == CATEGORY_HIGH

    def test_bands_cover_full_domain(self):
        assert SEVERITY_BANDS[0][0] == 0.0
        assert SEVERITY_BANDS[-1][1] >= 1.0
        for (lo1, hi1, _), (lo2, _, _) in zip(SEVERITY_BANDS, SEVERITY_BANDS[1:]):
            assert hi1 == lo2


class TestSeverityMath:

    def test_severity_result_has_all_fields(self):
        r = SeverityResult(
            plant_id=1, survey_id="s", disease_label="early_blight",
            severity_score=0.25, severity_category="medium",
            leaf_px=10000, lesion_px=2500, is_valid=True,
        )
        assert r.plant_id == 1
        assert r.severity_score == 0.25
        assert r.severity_category == "medium"
        assert r.leaf_px == 10000
        assert r.lesion_px == 2500
        assert r.is_valid is True
        assert r.model_version

    def test_healthy_plant_returns_zero_severity(self, synthetic_leaf_image):
        est = SeverityEstimator()
        r = est.estimate(synthetic_leaf_image, "healthy", 1, "s1")
        assert r.severity_score == 0.0
        assert r.severity_category == CATEGORY_LOW
        assert r.is_valid is True

    def test_clean_leaf_has_no_lesion(self, synthetic_leaf_image):
        est = SeverityEstimator()
        r = est.estimate(synthetic_leaf_image, "early_blight", 1, "s1")
        assert r.leaf_px > 0
        assert r.lesion_px == 0
        assert r.severity_score == 0.0
        assert r.is_valid is True

    def test_50pct_lesion_synthetic(self, early_blight_lesion_image):
        est = SeverityEstimator()
        r = est.estimate(early_blight_lesion_image, "early_blight", 1, "s1")
        assert 0.30 <= r.severity_score <= 0.70
        assert r.leaf_px > 0
        assert r.lesion_px > 0
        assert r.is_valid is True

    def test_score_is_clamped_to_0_1(self):
        est = SeverityEstimator()
        leaf = np.full((200, 200, 3), (34, 139, 34), dtype=np.uint8)
        leaf[:, :] = (40, 40, 10)
        r = est.estimate(leaf, "early_blight", 1, "s1")
        assert 0.0 <= r.severity_score <= 1.0

    def test_deterministic_on_same_input(self, early_blight_lesion_image):
        est = SeverityEstimator()
        r1 = est.estimate(early_blight_lesion_image, "early_blight", 1, "s1")
        r2 = est.estimate(early_blight_lesion_image, "early_blight", 1, "s1")
        assert r1.severity_score == r2.severity_score
        assert r1.leaf_px == r2.leaf_px
        assert r1.lesion_px == r2.lesion_px
        assert r1.severity_category == r2.severity_category


class TestEdgeCases:

    def test_black_image_returns_invalid(self, black_image):
        est = SeverityEstimator()
        r = est.estimate(black_image, "early_blight", 1, "s1")
        assert r.is_valid is False
        assert r.severity_score == 0.0

    def test_tiny_image_does_not_crash(self, tiny_image):
        est = SeverityEstimator()
        r = est.estimate(tiny_image, "early_blight", 1, "s1")
        assert isinstance(r, SeverityResult)
        assert r.is_valid is False

    def test_grayscale_returns_invalid(self, grayscale_image):
        est = SeverityEstimator()
        r = est.estimate(grayscale_image, "early_blight", 1, "s1")
        assert isinstance(r, SeverityResult)
        assert r.is_valid is False

    def test_empty_array_returns_invalid(self):
        est = SeverityEstimator()
        r = est.estimate(np.zeros((0, 0, 3), dtype=np.uint8), "early_blight", 1, "s1")
        assert r.is_valid is False

    def test_none_returns_invalid(self):
        est = SeverityEstimator()
        r = est.estimate(None, "early_blight", 1, "s1")
        assert r.is_valid is False


class TestDiseaseProfiles:

    def test_all_three_diseases_have_profiles(self):
        for label in ("early_blight", "late_blight", "septoria_leaf_spot"):
            assert label in DEFAULT_PROFILES

    def test_profiles_distinct(self):
        eb = DEFAULT_PROFILES["early_blight"]
        lb = DEFAULT_PROFILES["late_blight"]
        sp = DEFAULT_PROFILES["septoria_leaf_spot"]
        assert eb.hsv_lesion_upper != lb.hsv_lesion_upper
        assert sp.min_component_area_px > 0
        assert sp.max_component_area_ratio < 1.0

    def test_septoria_profile_values(self):
        sp = DEFAULT_PROFILES["septoria_leaf_spot"]
        # The Septoria profile is unchanged by the Stage 3A audit. The proposed
        # "Mask A" HSV addition was rejected by real-image evidence because it
        # over-fired on healthy leaf tissue.
        assert sp.disease == "septoria_leaf_spot"
        assert sp.hsv_lesion_lower == (10, 0, 90)
        assert sp.hsv_lesion_upper == (40, 255, 255)
        assert sp.lab_use is True
        assert sp.lab_lesion_lower == (25, 90, 0)
        assert sp.lab_lesion_upper == (255, 255, 255)
        assert sp.min_component_area_px == 12
        assert sp.max_component_area_ratio == 0.2

    def test_component_sizing_happens_after_leaf_intersection(self):
        """Regression: lesion components are sized AFTER the leaf intersection.

        Discovered during Stage 3A real-image verification of
        `app/severity/estimator.py`. The connected-component size filter ran on
        the full-image lesion mask *before* intersecting with the leaf, so an
        out-of-leaf blob inflated component areas and caused in-leaf lesions to
        be dropped by max_component_area_ratio. The fix moves the leaf
        intersection before the size filter.
        """
        profile = DiseaseThresholdProfile(
            disease="test_straddle",
            hsv_lesion_lower=(0, 0, 100),
            hsv_lesion_upper=(180, 255, 255),
            lab_use=False,
            open_kernel_size=1,
            close_kernel_size=1,
            min_component_area_px=1,
            max_component_area_ratio=1.0,  # cap == leaf area
        )
        est = SeverityEstimator(profiles={"test_straddle": profile})

        # Whole 200x200 image is "lesion" under the profile (V>=100).
        img = np.full((200, 200, 3), 220, dtype=np.uint8)

        # Leaf covers only the left half (200x100 = 20000 px).
        leaf_mask = np.zeros((200, 200), dtype=bool)
        leaf_mask[:, :100] = True

        lesion, lesion_px = est._segment_lesions(img, leaf_mask, profile)

        # Full-image component is 40000 px, but only 20000 px are inside the
        # leaf. With cap == leaf area (20000), the in-leaf component is exactly
        # at the cap and must be retained. Old ordering (size filter before
        # leaf intersection) would see 40000 px and drop everything -> 0.
        assert lesion_px == 20000

    def test_unknown_label_returns_invalid(self, synthetic_leaf_image):
        est = SeverityEstimator()
        r = est.estimate(synthetic_leaf_image, "mycoplasma", 1, "s1")
        assert r.is_valid is False


class TestSingleton:

    def test_get_estimator_returns_same_instance(self):
        assert get_estimator() is get_estimator()

    def test_separate_instances_are_independent(self):
        a = SeverityEstimator()
        b = SeverityEstimator()
        assert a is not b
        assert a._bands == b._bands


class TestDatabaseIntegration:

    def test_save_and_retrieve_severity(self, in_memory_db):
        conn = in_memory_db
        result = SeverityResult(
            plant_id=1, survey_id="s1", disease_label="early_blight",
            severity_score=0.45, severity_category="medium",
            leaf_px=10000, lesion_px=4500, is_valid=True,
        )
        save_severity(
            conn, result.survey_id, result.plant_id,
            result.severity_score, result.severity_category,
            result.leaf_px, result.lesion_px, result.is_valid,
            result.model_version,
        )
        rows = get_severity_for_survey(conn, "s1")
        assert len(rows) == 1
        assert rows[0]["plant_id"] == 1
        assert rows[0]["severity_score"] == 0.45
        assert rows[0]["severity_category"] == "medium"
        assert rows[0]["leaf_px"] == 10000
        assert rows[0]["lesion_px"] == 4500
        assert rows[0]["is_valid"] is True
        assert rows[0]["disease_label"] == "early_blight"

    def test_estimator_result_is_db_compatible(
        self, in_memory_db, early_blight_lesion_image
    ):
        conn = in_memory_db
        est = SeverityEstimator()
        result = est.estimate(early_blight_lesion_image, "early_blight", 1, "s1")
        save_severity(
            conn, result.survey_id, result.plant_id,
            result.severity_score, result.severity_category,
            result.leaf_px, result.lesion_px, result.is_valid,
            result.model_version,
        )
        rows = get_severity_for_survey(conn, "s1")
        assert len(rows) == 1
        assert rows[0]["severity_score"] == result.severity_score
        assert rows[0]["severity_category"] == result.severity_category
        assert rows[0]["leaf_px"] == result.leaf_px
        assert rows[0]["lesion_px"] == result.lesion_px
# ── Stage 4 additions: API / service / pipeline integration tests ─────────────
import cv2  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402
from app.severity import service as severity_service  # noqa: E402
from app.severity.estimator import SeverityResult  # noqa: E402

stage4_client = TestClient(app)


class TestSeverityAPI:

    def test_health_endpoint_returns_200(self):
        resp = stage4_client.get("/severity/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "module" in body
        assert "status" in body
        assert "bands" in body
        assert "note" in body
        assert "heuristic" in body["note"].lower() or "no ground-truth" in body["note"].lower()

    def test_health_bands_reflect_0_20_and_0_50(self):
        resp = stage4_client.get("/severity/health")
        assert resp.status_code == 200
        body = resp.json()
        bands_text = str(body["bands"])
        assert "0.20" in bands_text
        assert "0.50" in bands_text

    def test_post_severity_empty_survey_id_returns_400(self):
        resp = stage4_client.post("/severity", json={"survey_id": ""})
        assert resp.status_code == 400

    def test_post_severity_whitespace_survey_id_returns_400(self):
        resp = stage4_client.post("/severity", json={"survey_id": "   "})
        assert resp.status_code == 400

    def test_post_severity_nonexistent_survey_returns_400(
        self, monkeypatch
    ):
        # Real service path: no detections for this survey -> ValueError -> 400.
        resp = stage4_client.post(
            "/severity", json={"survey_id": "no_such_survey_xyz_stage4_test"}
        )
        assert resp.status_code == 400
        assert "no classified" in resp.json()["detail"].lower()

    def test_post_severity_response_structure_with_mocked_service(
        self, monkeypatch
    ):
        result = SeverityResult(
            plant_id=1,
            survey_id="mock_survey",
            disease_label="early_blight",
            severity_score=0.35,
            severity_category="medium",
            leaf_px=10000,
            lesion_px=3500,
            is_valid=True,
            model_version="opencv-severity-v1",
        )

        class _FakeConn:
            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            severity_service, "estimate_survey", lambda survey_id: [result]
        )
        monkeypatch.setattr(
            "app.severity.router.get_connection", lambda: _FakeConn()
        )
        monkeypatch.setattr(
            "app.severity.router.save_severity", lambda **kw: None
        )

        resp = stage4_client.post("/severity", json={"survey_id": "mock_survey"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["survey_id"] == "mock_survey"
        assert body["total_plants"] == 1
        assert body["valid_estimates"] == 1
        assert body["invalid_estimates"] == 0
        assert len(body["results"]) == 1
        r = body["results"][0]
        assert r["plant_id"] == 1
        assert r["severity_score"] == 0.35
        assert r["severity_category"] == "medium"
        assert r["is_valid"] is True
        assert r["model_version"] == "opencv-severity-v1"
        assert "leaf_px" in r
        assert "lesion_px" in r

    def test_post_severity_is_valid_false_included_in_response(
        self, monkeypatch
    ):
        valid = SeverityResult(
            plant_id=1,
            survey_id="mock_mixed",
            disease_label="early_blight",
            severity_score=0.35,
            severity_category="medium",
            leaf_px=10000,
            lesion_px=3500,
            is_valid=True,
            model_version="opencv-severity-v1",
        )
        invalid = SeverityResult(
            plant_id=2,
            survey_id="mock_mixed",
            disease_label="late_blight",
            severity_score=0.0,
            severity_category="low",
            leaf_px=0,
            lesion_px=0,
            is_valid=False,
            model_version="opencv-severity-v1",
        )

        class _FakeConn:
            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            severity_service,
            "estimate_survey",
            lambda survey_id: [valid, invalid],
        )
        monkeypatch.setattr(
            "app.severity.router.get_connection", lambda: _FakeConn()
        )
        monkeypatch.setattr(
            "app.severity.router.save_severity", lambda **kw: None
        )

        resp = stage4_client.post("/severity", json={"survey_id": "mock_mixed"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_plants"] == 2
        assert body["valid_estimates"] == 1
        assert body["invalid_estimates"] == 1
        assert len(body["results"]) == 2
        by_id = {r["plant_id"]: r for r in body["results"]}
        assert 1 in by_id and 2 in by_id
        assert by_id[1]["is_valid"] is True
        assert by_id[2]["is_valid"] is False
class TestSeverityService:

    def test_estimate_survey_raises_valueerror_for_no_detections(
        self, monkeypatch
    ):
        # No detections/classifications for this survey -> ValueError.
        with pytest.raises(ValueError) as exc_info:
            severity_service.estimate_survey(
                "no_detections_survey_stage4_xyz"
            )
        assert "no classified" in str(exc_info.value).lower()

    def test_estimate_survey_skips_missing_image_without_crashing(
        self, monkeypatch, caplog
    ):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE detections (detection_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
            " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
            " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, confidence REAL NOT NULL,"
            " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')))"
        )
        conn.execute(
            "CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
            " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
            " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, label TEXT NOT NULL,"
            " confidence REAL NOT NULL, all_probs_json TEXT NOT NULL,"
            " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')),"
            " UNIQUE (survey_id, plant_id))"
        )
        conn.execute(
            "INSERT INTO detections"
            " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,model_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s4missing", "does_not_exist.jpg", 1, 0.0, 0.0, 10.0, 10.0, 0.5, "m2"),
        )
        conn.execute(
            "INSERT INTO classifications"
            " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,label,confidence,all_probs_json,model_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s4missing", "does_not_exist.jpg", 1, 0.0, 0.0, 10.0, 10.0, "early_blight", 0.9, "{}", "m3"),
        )
        conn.commit()
        monkeypatch.setattr(
            severity_service, "get_connection", lambda: conn
        )
        with caplog.at_level("WARNING"):
            results = severity_service.estimate_survey("s4missing")
        assert results == []
        assert any(
            "not found" in r.message.lower() for r in caplog.records
        )
        conn.close()

    def test_estimate_survey_passes_is_valid_false_through(
        self, monkeypatch
    ):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE detections (detection_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
            " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
            " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, confidence REAL NOT NULL,"
            " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')))"
        )
        conn.execute(
            "CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
            " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
            " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, label TEXT NOT NULL,"
            " confidence REAL NOT NULL, all_probs_json TEXT NOT NULL,"
            " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')),"
            " UNIQUE (survey_id, plant_id))"
        )
        conn.execute(
            "INSERT INTO detections"
            " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,model_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s4valid", "img.jpg", 1, 0.0, 0.0, 10.0, 10.0, 0.5, "m2"),
        )
        conn.execute(
            "INSERT INTO classifications"
            " (survey_id,image_filename,plant_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,label,confidence,all_probs_json,model_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s4valid", "img.jpg", 1, 0.0, 0.0, 10.0, 10.0, "late_blight", 0.8, "{}", "m3"),
        )
        conn.commit()

        fake_result = SeverityResult(
            plant_id=1,
            survey_id="s4valid",
            disease_label="late_blight",
            severity_score=0.0,
            severity_category="low",
            leaf_px=0,
            lesion_px=0,
            is_valid=False,
            model_version="opencv-severity-v1",
        )

        class _FakeEstimator:
            def estimate(self, *args, **kwargs):
                return fake_result

        monkeypatch.setattr(
            severity_service, "get_connection", lambda: conn
        )
        monkeypatch.setattr(
            severity_service, "_load_and_crop",
            lambda image_path, x1, y1, x2, y2: np.zeros((10, 10, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            severity_service, "get_estimator", lambda: _FakeEstimator()
        )

        results = severity_service.estimate_survey("s4valid")
        # is_valid=False must be passed through, not filtered.
        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].plant_id == 1
        conn.close()

    def test_load_and_crop_matches_module3_crop_logic(self, tmp_path):
        # Contract test: severity _load_and_crop must crop identically to M3.
        # Create a 128x128 synthetic image with a distinct marker region.
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:, :] = (30, 90, 150)  # base colour
        img[5:15, 5:15] = (255, 255, 255)  # white top-left patch
        img_path = tmp_path / "crop_contract.jpg"
        cv2.imwrite(str(img_path), img)

        bbox = (10.0, 10.0, 80.0, 80.0)

        # M3 crop logic.
        try:
            from app.classification.service import (
                _load_and_crop as m3_load_and_crop,
            )
            m3_crop = m3_load_and_crop(img_path, *bbox)
        except ImportError as exc:  # pragma: no cover - direct import check
            raise AssertionError(
                f"app.classification.service._load_and_crop not importable: {exc}"
            )

        # M4 crop logic.
        m4_crop = severity_service._load_and_crop(img_path, *bbox)

        assert m4_crop.shape == (70, 70, 3)
        assert m4_crop.dtype == np.uint8
        # identical crop geometry (M3 is BGR, M4 is RGB: channels differ, size not)
        assert m3_crop.shape == m4_crop.shape
        # both crops must be non-empty
        assert m4_crop[0, 0, :].tolist() != [0, 0, 0]


class TestPipelineStage4Integration:

    def test_pipeline_result_contains_stage_4_severity_key(
        self, monkeypatch
    ):
        # Monkeypatch the DB layer used by pipeline Stage 1 so run_pipeline
        # does not touch the real database (Stage 1 queries surveys).
        class _Cur:
            def fetchone(self):
                return ("stage4_test", "Tomato", "10m", 2, "now")

        class _FakeConn:
            def execute(self, *a, **k):
                return _Cur()

            def commit(self):
                pass

            def close(self):
                pass

        class _FakeSurveyDir:
            def exists(self):
                return False

            def iterdir(self):
                return iter([])

            def __truediv__(self, other):
                return _FakeSurveyDir()

        fake_result = SeverityResult(
            plant_id=1,
            survey_id="pipeline_survey",
            disease_label="early_blight",
            severity_score=0.35,
            severity_category="medium",
            leaf_px=10000,
            lesion_px=3500,
            is_valid=True,
            model_version="opencv-severity-v1",
        )

        monkeypatch.setattr("app.pipeline.get_connection", lambda: _FakeConn())
        # Stage 2 tile scan: fake IMAGES_ROOT so the survey dir "doesn't exist".
        monkeypatch.setattr("app.pipeline.IMAGES_ROOT", _FakeSurveyDir())
        from app.severity import service as _svc

        monkeypatch.setattr(_svc, "estimate_survey", lambda survey_id: [fake_result])
        monkeypatch.setattr("app.severity.database.save_severity", lambda **kw: None)

        from app.pipeline import run_pipeline

        result = run_pipeline("pipeline_survey")
        s4 = result.get("stage_4_severity")
        assert s4 is not None
        assert s4["status"] == "success"
        assert s4["total_plants"] == 1
        assert s4["valid_estimates"] == 1
        assert s4["invalid_estimates"] == 0
        assert len(s4["results"]) == 1

    def test_pipeline_stage4_skipped_when_no_detections(
        self, monkeypatch
    ):
        class _FakeCursor:
            def fetchone(self):
                return ("stage4_no_det", "Tomato", "10m", 2, "now")

        class _FakeConn:
            def execute(self, *a, **k):
                return _FakeCursor()

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            "app.pipeline.get_connection", lambda: _FakeConn()
        )
        monkeypatch.setattr(
            "app.severity.database.get_connection", lambda: _FakeConn()
        )
        class _FakeSurveyDir:
            def exists(self):
                return False

            def iterdir(self):
                return iter([])

            def __truediv__(self, other):
                return _FakeSurveyDir()

        monkeypatch.setattr("app.pipeline.IMAGES_ROOT", _FakeSurveyDir())
        from app.severity import service as _svc
        monkeypatch.setattr(
            _svc, "estimate_survey",
            lambda survey_id: (_ for _ in ()).throw(
                ValueError("no classified detections")
            ),
        )
        monkeypatch.setattr(
            "app.severity.database.save_severity", lambda **kw: None
        )

        from app.pipeline import run_pipeline

        result = run_pipeline("no_det_pipeline")
        s4 = result.get("stage_4_severity")
        assert s4 is not None
        assert s4["status"] == "skipped"
        assert "reason" in s4
        # pipeline returned a dict with all stage keys
        assert "stage_4_severity" in result
        assert "stage_5_spatial_mapping" in result

    def test_pipeline_stage4_error_does_not_abort_pipeline(
        self, monkeypatch
    ):
        class _FakeCursor:
            def fetchone(self):
                return ("stage4_err", "Tomato", "10m", 2, "now")

        class _FakeConn:
            def execute(self, *a, **k):
                return _FakeCursor()

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            "app.pipeline.get_connection", lambda: _FakeConn()
        )
        monkeypatch.setattr(
            "app.severity.database.get_connection", lambda: _FakeConn()
        )
        class _FakeSurveyDir:
            def exists(self):
                return False
            def iterdir(self):
                return iter([])
            def __truediv__(self, other):
                return _FakeSurveyDir()

        monkeypatch.setattr("app.pipeline.IMAGES_ROOT", _FakeSurveyDir())
        from app.severity import service as _svc
        monkeypatch.setattr(
            _svc, "estimate_survey",
            lambda survey_id: (_ for _ in ()).throw(
                RuntimeError("unexpected failure")
            ),
        )
        monkeypatch.setattr(
            "app.severity.database.save_severity", lambda **kw: None
        )

        from app.pipeline import run_pipeline

        result = run_pipeline("error_pipeline")
        s4 = result.get("stage_4_severity")
        assert s4 is not None
        assert s4["status"] == "error"
        assert "error" in s4
        # pipeline must continue; downstream stage keys still present
        assert "stage_5_spatial_mapping" in result
        assert "stage_6_treatment_decision" in result
        print("error reason:", s4["error"])
