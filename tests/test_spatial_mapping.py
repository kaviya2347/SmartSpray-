"""
FILE PATH: SmartSpray/tests/test_spatial_mapping.py

PURPOSE:
    Automated test suite for Module 5 - Spatial Mapping.

    Six groups:
        1. ArUco calibration (app/spatial_mapping/calibration.py)
        2. Database layer (app/spatial_mapping/database.py)
        3. Mapper primitives (app/spatial_mapping/mapper.py)
        4. map_survey() integration against a temporary SQLite database
        5. HTTP API behaviour (app/spatial_mapping/router.py)
        6. Pipeline Stage 5 wiring (app/pipeline.py)

    All marker images are SYNTHETIC (generated with cv2.aruco.generateImageMarker).
    The repository's real survey imagery contains no ArUco markers, so these
    tests verify the geometry and the persistence contract, NOT real-world
    field accuracy.

HOW TO RUN (repo root, inside .venv):
    pytest tests/test_spatial_mapping.py -v
"""

import json
import re
import sqlite3

import cv2
import numpy as np
import pytest

from app.spatial_mapping.calibration import (
    DEFAULT_ARUCO_DICT,
    FIELD_CORNER_IDS,
    MIN_MARKERS,
    ArUcoCalibrator,
    build_field_corner_points,
    get_calibrator,
    homography_from_json,
    make_calibration_id,
)
from app.spatial_mapping.database import (
    ensure_spatial_tables,
    get_calibration,
    get_detections_for_mapping,
    get_latest_calibration_for_survey,
    get_locations_for_survey,
    save_calibration,
    save_location,
)
from app.spatial_mapping import mapper as spatial_mapper

# ── Shared constants ──────────────────────────────────────────────────────────

MARKER_PX = 80          # rendered marker size in the synthetic canvases
CANVAS_H = 800
CANVAS_W = 1000
FIELD_W = 10.0          # metres declared for the synthetic field
FIELD_H = 8.0

# Corner marker centres used by the synthetic canvases.
CORNER_PLACEMENTS = {0: (100, 100), 1: (900, 100), 2: (900, 700), 3: (100, 700)}

# A known pixel -> field homography: origin at pixel (100, 100), 1 cm per pixel.
# pixel (px, py) -> ((px - 100) * 0.01, (py - 100) * 0.01)
CALIBRATED_H = [[0.01, 0.0, -1.0], [0.0, 0.01, -1.0], [0.0, 0.0, 1.0]]


def _make_marker_canvas(placements=None, size=(CANVAS_H, CANVAS_W)):
    """Build a white BGR image with ArUco markers at the given pixel centres."""
    placements = CORNER_PLACEMENTS if placements is None else placements
    height, width = size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    half = MARKER_PX // 2
    for marker_id, (cx, cy) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
        canvas[cy - half:cy + half, cx - half:cx + half] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR
        )
    return canvas


def _create_module_schema(conn):
    """Create the M2 detections table, M4 plant_severity table and the two M5
    tables in a throwaway database."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS detections (detection_id INTEGER PRIMARY KEY"
        " AUTOINCREMENT, survey_id TEXT NOT NULL, image_filename TEXT NOT NULL,"
        " plant_id INTEGER NOT NULL, bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,"
        " bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL, confidence REAL NOT NULL,"
        " model_version TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now')))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS plant_severity (severity_id INTEGER PRIMARY"
        " KEY AUTOINCREMENT, survey_id TEXT NOT NULL, plant_id INTEGER NOT NULL,"
        " image_filename TEXT NOT NULL, disease_label TEXT NOT NULL,"
        " severity_score REAL NOT NULL, severity_category TEXT NOT NULL,"
        " leaf_px INTEGER NOT NULL, lesion_px INTEGER NOT NULL,"
        " is_valid INTEGER NOT NULL DEFAULT 1, model_version TEXT NOT NULL,"
        " estimated_at TEXT NOT NULL DEFAULT (datetime('now')),"
        " UNIQUE (survey_id, plant_id))"
    )
    ensure_spatial_tables(conn)
    conn.commit()


def _seed_detection(conn, survey_id, plant_id, image_filename, bbox,
                    confidence=0.9):
    """Insert one Module 2 detection row."""
    conn.execute(
        "INSERT INTO detections (survey_id, image_filename, plant_id, bbox_x1,"
        " bbox_y1, bbox_x2, bbox_y2, confidence, model_version)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (survey_id, image_filename, plant_id, bbox[0], bbox[1], bbox[2],
         bbox[3], confidence, "m2-test"),
    )
    conn.commit()


def _seed_calibration(conn, survey_id, calibration_id="cal_test_1",
                      is_valid=1, homography=None):
    """Insert one spatial_calibrations row."""
    save_calibration(
        conn=conn,
        calibration_id=calibration_id,
        survey_id=survey_id,
        image_filename="calibration.jpg",
        field_width_m=FIELD_W,
        field_height_m=FIELD_H,
        aruco_dict=DEFAULT_ARUCO_DICT,
        marker_count=MIN_MARKERS,
        homography_json=json.dumps(
            CALIBRATED_H if homography is None else homography
        ),
        reprojection_error=0.0,
        is_valid=is_valid,
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def in_memory_db():
    """In-memory SQLite holding detections + plant_severity + the M5 tables.

    Seed data:
        plant 1 - has a plant_severity row
        plant 2 - has NO plant_severity row (exercises the LEFT JOIN)
    """
    conn = sqlite3.connect(":memory:")
    _create_module_schema(conn)
    _seed_detection(conn, "s1", 1, "i1.jpg", (10.0, 20.0, 80.0, 90.0))
    _seed_detection(conn, "s1", 2, "i1.jpg", (100.0, 200.0, 200.0, 300.0))
    conn.execute(
        "INSERT INTO plant_severity (survey_id, plant_id, image_filename,"
        " disease_label, severity_score, severity_category, leaf_px, lesion_px,"
        " is_valid, model_version) VALUES"
        " ('s1', 1, 'i1.jpg', 'early_blight', 0.3, 'medium', 100, 30, 1, 'm4')"
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def marker_canvas():
    """Synthetic BGR image with the four field-corner ArUco markers."""
    return _make_marker_canvas()


@pytest.fixture
def temp_db_env(tmp_path, monkeypatch):
    """Point app.spatial_mapping.mapper at a throwaway file database.

    A file (not :memory:) DB is used because map_survey() opens and closes its
    own connection, which would discard an in-memory database.
    """
    db_path = tmp_path / "spatial_test.db"

    def _connect():
        return sqlite3.connect(db_path)

    monkeypatch.setattr(spatial_mapper, "get_connection", _connect)

    conn = _connect()
    _create_module_schema(conn)
    conn.close()

    yield {"path": db_path, "connect": _connect}


@pytest.fixture
def api_db_env(tmp_path, monkeypatch):
    """Point BOTH the router and the mapper at a throwaway file database."""
    import app.spatial_mapping.router as spatial_router

    db_path = tmp_path / "api_test.db"

    def _connect():
        return sqlite3.connect(db_path)

    monkeypatch.setattr(spatial_mapper, "get_connection", _connect)
    monkeypatch.setattr(spatial_router, "get_connection", _connect)

    conn = _connect()
    _create_module_schema(conn)
    conn.close()

    yield {"path": db_path, "connect": _connect}

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 - ArUco calibration
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationId:

    def test_matches_documented_format(self):
        cal_id = make_calibration_id("survey_001")
        assert re.match(r"^cal_survey_001_\d{8}T\d{6}Z$", cal_id), cal_id

    def test_contains_survey_id(self):
        assert "my_survey" in make_calibration_id("my_survey")

    def test_is_deterministic_for_a_given_timestamp(self):
        from datetime import datetime, timezone
        when = datetime(2026, 7, 14, 10, 15, 30, tzinfo=timezone.utc)
        assert make_calibration_id("s1", when=when) == "cal_s1_20260714T101530Z"


class TestFieldCornerPoints:

    def test_returns_four_corner_positions(self):
        corners = build_field_corner_points(FIELD_W, FIELD_H)
        assert set(corners) == set(FIELD_CORNER_IDS)
        assert corners[0] == (0.0, 0.0)
        assert corners[1] == (FIELD_W, 0.0)
        assert corners[2] == (FIELD_W, FIELD_H)
        assert corners[3] == (0.0, FIELD_H)

    @pytest.mark.parametrize("width,height", [(0.0, 5.0), (-1.0, 5.0)])
    def test_rejects_bad_width(self, width, height):
        with pytest.raises(ValueError, match="field_width_m"):
            build_field_corner_points(width, height)

    @pytest.mark.parametrize("width,height", [(5.0, 0.0), (5.0, -2.0)])
    def test_rejects_bad_height(self, width, height):
        with pytest.raises(ValueError, match="field_height_m"):
            build_field_corner_points(width, height)


class TestHomographyFromJson:

    def test_round_trips_a_matrix(self):
        matrix = np.array(CALIBRATED_H, dtype=np.float64)
        restored = homography_from_json(json.dumps(matrix.tolist()))
        assert restored.shape == (3, 3)
        assert np.allclose(restored, matrix)

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            homography_from_json("not-json-at-all")

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="3x3"):
            homography_from_json(json.dumps([[1.0, 0.0], [0.0, 1.0]]))


class TestMarkerDetection:

    def test_detects_all_four_corner_markers(self, marker_canvas):
        detections = ArUcoCalibrator().detect_markers(marker_canvas)
        assert [d.marker_id for d in detections] == [0, 1, 2, 3]

    def test_detected_ids_are_sorted_even_when_drawn_out_of_order(self):
        placements = {3: (100, 100), 0: (900, 100), 2: (900, 700), 1: (100, 700)}
        detections = ArUcoCalibrator().detect_markers(_make_marker_canvas(placements))
        ids = [d.marker_id for d in detections]
        assert ids == sorted(ids) == [0, 1, 2, 3]

    def test_marker_centre_matches_placement(self, marker_canvas):
        detections = {d.marker_id: d for d in ArUcoCalibrator().detect_markers(marker_canvas)}
        for marker_id, (cx, cy) in CORNER_PLACEMENTS.items():
            assert detections[marker_id].center[0] == pytest.approx(cx, abs=1.0)
            assert detections[marker_id].center[1] == pytest.approx(cy, abs=1.0)

    def test_returns_empty_list_for_blank_image(self):
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        assert ArUcoCalibrator().detect_markers(blank) == []

    def test_accepts_grayscale_input(self, marker_canvas):
        gray = cv2.cvtColor(marker_canvas, cv2.COLOR_BGR2GRAY)
        assert len(ArUcoCalibrator().detect_markers(gray)) == 4

    def test_accepts_bgr_input(self, marker_canvas):
        assert len(ArUcoCalibrator().detect_markers(marker_canvas)) == 4

    def test_detection_is_deterministic(self, marker_canvas):
        calibrator = ArUcoCalibrator()
        first = [d.as_dict() for d in calibrator.detect_markers(marker_canvas)]
        second = [d.as_dict() for d in calibrator.detect_markers(marker_canvas)]
        assert first == second

    def test_rejects_none_image(self):
        with pytest.raises(ValueError, match="must not be None"):
            ArUcoCalibrator().detect_markers(None)

    def test_rejects_empty_image(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ArUcoCalibrator().detect_markers(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_rejects_unsupported_channel_count(self):
        with pytest.raises(ValueError, match="Unsupported image shape"):
            ArUcoCalibrator().detect_markers(np.zeros((10, 10, 2), dtype=np.uint8))

class TestHomographySolving:

    PIXEL_POINTS = [(100.0, 100.0), (900.0, 100.0), (900.0, 700.0), (100.0, 700.0)]
    FIELD_POINTS = [(0.0, 0.0), (FIELD_W, 0.0), (FIELD_W, FIELD_H), (0.0, FIELD_H)]

    def test_four_point_solve_maps_corners_exactly(self):
        calibrator = ArUcoCalibrator()
        homography, error = calibrator.compute_homography(
            self.PIXEL_POINTS, self.FIELD_POINTS
        )
        assert homography.shape == (3, 3)
        assert error == pytest.approx(0.0, abs=1e-9)

        for (px, py), (fx, fy) in zip(self.PIXEL_POINTS, self.FIELD_POINTS):
            mapped = cv2.perspectiveTransform(
                np.array([[[px, py]]], dtype=np.float64), homography
            ).ravel()
            assert mapped[0] == pytest.approx(fx, abs=1e-6)
            assert mapped[1] == pytest.approx(fy, abs=1e-6)

    def test_interior_point_is_interpolated_correctly(self):
        calibrator = ArUcoCalibrator()
        homography, _error = calibrator.compute_homography(
            self.PIXEL_POINTS, self.FIELD_POINTS
        )
        mapped = cv2.perspectiveTransform(
            np.array([[[500.0, 400.0]]], dtype=np.float64), homography
        ).ravel()
        assert mapped[0] == pytest.approx(FIELD_W / 2.0, abs=1e-6)
        assert mapped[1] == pytest.approx(FIELD_H / 2.0, abs=1e-6)

    def test_more_than_four_points_uses_ransac_and_still_solves(self):
        calibrator = ArUcoCalibrator()
        pixels = self.PIXEL_POINTS + [(500.0, 400.0)]
        fields = self.FIELD_POINTS + [(FIELD_W / 2.0, FIELD_H / 2.0)]
        homography, error = calibrator.compute_homography(pixels, fields)
        assert homography is not None
        assert error < 1.0

    def test_rejects_fewer_than_four_points(self):
        calibrator = ArUcoCalibrator()
        with pytest.raises(ValueError, match="At least 4 correspondences"):
            calibrator.compute_homography(
                self.PIXEL_POINTS[:3], self.FIELD_POINTS[:3]
            )

    def test_rejects_mismatched_point_counts(self):
        calibrator = ArUcoCalibrator()
        with pytest.raises(ValueError, match="same shape"):
            calibrator.compute_homography(self.PIXEL_POINTS, self.FIELD_POINTS[:3])

    def test_collinear_pixels_produce_no_homography(self):
        calibrator = ArUcoCalibrator()
        collinear = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
        with pytest.raises(ValueError, match="degenerate|no solution"):
            calibrator.compute_homography(collinear, self.FIELD_POINTS)


class TestCalibrate:

    def test_calibrates_synthetic_corner_markers(self, marker_canvas):
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        assert result.is_valid is True
        assert result.marker_count == MIN_MARKERS
        assert result.homography is not None
        assert result.reprojection_error == pytest.approx(0.0, abs=1e-6)
        assert result.notes == []

    def test_maps_corner_markers_onto_field_corners(self, marker_canvas):
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        for marker_id, (cx, cy) in CORNER_PLACEMENTS.items():
            mapped = cv2.perspectiveTransform(
                np.array([[[float(cx), float(cy)]]], dtype=np.float64),
                result.homography,
            ).ravel()
            expected = build_field_corner_points(FIELD_W, FIELD_H)[marker_id]
            assert mapped[0] == pytest.approx(expected[0], abs=0.02)
            assert mapped[1] == pytest.approx(expected[1], abs=0.02)

    def test_maps_an_interior_plant_correctly(self, marker_canvas):
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        mapped = cv2.perspectiveTransform(
            np.array([[[500.0, 400.0]]], dtype=np.float64), result.homography
        ).ravel()
        assert mapped[0] == pytest.approx(5.0, abs=0.02)
        assert mapped[1] == pytest.approx(4.0, abs=0.02)

    def test_ignores_markers_outside_the_corner_id_set(self):
        placements = dict(CORNER_PLACEMENTS)
        placements[7] = (500, 400)
        canvas = _make_marker_canvas(placements)
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", canvas, FIELD_W, FIELD_H
        )
        assert result.marker_count == MIN_MARKERS
        assert result.is_valid is True
        assert any("outside the field-corner set" in note for note in result.notes)

    def test_refuses_when_no_markers_are_present(self):
        blank = np.full((400, 400, 3), 255, dtype=np.uint8)
        with pytest.raises(ValueError, match="Refusing to invent field coordinates"):
            ArUcoCalibrator().calibrate(
                "survey_001", "blank.jpg", blank, FIELD_W, FIELD_H
            )

    def test_refuses_with_only_two_corner_markers(self):
        partial = {0: (100, 100), 1: (900, 100)}
        with pytest.raises(ValueError, match="found 2 usable marker"):
            ArUcoCalibrator().calibrate(
                "survey_001", "partial.jpg", _make_marker_canvas(partial),
                FIELD_W, FIELD_H,
            )

    def test_refuses_empty_survey_id(self, marker_canvas):
        with pytest.raises(ValueError, match="survey_id must be a non-empty"):
            ArUcoCalibrator().calibrate(
                "   ", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
            )

    def test_homography_json_round_trips(self, marker_canvas):
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        assert np.allclose(homography_from_json(result.homography_json),
                           result.homography)

    def test_as_dict_is_json_friendly(self, marker_canvas):
        result = ArUcoCalibrator().calibrate(
            "survey_001", "calibration.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        payload = result.as_dict()
        json.dumps(payload)
        assert payload["survey_id"] == "survey_001"
        assert payload["field_width_m"] == FIELD_W

    def test_calibration_is_deterministic_for_the_same_image(self, marker_canvas):
        first = ArUcoCalibrator().calibrate(
            "survey_001", "c.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        second = ArUcoCalibrator().calibrate(
            "survey_001", "c.jpg", marker_canvas, FIELD_W, FIELD_H
        )
        assert np.allclose(first.homography, second.homography)
        assert first.reprojection_error == second.reprojection_error

    def test_calibrate_from_file_missing_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            ArUcoCalibrator().calibrate_from_file(
                "survey_001", "missing.jpg", tmp_path / "missing.jpg",
                FIELD_W, FIELD_H,
            )

    def test_calibrate_from_file_reads_a_real_image(self, tmp_path, marker_canvas):
        image_path = tmp_path / "calibration.png"
        assert cv2.imwrite(str(image_path), marker_canvas) is True
        result = ArUcoCalibrator().calibrate_from_file(
            "survey_001", "calibration.png", image_path, FIELD_W, FIELD_H
        )
        assert result.is_valid is True
        assert result.marker_count == MIN_MARKERS


class TestCalibratorSingleton:

    def test_get_calibrator_returns_the_same_instance(self):
        assert get_calibrator() is get_calibrator()

    def test_get_calibrator_uses_the_default_dictionary(self):
        assert get_calibrator().aruco_dict == DEFAULT_ARUCO_DICT
        assert get_calibrator().min_markers == MIN_MARKERS

    def test_unknown_dictionary_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
            ArUcoCalibrator(aruco_dict="DICT_NOT_REAL")

    def test_empty_dictionary_name_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ArUcoCalibrator(aruco_dict="  ")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 - Database layer
# ═════════════════════════════════════════════════════════════════════════════

class TestSpatialTables:

    def test_creates_both_module_tables(self):
        conn = sqlite3.connect(":memory:")
        ensure_spatial_tables(conn)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"plant_locations", "spatial_calibrations"} <= names
        conn.close()

    def test_is_idempotent(self):
        conn = sqlite3.connect(":memory:")
        ensure_spatial_tables(conn)
        ensure_spatial_tables(conn)
        ensure_spatial_tables(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='plant_locations'"
        ).fetchone()[0] == 1
        conn.close()

    def test_runtime_ddl_matches_database_schema_sql(self):
        """Guard against drift between schema.sql and the runtime DDL."""
        from app.db import SCHEMA_PATH

        from_schema = sqlite3.connect(":memory:")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            from_schema.executescript(handle.read())

        from_runtime = sqlite3.connect(":memory:")
        ensure_spatial_tables(from_runtime)

        for table in ("plant_locations", "spatial_calibrations"):
            expected = [r[1] for r in from_schema.execute(
                "PRAGMA table_info(%s)" % table)]
            actual = [r[1] for r in from_runtime.execute(
                "PRAGMA table_info(%s)" % table)]
            assert actual == expected, "column drift in %s" % table

        from_schema.close()
        from_runtime.close()


class TestLocationsPersistence:

    def test_save_then_read_back(self, in_memory_db):
        save_location(
            conn=in_memory_db, survey_id="s1", plant_id=1,
            image_filename="i1.jpg", bbox_cx_px=45.0, bbox_cy_px=55.0,
            field_x_m=1.25, field_y_m=2.5, is_mapped=1,
            calibration_id="cal_x", mapping_mode="calibrated",
            model_version="spatial-mapping-v1",
        )
        rows = get_locations_for_survey(in_memory_db, "s1")
        assert len(rows) == 1
        assert rows[0]["field_x_m"] == 1.25
        assert rows[0]["field_y_m"] == 2.5
        assert rows[0]["is_mapped"] == 1
        assert rows[0]["mapping_mode"] == "calibrated"
        assert rows[0]["calibration_id"] == "cal_x"

    def test_unmapped_row_stores_null_coordinates(self, in_memory_db):
        save_location(
            conn=in_memory_db, survey_id="s1", plant_id=2,
            image_filename="i1.jpg", bbox_cx_px=150.0, bbox_cy_px=250.0,
            field_x_m=None, field_y_m=None, is_mapped=0,
            calibration_id=None, mapping_mode="uncalibrated",
            model_version="spatial-mapping-v1",
        )
        row = get_locations_for_survey(in_memory_db, "s1")[0]
        assert row["field_x_m"] is None
        assert row["field_y_m"] is None
        assert row["is_mapped"] == 0
        assert row["calibration_id"] is None

    def test_results_are_ordered_by_plant_id(self, in_memory_db):
        for plant_id in (3, 1, 2):
            save_location(
                conn=in_memory_db, survey_id="s1", plant_id=plant_id,
                image_filename="i1.jpg", bbox_cx_px=1.0, bbox_cy_px=1.0,
                field_x_m=None, field_y_m=None, is_mapped=0,
                calibration_id=None, mapping_mode="uncalibrated",
                model_version="spatial-mapping-v1",
            )
        rows = get_locations_for_survey(in_memory_db, "s1")
        assert [r["plant_id"] for r in rows] == [1, 2, 3]

    def test_rerun_overwrites_instead_of_duplicating(self, in_memory_db):
        for field_x in (1.0, 9.9):
            save_location(
                conn=in_memory_db, survey_id="s1", plant_id=1,
                image_filename="i1.jpg", bbox_cx_px=45.0, bbox_cy_px=55.0,
                field_x_m=field_x, field_y_m=2.5, is_mapped=1,
                calibration_id="cal_x", mapping_mode="calibrated",
                model_version="spatial-mapping-v1",
            )
        rows = [r for r in get_locations_for_survey(in_memory_db, "s1")
                if r["plant_id"] == 1]
        assert len(rows) == 1
        assert rows[0]["field_x_m"] == 9.9

    def test_unknown_survey_returns_empty_list(self, in_memory_db):
        assert get_locations_for_survey(in_memory_db, "no_such_survey") == []


class TestCalibrationPersistence:

    def test_save_then_get(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_one")
        row = get_calibration(in_memory_db, "cal_one")
        assert row is not None
        assert row["survey_id"] == "s1"
        assert row["marker_count"] == MIN_MARKERS
        assert row["is_valid"] == 1
        assert np.allclose(homography_from_json(row["homography_json"]),
                           np.array(CALIBRATED_H))

    def test_get_missing_returns_none(self, in_memory_db):
        assert get_calibration(in_memory_db, "nope") is None

    def test_latest_returns_the_most_recent_valid_row(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_old")
        in_memory_db.execute(
            "UPDATE spatial_calibrations SET calibrated_at='2026-01-01 00:00:00'"
            " WHERE calibration_id='cal_old'")
        _seed_calibration(in_memory_db, "s1", "cal_new")
        in_memory_db.execute(
            "UPDATE spatial_calibrations SET calibrated_at='2026-06-01 00:00:00'"
            " WHERE calibration_id='cal_new'")
        in_memory_db.commit()

        latest = get_latest_calibration_for_survey(in_memory_db, "s1")
        assert latest is not None
        assert latest["calibration_id"] == "cal_new"

    def test_latest_ignores_invalid_rows(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_bad", is_valid=0)
        in_memory_db.execute(
            "UPDATE spatial_calibrations SET calibrated_at='2026-09-09 00:00:00'"
            " WHERE calibration_id='cal_bad'")
        in_memory_db.commit()
        assert get_latest_calibration_for_survey(in_memory_db, "s1") is None

    def test_latest_returns_none_when_no_calibration_exists(self, in_memory_db):
        assert get_latest_calibration_for_survey(in_memory_db, "s1") is None

class TestDetectionsForMapping:

    def test_computes_bounding_box_centres(self, in_memory_db):
        rows = get_detections_for_mapping(in_memory_db, "s1")
        assert [r["plant_id"] for r in rows] == [1, 2]
        assert rows[0]["bbox_cx_px"] == pytest.approx(45.0)
        assert rows[0]["bbox_cy_px"] == pytest.approx(55.0)
        assert rows[1]["bbox_cx_px"] == pytest.approx(150.0)
        assert rows[1]["bbox_cy_px"] == pytest.approx(250.0)

    def test_carries_severity_when_present(self, in_memory_db):
        rows = {r["plant_id"]: r for r in get_detections_for_mapping(in_memory_db, "s1")}
        assert rows[1]["severity_score"] == pytest.approx(0.3)
        assert rows[1]["severity_category"] == "medium"

    def test_left_joins_plants_without_a_severity_row(self, in_memory_db):
        rows = {r["plant_id"]: r for r in get_detections_for_mapping(in_memory_db, "s1")}
        assert 2 in rows
        assert rows[2]["severity_score"] is None
        assert rows[2]["severity_category"] is None

    def test_returns_empty_list_for_unknown_survey(self, in_memory_db):
        assert get_detections_for_mapping(in_memory_db, "unknown") == []


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 - Mapper primitives
# ══════════════════════════════════════════════════════════════════════════════

class TestBboxCentre:

    def test_computes_the_geometric_centre(self):
        assert spatial_mapper.bbox_center_px(10.0, 20.0, 80.0, 90.0) == (45.0, 55.0)

    def test_handles_non_integer_boxes(self):
        assert spatial_mapper.bbox_center_px(0.0, 0.0, 5.0, 7.0) == (2.5, 3.5)


class TestPixelToField:

    def test_scale_and_translate_homography(self):
        homography = np.array(CALIBRATED_H, dtype=np.float64)
        assert spatial_mapper.pixel_to_field(homography, 500.0, 400.0) == pytest.approx((4.0, 3.0))
        assert spatial_mapper.pixel_to_field(homography, 100.0, 100.0) == pytest.approx((0.0, 0.0))

    def test_rejects_non_3x3_matrix(self):
        with pytest.raises(ValueError, match="3x3"):
            spatial_mapper.pixel_to_field(np.eye(2), 1.0, 1.0)

    def test_rejects_non_finite_matrix(self):
        bad = np.array(CALIBRATED_H, dtype=np.float64)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            spatial_mapper.pixel_to_field(bad, 1.0, 1.0)


class TestResolveCalibration:

    def test_auto_selects_the_latest_valid_calibration(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_auto")
        resolved = spatial_mapper.resolve_calibration(in_memory_db, "s1")
        assert resolved["calibration_id"] == "cal_auto"

    def test_returns_none_when_survey_is_uncalibrated(self, in_memory_db):
        assert spatial_mapper.resolve_calibration(in_memory_db, "s1") is None

    def test_explicit_calibration_is_returned(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_explicit")
        resolved = spatial_mapper.resolve_calibration(in_memory_db, "s1", "cal_explicit")
        assert resolved["calibration_id"] == "cal_explicit"

    def test_explicit_unknown_calibration_raises(self, in_memory_db):
        with pytest.raises(ValueError, match="was not found"):
            spatial_mapper.resolve_calibration(in_memory_db, "s1", "cal_missing")

    def test_explicit_calibration_from_another_survey_raises(self, in_memory_db):
        _seed_calibration(in_memory_db, "other_survey", "cal_other")
        with pytest.raises(ValueError, match="belongs to survey"):
            spatial_mapper.resolve_calibration(in_memory_db, "s1", "cal_other")

    def test_explicit_invalid_calibration_raises(self, in_memory_db):
        _seed_calibration(in_memory_db, "s1", "cal_invalid", is_valid=0)
        with pytest.raises(ValueError, match="flagged invalid"):
            spatial_mapper.resolve_calibration(in_memory_db, "s1", "cal_invalid")


class TestSummarise:

    def test_counts_mapped_and_unmapped(self):
        records = [{"is_mapped": 1}, {"is_mapped": 1}, {"is_mapped": 0}]
        assert spatial_mapper.summarise(records) == {
            "total_plants": 3, "mapped_plants": 2, "unmapped_plants": 1,
        }

    def test_empty_list(self):
        assert spatial_mapper.summarise([]) == {
            "total_plants": 0, "mapped_plants": 0, "unmapped_plants": 0,
        }

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 - map_survey() integration against a temporary database
# ════════════════════════════════════════════════════════════════════════════

class TestMapSurveyUncalibrated:

    def test_records_null_coordinates_when_no_calibration_exists(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        conn.close()

        records = spatial_mapper.map_survey("s1")

        assert len(records) == 1
        assert records[0]["plant_id"] == 1
        assert records[0]["bbox_cx_px"] == pytest.approx(200.0)
        assert records[0]["bbox_cy_px"] == pytest.approx(300.0)
        assert records[0]["field_x_m"] is None
        assert records[0]["field_y_m"] is None
        assert records[0]["is_mapped"] == 0
        assert records[0]["calibration_id"] is None
        assert records[0]["mapping_mode"] == spatial_mapper.MAPPING_MODE_UNCALIBRATED
        assert records[0]["model_version"] == spatial_mapper.MODEL_VERSION

    def test_persists_the_uncalibrated_rows(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        conn.close()

        spatial_mapper.map_survey("s1")

        conn = temp_db_env["connect"]()
        try:
            rows = get_locations_for_survey(conn, "s1")
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0]["field_x_m"] is None
        assert rows[0]["is_mapped"] == 0

    def test_falls_back_to_uncalibrated_when_only_invalid_calibrations_exist(
        self, temp_db_env
    ):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        _seed_calibration(conn, "s1", "cal_bad", is_valid=0)
        conn.close()

        records = spatial_mapper.map_survey("s1")

        assert records[0]["is_mapped"] == 0
        assert records[0]["field_x_m"] is None
        assert records[0]["mapping_mode"] == spatial_mapper.MAPPING_MODE_UNCALIBRATED


class TestMapSurveyCalibrated:

    def test_converts_pixels_to_metres(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        _seed_calibration(conn, "s1", "cal_ok")
        conn.close()

        records = spatial_mapper.map_survey("s1")

        assert records[0]["field_x_m"] == pytest.approx(1.0)
        assert records[0]["field_y_m"] == pytest.approx(2.0)
        assert records[0]["is_mapped"] == 1
        assert records[0]["mapping_mode"] == spatial_mapper.MAPPING_MODE_CALIBRATED
        assert records[0]["calibration_id"] == "cal_ok"

    def test_explicit_calibration_id_is_used_and_recorded(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 100.0, 100.0, 100.0))
        _seed_calibration(conn, "s1", "cal_chosen")
        conn.close()

        records = spatial_mapper.map_survey("s1", calibration_id="cal_chosen")

        assert records[0]["calibration_id"] == "cal_chosen"
        assert records[0]["field_x_m"] == pytest.approx(0.0)
        assert records[0]["field_y_m"] == pytest.approx(0.0)

    def test_maps_every_detected_plant(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 100.0, 100.0, 100.0))
        _seed_detection(conn, "s1", 2, "i1.jpg", (200.0, 200.0, 200.0, 200.0))
        _seed_detection(conn, "s1", 3, "i1.jpg", (300.0, 300.0, 300.0, 300.0))
        _seed_calibration(conn, "s1", "cal_ok")
        conn.close()

        records = spatial_mapper.map_survey("s1")

        assert [r["plant_id"] for r in records] == [1, 2, 3]
        assert [r["field_x_m"] for r in records] == pytest.approx([0.0, 1.0, 2.0])
        assert all(r["is_mapped"] == 1 for r in records)

    def test_rerun_overwrites_rows_without_duplicating(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 100.0, 100.0, 100.0))
        conn.close()

        spatial_mapper.map_survey("s1")
        conn = temp_db_env["connect"]()
        _seed_calibration(conn, "s1", "cal_ok")
        conn.close()
        second = spatial_mapper.map_survey("s1")

        assert second[0]["is_mapped"] == 1
        conn = temp_db_env["connect"]()
        try:
            rows = get_locations_for_survey(conn, "s1")
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0]["is_mapped"] == 1
        assert rows[0]["mapping_mode"] == spatial_mapper.MAPPING_MODE_CALIBRATED


class TestMapSurveyValidation:

    def test_no_detections_raises_value_error(self, temp_db_env):
        with pytest.raises(ValueError, match="No detections found"):
            spatial_mapper.map_survey("empty_survey")

    @pytest.mark.parametrize("bad_survey_id", ["", "   "])
    def test_empty_survey_id_raises(self, temp_db_env, bad_survey_id):
        with pytest.raises(ValueError, match="non-empty string"):
            spatial_mapper.map_survey(bad_survey_id)

    def test_unknown_explicit_calibration_raises(self, temp_db_env):
        conn = temp_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 100.0, 100.0, 100.0))
        conn.close()
        with pytest.raises(ValueError, match="was not found"):
            spatial_mapper.map_survey("s1", calibration_id="cal_ghost")

# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 - HTTP API
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

spatial_client = TestClient(app)


def _prepare_survey_image_dir(tmp_path, monkeypatch, image=None):
    """Point the router's IMAGES_ROOT at tmp_path and (optionally) write a
    calibration image into data/images/<survey>/ ."""
    import app.spatial_mapping.router as spatial_router

    monkeypatch.setattr(spatial_router, "IMAGES_ROOT", tmp_path)
    survey_dir = tmp_path / "survey_001"
    survey_dir.mkdir(exist_ok=True)
    if image is not None:
        assert cv2.imwrite(str(survey_dir / "calibration.jpg"), image) is True
    return survey_dir


class TestSpatialHealthEndpoint:

    def test_health_returns_200(self):
        response = spatial_client.get("/spatial/health")
        assert response.status_code == 200
        body = response.json()
        assert body["module"] == "Spatial Mapping"
        assert body["calibrator_ready"] is True
        assert body["aruco_dict"] == DEFAULT_ARUCO_DICT
        assert body["min_markers"] == MIN_MARKERS
        assert body["status"] == "ready"

    def test_health_lists_corner_ids(self):
        body = spatial_client.get("/spatial/health").json()
        assert body["field_corner_ids"] == list(FIELD_CORNER_IDS)

    def test_health_documents_no_fabrication_policy(self):
        body = spatial_client.get("/spatial/health").json()
        assert "never fabricated" in body["note"]


class TestMapEndpoint:

    def test_empty_survey_id_returns_400(self):
        response = spatial_client.post("/spatial/map", json={"survey_id": ""})
        assert response.status_code == 400

    def test_whitespace_survey_id_returns_400(self):
        response = spatial_client.post("/spatial/map", json={"survey_id": "   "})
        assert response.status_code == 400

    def test_missing_field_returns_422(self):
        assert spatial_client.post("/spatial/map", json={}).status_code == 422

    def test_survey_without_detections_returns_400(self, api_db_env):
        response = spatial_client.post("/spatial/map", json={"survey_id": "ghost"})
        assert response.status_code == 400
        assert "no detections" in response.json()["detail"].lower()

    def test_uncalibrated_survey_returns_null_coordinates(self, api_db_env):
        conn = api_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        conn.close()

        response = spatial_client.post("/spatial/map", json={"survey_id": "s1"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["mapping_mode"] == "uncalibrated"
        assert body["total_plants"] == 1
        assert body["mapped_plants"] == 0
        assert body["unmapped_plants"] == 1
        assert body["calibration_id"] is None
        assert body["results"][0]["field_x_m"] is None
        assert body["results"][0]["field_y_m"] is None
        assert body["results"][0]["is_mapped"] is False

    def test_calibrated_survey_returns_metre_coordinates(self, api_db_env):
        conn = api_db_env["connect"]()
        _seed_detection(conn, "s1", 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        _seed_calibration(conn, "s1", "cal_ok")
        conn.close()

        response = spatial_client.post("/spatial/map", json={"survey_id": "s1"})
        assert response.status_code == 200
        body = response.json()
        assert body["mapping_mode"] == "calibrated"
        assert body["calibration_id"] == "cal_ok"
        assert body["mapped_plants"] == 1
        assert body["results"][0]["field_x_m"] == pytest.approx(1.0)
        assert body["results"][0]["field_y_m"] == pytest.approx(2.0)
        assert body["results"][0]["is_mapped"] is True


class TestLocationsEndpoint:

    def test_unknown_survey_returns_empty_list(self, api_db_env):
        response = spatial_client.get("/spatial/locations/nothing_here")
        assert response.status_code == 200
        body = response.json()
        assert body["total_locations"] == 0
        assert body["locations"] == []

    def test_returns_stored_rows(self, api_db_env):
        conn = api_db_env["connect"]()
        save_location(
            conn=conn, survey_id="s1", plant_id=5, image_filename="i1.jpg",
            bbox_cx_px=10.0, bbox_cy_px=20.0, field_x_m=3.0, field_y_m=4.0,
            is_mapped=1, calibration_id="cal_ok", mapping_mode="calibrated",
            model_version="spatial-mapping-v1",
        )
        conn.close()

        response = spatial_client.get("/spatial/locations/s1")
        assert response.status_code == 200
        body = response.json()
        assert body["total_locations"] == 1
        assert body["locations"][0]["plant_id"] == 5
        assert body["locations"][0]["field_x_m"] == pytest.approx(3.0)
        assert body["locations"][0]["field_y_m"] == pytest.approx(4.0)
        assert body["locations"][0]["is_mapped"] is True

class TestCalibrateEndpoint:

    def test_empty_survey_id_returns_400(self, api_db_env):
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "", "image_filename": "calibration.jpg",
            "field_width_m": FIELD_W, "field_height_m": FIELD_H,
        })
        assert response.status_code == 400

    def test_empty_image_filename_returns_400(self, api_db_env):
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "  ",
            "field_width_m": FIELD_W, "field_height_m": FIELD_H,
        })
        assert response.status_code == 400

    @pytest.mark.parametrize("width,height", [(0.0, 8.0), (10.0, 0.0), (-1.0, 8.0)])
    def test_non_positive_field_dimensions_return_400(
        self, api_db_env, width, height
    ):
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "calibration.jpg",
            "field_width_m": width, "field_height_m": height,
        })
        assert response.status_code == 400
        assert "positive" in response.json()["detail"]

    def test_missing_image_returns_404(self, api_db_env, tmp_path, monkeypatch):
        _prepare_survey_image_dir(tmp_path, monkeypatch)
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "calibration.jpg",
            "field_width_m": FIELD_W, "field_height_m": FIELD_H,
        })
        assert response.status_code == 404

    def test_blank_image_returns_400(self, api_db_env, tmp_path, monkeypatch):
        blank = np.full((400, 400, 3), 255, dtype=np.uint8)
        _prepare_survey_image_dir(tmp_path, monkeypatch, image=blank)
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "calibration.jpg",
            "field_width_m": FIELD_W, "field_height_m": FIELD_H,
        })
        assert response.status_code == 400
        assert "refusing to invent" in response.json()["detail"].lower()

    def test_invalid_body_returns_422(self, api_db_env):
        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "calibration.jpg",
        })
        assert response.status_code == 422

    def test_successful_calibration_is_persisted(self, api_db_env, tmp_path,
                                                 monkeypatch, marker_canvas):
        _prepare_survey_image_dir(tmp_path, monkeypatch, image=marker_canvas)

        response = spatial_client.post("/spatial/calibrate", json={
            "survey_id": "survey_001", "image_filename": "calibration.jpg",
            "field_width_m": FIELD_W, "field_height_m": FIELD_H,
            "aruco_dict": DEFAULT_ARUCO_DICT,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["marker_count"] == MIN_MARKERS
        assert body["is_valid"] is True
        assert body["reprojection_error"] == pytest.approx(0.0, abs=1e-6)
        assert body["homography"] is not None

        conn = api_db_env["connect"]()
        try:
            stored = get_calibration(conn, body["calibration_id"])
        finally:
            conn.close()
        assert stored is not None
        assert stored["survey_id"] == "survey_001"
        assert stored["marker_count"] == MIN_MARKERS
        assert stored["is_valid"] == 1

        # The saved calibration is immediately reusable by the mapper.
        conn = api_db_env["connect"]()
        try:
            active = get_latest_calibration_for_survey(conn, "survey_001")
        finally:
            conn.close()
        assert active["calibration_id"] == body["calibration_id"]


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 - Pipeline Stage 5
# ══════════════════════════════════════════════════════════════════════════════

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Minimal connection stub for run_pipeline()'s Stage 1 survey lookup."""

    def __init__(self, row):
        self._row = row

    def execute(self, *args, **kwargs):
        return _FakeCursor(self._row)

    def commit(self):
        pass

    def close(self):
        pass


class _FakeSurveyDir:
    """Stands in for data/images/<survey> so Stage 2 finds no images."""

    def exists(self):
        return False

    def iterdir(self):
        return iter([])

    def __truediv__(self, other):
        return _FakeSurveyDir()


def _stub_pipeline_prerequisites(monkeypatch, survey_id, row):
    """Make Stages 1-4 deterministic so Stage 5 can be tested in isolation."""
    monkeypatch.setattr("app.pipeline.get_connection", lambda: _FakeConn(row))
    monkeypatch.setattr("app.pipeline.IMAGES_ROOT", _FakeSurveyDir())
    from app.severity import service as severity_service
    monkeypatch.setattr(
        severity_service, "estimate_survey",
        lambda sid: (_ for _ in ()).throw(ValueError("no severity fixtures")),
    )


def _pipeline_survey_row(survey_id):
    return (survey_id, "Tomato", "10m x 10m", 3, "2026-07-14T00:00:00")


class TestPipelineStage5:

    def test_stage5_key_always_present(self, monkeypatch):
        _stub_pipeline_prerequisites(monkeypatch, "s_no_det",
                                     _pipeline_survey_row("s_no_det"))
        monkeypatch.setattr(
            spatial_mapper, "map_survey",
            lambda sid, calibration_id=None: (_ for _ in ()).throw(
                ValueError("No detections found")
            ),
        )
        from app.pipeline import run_pipeline
        result = run_pipeline("s_no_det")
        assert "stage_5_spatial_mapping" in result
        assert result["stage_5_spatial_mapping"]["status"] == "skipped"
        assert "reason" in result["stage_5_spatial_mapping"]

    def test_stage5_reports_calibrated_success(self, monkeypatch):
        survey_id = "s_cal"
        _stub_pipeline_prerequisites(monkeypatch, survey_id,
                                     _pipeline_survey_row(survey_id))
        records = [
            {"plant_id": 1, "field_x_m": 1.5, "field_y_m": 2.5,
             "is_mapped": 1, "mapping_mode": "calibrated",
             "calibration_id": "cal_x"},
            {"plant_id": 2, "field_x_m": 3.5, "field_y_m": 4.5,
             "is_mapped": 1, "mapping_mode": "calibrated",
             "calibration_id": "cal_x"},
        ]
        monkeypatch.setattr(
            spatial_mapper, "map_survey",
            lambda sid, calibration_id=None: records,
        )

        from app.pipeline import run_pipeline
        stage5 = run_pipeline(survey_id)["stage_5_spatial_mapping"]

        assert stage5["status"] == "success"
        assert stage5["mapping_mode"] == "calibrated"
        assert stage5["calibration_id"] == "cal_x"
        assert stage5["total_plants"] == 2
        assert stage5["mapped_plants"] == 2
        assert stage5["unmapped_plants"] == 0
        assert [r["plant_id"] for r in stage5["results"]] == [1, 2]
        assert stage5["results"][0]["field_x_m"] == 1.5

    def test_stage5_reports_uncalibrated_success(self, monkeypatch):
        survey_id = "s_uncal"
        _stub_pipeline_prerequisites(monkeypatch, survey_id,
                                     _pipeline_survey_row(survey_id))
        records = [
            {"plant_id": 1, "field_x_m": None, "field_y_m": None,
             "is_mapped": 0, "mapping_mode": "uncalibrated",
             "calibration_id": None},
        ]
        monkeypatch.setattr(
            spatial_mapper, "map_survey",
            lambda sid, calibration_id=None: records,
        )

        from app.pipeline import run_pipeline
        stage5 = run_pipeline(survey_id)["stage_5_spatial_mapping"]

        assert stage5["status"] == "success"
        assert stage5["mapping_mode"] == "uncalibrated"
        assert stage5["calibration_id"] is None
        assert stage5["mapped_plants"] == 0
        assert stage5["unmapped_plants"] == 1
        assert stage5["results"][0]["field_x_m"] is None

    def test_stage5_unexpected_error_does_not_abort_pipeline(self, monkeypatch):
        survey_id = "s_err"
        _stub_pipeline_prerequisites(monkeypatch, survey_id,
                                     _pipeline_survey_row(survey_id))
        monkeypatch.setattr(
            spatial_mapper, "map_survey",
            lambda sid, calibration_id=None: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )

        from app.pipeline import run_pipeline
        result = run_pipeline(survey_id)

        stage5 = result["stage_5_spatial_mapping"]
        assert stage5["status"] == "error"
        assert "boom" in stage5["error"]
        assert "stage_6_treatment_decision" in result
        assert "stage_9_drone_commands" in result

    def test_stage5_real_mapping_through_temp_database(self, monkeypatch,
                                                       temp_db_env):
        survey_id = "sv1"
        conn = temp_db_env["connect"]()
        _seed_detection(conn, survey_id, 1, "i1.jpg", (100.0, 200.0, 300.0, 400.0))
        _seed_calibration(conn, survey_id, "cal_e2e")
        conn.close()

        _stub_pipeline_prerequisites(monkeypatch, survey_id,
                                     _pipeline_survey_row(survey_id))

        from app.pipeline import run_pipeline
        stage5 = run_pipeline(survey_id)["stage_5_spatial_mapping"]

        assert stage5["status"] == "success"
        assert stage5["mapping_mode"] == "calibrated"
        assert stage5["calibration_id"] == "cal_e2e"
        assert stage5["total_plants"] == 1
        assert stage5["mapped_plants"] == 1
        # bbox centre (200, 300) -> ((200-100)*0.01, (300-100)*0.01) = (1.0, 2.0) m
        assert stage5["results"][0]["field_x_m"] == pytest.approx(1.0)
        assert stage5["results"][0]["field_y_m"] == pytest.approx(2.0)

        conn = temp_db_env["connect"]()
        try:
            rows = get_locations_for_survey(conn, survey_id)
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0]["is_mapped"] == 1
