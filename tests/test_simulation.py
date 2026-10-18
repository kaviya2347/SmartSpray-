"""
Tests for the drone mission simulator backend (app/simulation) and the
re-planning helpers it adds to Module 8.

Tests that need the PlantVillage dataset or the trained classifier skip
themselves when those files are missing.
"""

import math
import random
from pathlib import Path

import pytest

from app.mission_planner.route_optimizer import (
    lawnmower_length,
    path_length,
    plan_from_position,
    two_opt_path,
)
from app.simulation import service
from app.simulation.service import _serpentine, find_row_gaps
from app.simulation.world import DATASET_ROOT, MANIFEST, PRESETS, make_world

REPO = Path(__file__).resolve().parent.parent
HAS_DATA = MANIFEST.exists() and DATASET_ROOT.exists()
HAS_MODEL = (REPO / "models" / "classification" / "best.keras").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="PlantVillage dataset / split manifest missing")
needs_model = pytest.mark.skipif(not (HAS_DATA and HAS_MODEL), reason="dataset or classifier missing")


# ── world ────────────────────────────────────────────────────────────────────

@needs_data
def test_demo_layout_matches_scripted_demo():
    w = make_world("sim_test_layout", seed=7, preset="small")
    assert len(w["plants"]) == 63
    assert w["label_counts"] == {"healthy": 44, "early_blight": 7, "late_blight": 6,
                                 "septoria_leaf_spot": 6}
    assert all(p["source_split"] == "test" for p in w["plants"])
    assert len({p["source_image"] for p in w["plants"]}) == 63        # no photo reused


@needs_data
def test_random_patches_cover_every_disease():
    w = make_world("sim_test_random", seed=3, preset="medium")
    p = PRESETS["medium"]
    assert len(w["plants"]) == p["rows"] * p["cols"]
    for label in ("late_blight", "early_blight", "septoria_leaf_spot"):
        assert w["label_counts"].get(label, 0) >= 4
    xs = [pl["x_m"] for pl in w["plants"]]
    ys = [pl["y_m"] for pl in w["plants"]]
    assert 0 < min(xs) and max(xs) < p["width_m"] and 0 < min(ys) and max(ys) < p["height_m"]


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        make_world("sim_x", 1, "huge")


# ── row / gap check and scan order ───────────────────────────────────────────

def _grid(rows=4, cols=6, skip=()):
    return [(1.0 + c, 1.0 + r) for r in range(rows) for c in range(cols) if (r, c) not in skip]


def test_no_gaps_in_a_full_grid():
    assert find_row_gaps(_grid(), 8.0, 6.0) == []


def test_gap_inside_a_row_is_found():
    gaps = find_row_gaps(_grid(skip={(2, 3)}), 8.0, 6.0)
    assert len(gaps) == 1
    assert math.dist(gaps[0], (4.0, 3.0)) < 0.05


def test_missing_first_plant_of_a_row_is_found():
    gaps = find_row_gaps(_grid(skip={(1, 0)}), 8.0, 6.0)
    assert any(math.dist(g, (1.0, 2.0)) < 0.05 for g in gaps)


def test_serpentine_alternates_direction():
    targets = [{"target_id": f"t{i}", "x_m": x, "y_m": y} for i, (x, y) in enumerate(_grid(3, 4))]
    order = _serpentine(targets)
    ys = [t["y_m"] for t in order]
    assert ys == sorted(ys)                                  # south to north
    row0 = [t["x_m"] for t in order[:4]]
    row1 = [t["x_m"] for t in order[4:8]]
    assert row0 == sorted(row0) and row1 == sorted(row1, reverse=True)


# ── re-planning from the drone's position ────────────────────────────────────

def test_two_opt_path_never_longer():
    rnd = random.Random(4)
    pts = [(rnd.uniform(0, 10), rnd.uniform(0, 8)) for _ in range(12)]
    order = list(range(12))
    start, end = (5.0, 4.0), (-2.0, -1.6)
    better = two_opt_path(pts, order, start, end)
    assert sorted(better) == order
    assert path_length(pts, better, start, end) <= path_length(pts, order, start, end) + 1e-9


def test_plan_from_position_respects_tank_and_covers_all():
    rnd = random.Random(1)
    pts = [(rnd.uniform(0, 10), rnd.uniform(0, 8)) for _ in range(15)]
    dem = [rnd.choice([5, 6, 8, 9, 10]) for _ in pts]
    r = plan_from_position(pts, dem, (5.0, 4.0), 30.0, (-2.0, -1.6), 237.5)
    assert sum(dem[i] for i in r["first"]) <= 30.0 + 1e-6
    visited = r["first"] + [i for c in r["later"] for i in c]
    assert sorted(visited) == list(range(15))
    for chunk in r["later"]:
        assert sum(dem[i] for i in chunk) <= 237.5


def test_plan_from_position_empty_tank_goes_home_first():
    r = plan_from_position([(1, 1), (2, 2)], [5, 5], (5.0, 5.0), 0.0, (0.0, 0.0), 100)
    assert r["first"] == [] and len(r["later"]) == 1


def test_lawnmower_ignores_centimetre_overhang():
    assert lawnmower_length(10, 8.015, 1.0, (-2, -1.6)) == pytest.approx(
        lawnmower_length(10, 8.0, 1.0, (-2, -1.6)))


# ── rendering → real calibration and detection ──────────────────────────────

@needs_data
def test_overview_photo_gives_boundary_and_plants():
    from app.detection.detector import ClassicalPlantDetector
    from app.simulation.world import FieldRenderer
    from app.spatial_mapping.calibration import get_calibrator

    w = make_world("sim_test_render", seed=7, preset="small")
    r = FieldRenderer(w)
    drops = [{"marker_id": i, "x_m": c[0], "y_m": c[1]} for i, c in enumerate(w["planned_corners"])]
    img = r.overview(drops)
    cal = get_calibrator().calibrate("sim_test_render", "overview.jpg", img, 10.0, 8.0)
    assert cal.is_valid and cal.marker_count == 4
    dets = ClassicalPlantDetector(kernel_frac=service.OVERVIEW_DETECTOR_KERNEL).detect_array(img)
    assert len(dets) >= 60
    close = r.closeup(drops, w["plants"][0]["x_m"], w["plants"][0]["y_m"])
    assert service._locate_in_closeup(close) is not None


# ── API flow ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    created = []
    with TestClient(app) as c:
        c.created = created
        yield c
    for sim_id in created:
        service.delete_simulation(sim_id)


def test_bad_ids_and_paths_are_rejected(client):
    assert client.post("/sim/not_a_sim/overview").status_code in (400, 404)
    assert client.get("/sim/sim_nope/files/../../app/main.py").status_code == 404
    assert client.get("/sim/aruco/9.png").status_code == 404
    assert client.get("/sim/aruco/0.png").headers["content-type"] == "image/png"


@needs_data
def test_overview_requires_all_markers(client):
    s = client.post("/sim/session", json={"seed": 7, "preset": "small"}).json()
    client.created.append(s["sim_id"])
    client.post(f"/sim/{s['sim_id']}/marker", json={"marker_id": 0, "x_m": 0, "y_m": 0})
    r = client.post(f"/sim/{s['sim_id']}/overview")
    assert r.status_code == 400


@needs_model
def test_full_mission_through_the_api(client):
    s = client.post("/sim/session", json={"seed": 7, "preset": "small"}).json()
    sim = s["sim_id"]
    client.created.append(sim)
    for i, (x, y) in enumerate(s["world"]["planned_corners"]):
        assert client.post(f"/sim/{sim}/marker", json={"marker_id": i, "x_m": x, "y_m": y}).status_code == 200
    ov = client.post(f"/sim/{sim}/overview").json()
    assert ov["calibration_valid"] and ov["markers_found"] == 4
    assert ov["eval"]["recall"] >= 0.95
    assert ov["eval"]["mean_position_error_m"] < 0.15
    results = []
    for t in ov["scan_targets"]:
        r = client.post(f"/sim/{sim}/capture", json={"target_id": t["target_id"], "drone_x_m": t["x_m"],
                                                      "drone_y_m": t["y_m"], "gradcam": False})
        assert r.status_code == 200
        results.append(r.json())
    found = [r for r in results if r["found"]]
    assert len(found) >= 60
    assert all(r["decision"]["action"] in ("spray", "no_spray", "monitor", "inspect") for r in found)
    plan = client.post(f"/sim/{sim}/plan").json()
    assert plan["status"].startswith("ready")
    sprays = {d["plant_id"]: d for d in plan["decisions"] if d["action"] == "spray"}
    planned = [p for so in plan["sorties"] for p in so["plant_ids"]]
    assert sorted(planned) == sorted(sprays)                     # every target exactly once
    assert plan["metrics"]["optimised_distance_m"] < plan["metrics"]["blanket_distance_m"]
    assert plan["waypoint_files"]
    first = plan["sorties"][0]
    rp = client.post(f"/sim/{sim}/replan", json={
        "position": [5.0, 4.0], "tank_ml": first["load_ml"] / 2, "tank_treatment": first["treatment"],
        "remaining": planned}).json()
    replanned = [st["plant_id"] for so in rp["sorties"] for st in so["stops"]]
    assert sorted(replanned) == sorted(planned)
    assert rp["sorties"][0]["from_current_position"]
    fin = client.post(f"/sim/{sim}/finish", json={"telemetry": {
        "sprayed_ml": {str(k): v["dosage_ml"] for k, v in sprays.items()},
        "spray_route_m": plan["metrics"]["optimised_distance_m"], "flight_time_s": 300}}).json()
    assert fin["targets_fully_sprayed"] == len(sprays)
    assert fin["chemical"]["used_ml"] < fin["chemical"]["blanket_ml"]
    assert fin["classification"]["accuracy"] >= 0.85
    assert client.get(f"/sim/{sim}/files/flight_log.json").status_code == 200
    assert client.get(f"/drone/mission/{sim}/1.waypoints").status_code == 200


def test_simulated_sprays_do_not_trigger_retreatment_holds():
    """A simulated mission's sprays must not put a real survey's plants on hold."""
    import sqlite3

    from app.decision.database import (DECISION_COLUMNS, ensure_decision_tables,
                                       get_recent_sprays, replace_decisions)

    conn = sqlite3.connect(":memory:")
    ensure_decision_tables(conn)
    row = {k: None for k in DECISION_COLUMNS}
    row.update(plant_id=1, disease_label="early_blight", action="spray", is_mapped=1,
               field_x_m=1.0, field_y_m=1.0, dosage_ml=5.0, treatment="protectant_fungicide",
               reason="test", rules_version="test", is_uncertain=0, image_filename="x.jpg",
               severity_valid=1)
    replace_decisions(conn, "sim_20260101_000000_000", [row])
    replace_decisions(conn, "real_survey_a", [dict(row)])
    recent = get_recent_sprays(conn, "real_survey_b", 7)
    assert [r["survey_id"] for r in recent] == ["real_survey_a"]
