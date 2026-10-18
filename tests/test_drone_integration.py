"""
tests/test_drone_integration.py

Module 9 — drone command queue, QGC waypoint export and the executor simulator.
"""

import pytest

from app.drone_integration.command_queue import build_commands, local_to_gps, to_qgc_waypoints
from app.mission_planner.service import build_plan
from app.resource_optimization.grouping import MissionConfig
from simulator.executor_simulator import simulate


def _plan():
    cfg = MissionConfig(tank_capacity_ml=30, tank_reserve_pct=0)
    ds = [{"plant_id": i, "field_x_m": float(i), "field_y_m": 2.0, "is_mapped": True,
           "dosage_ml": 10.0, "treatment": "protectant_fungicide", "action": "spray",
           "disease_label": "early_blight", "severity_category": "high"}
          for i in range(1, 5)]
    return build_plan("s9", ds, 10, 8, cfg,
                      {"protectant_fungicide": {"cost_per_litre_mix_inr": 60}})


def test_commands_follow_design_format():
    cmds = build_commands(_plan())
    sprays = [c for c in cmds if c["type"] == "spray"]
    assert len(sprays) == 4
    for c in sprays:
        assert {"plant_id", "location", "action", "dosage"} <= set(c)
        assert c["action"] == "spray" and len(c["location"]) == 2
    assert [c["seq"] for c in cmds] == list(range(len(cmds)))
    # each sortie: load_tank, takeoff, ..., return_to_depot, land
    assert cmds[0]["type"] == "load_tank" and cmds[-1]["type"] == "land"
    assert sum(1 for c in cmds if c["type"] == "takeoff") == 2   # 40 mL / 30 mL tank


def test_local_to_gps_moves_north_and_east():
    lat0, lon0 = 12.84, 80.15
    lat, lon = local_to_gps(0, 111.32, lat0, lon0)
    assert lat == pytest.approx(lat0 + 0.001, abs=1e-6) and lon == pytest.approx(lon0)
    lat, lon = local_to_gps(10, 0, lat0, lon0)
    assert lon > lon0 and lat == pytest.approx(lat0)


def test_qgc_waypoint_file_structure():
    plan = _plan()
    sortie = plan["programmes"][0]["sorties"][0]
    text = to_qgc_waypoints(sortie, plan["config"])
    lines = text.strip().splitlines()
    assert lines[0] == "QGC WPL 110"
    rows = [l.split("\t") for l in lines[1:]]
    assert all(len(r) == 12 for r in rows)
    assert [int(r[0]) for r in rows] == list(range(len(rows)))
    cmds = [int(r[3]) for r in rows]
    assert cmds[1] == 22 and cmds[-1] == 20            # takeoff ... RTL
    assert cmds.count(216) == 2 * len(sortie["stops"])  # sprayer on/off per plant


def test_simulator_matches_plan_and_reports_no_violations():
    plan = _plan()
    cmds = build_commands(plan)
    summary, events = simulate(cmds, plan, noise_m=0.05, seed=1)
    assert summary["status"] == "completed"
    assert summary["chemical_sprayed_ml"] == pytest.approx(40.0)
    assert summary["distance_flown_m"] == pytest.approx(plan["metrics"]["optimised_distance_m"],
                                                        abs=0.05)
    assert 0 < summary["mean_placement_error_m"] < 0.2
    assert len(events) == len(cmds)


def test_simulator_flags_an_empty_tank():
    plan = _plan()
    cmds = build_commands(plan)
    for c in cmds:
        if c["type"] == "load_tank":
            c["volume_ml"] = 5.0
    summary, _ = simulate(cmds, plan)
    assert summary["status"] == "completed_with_warnings"
    assert any("tank empty" in v for v in summary["safety_violations"])
