"""
tests/test_resource_optimization.py

Module 7 — resource optimisation.
"""

import json

import pytest

from app.resource_optimization.grouping import (
    MissionConfig,
    group_by_treatment,
    load_mission_config,
    mix_plan,
)


def test_default_config_file_loads():
    cfg = load_mission_config()
    assert cfg.tank_capacity_ml > 0
    assert cfg.usable_capacity_ml < cfg.tank_capacity_ml


def test_overrides_and_validation(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"tank_capacity_ml": 500, "unknown_key": 1}))
    cfg = load_mission_config(path, cruise_speed_mps=4.0)
    assert cfg.tank_capacity_ml == 500 and cfg.cruise_speed_mps == 4.0
    with pytest.raises(ValueError):
        load_mission_config(path, tank_capacity_ml=0)


def test_grouping_is_by_treatment_largest_first():
    targets = [{"treatment": "a", "dosage_ml": 1}, {"treatment": "b", "dosage_ml": 1},
               {"treatment": "b", "dosage_ml": 1}]
    groups = group_by_treatment(targets)
    assert list(groups) == ["b", "a"]


def test_mix_plan_volumes_loads_and_cost():
    cfg = MissionConfig(tank_capacity_ml=100, tank_reserve_pct=10)
    groups = {"p": [{"dosage_ml": 60}, {"dosage_ml": 60}]}
    plan = mix_plan(groups, cfg, {"p": {"cost_per_litre_mix_inr": 100, "description": "x"}})
    row = plan[0]
    assert row["mix_needed_ml"] == 120
    assert row["mix_to_prepare_ml"] == 132
    assert row["min_tank_loads"] == 2          # 120 mL / 90 mL usable
    assert row["estimated_cost_inr"] == pytest.approx(12.0)
