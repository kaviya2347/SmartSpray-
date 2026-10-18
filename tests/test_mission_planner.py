"""
tests/test_mission_planner.py

Module 8 — route optimisation and mission planning.

    pytest tests/test_mission_planner.py -v
"""

import itertools
import math

import pytest

from app.mission_planner.route_optimizer import (
    lawnmower_length,
    nearest_neighbour,
    optimise_route,
    split_into_sorties,
    tour_length,
    two_opt,
)
from app.mission_planner.service import build_plan
from app.resource_optimization.grouping import MissionConfig


def brute_force(points, depot):
    best = min(itertools.permutations(range(len(points))),
               key=lambda o: tour_length(points, o, depot))
    return tour_length(points, best, depot)


class TestRouteOptimizer:
    def test_empty_and_single(self):
        assert optimise_route([], (0, 0)) == []
        assert optimise_route([(1, 1)], (0, 0)) == [0]

    def test_visits_every_point_once(self):
        pts = [(x * 0.9 % 7, (x * 3.7) % 5) for x in range(25)]
        order = optimise_route(pts, (0, 0))
        assert sorted(order) == list(range(25))

    def test_two_opt_never_worse_than_nearest_neighbour(self):
        pts = [((i * 7919) % 97 / 10, (i * 104729) % 89 / 10) for i in range(30)]
        nn = nearest_neighbour(pts, (0, 0))
        opt = two_opt(pts, nn, (0, 0))
        assert tour_length(pts, opt, (0, 0)) <= tour_length(pts, nn, (0, 0)) + 1e-9

    def test_close_to_optimal_on_small_instances(self):
        pts = [(1, 5), (4, 1), (2, 2), (5, 5), (3, 4), (6, 2), (0.5, 3)]
        got = tour_length(pts, optimise_route(pts, (0, 0)), (0, 0))
        assert got <= brute_force(pts, (0, 0)) * 1.05

    def test_square_is_solved_exactly(self):
        pts = [(0, 1), (1, 1), (1, 0), (0.5, 1)]
        got = tour_length(pts, optimise_route(pts, (0, 0)), (0, 0))
        assert got == pytest.approx(4.0)

    def test_lawnmower_length(self):
        # 10 x 2 field, 1 m swath: 2 passes of 10 m + connections
        length = lawnmower_length(10, 2, 1.0)
        assert length == pytest.approx(0.5 + 10 + 1 + 10 + math.hypot(0, 1.5))

    def test_split_respects_capacity(self):
        order = [0, 1, 2, 3, 4]
        demands = [10, 10, 10, 10, 10]
        sorties = split_into_sorties(order, demands, 25)
        assert sorties == [[0, 1], [2, 3], [4]]
        with pytest.raises(ValueError):
            split_into_sorties([0], [30], 25)
        with pytest.raises(ValueError):
            split_into_sorties([0], [1], 0)


def decision(pid, x, y, dose=5, treatment="protectant_fungicide", action="spray",
             label="early_blight", mapped=True):
    return {"plant_id": pid, "field_x_m": x if mapped else None,
            "field_y_m": y if mapped else None, "is_mapped": mapped,
            "dosage_ml": dose if action == "spray" else 0, "treatment": treatment,
            "action": action, "disease_label": label, "severity_category": "medium"}


class TestBuildPlan:
    cfg = MissionConfig(tank_capacity_ml=20, tank_reserve_pct=0)
    treatments = {"protectant_fungicide": {"cost_per_litre_mix_inr": 60},
                  "late_blight_fungicide": {"cost_per_litre_mix_inr": 95}}

    def test_no_targets(self):
        plan = build_plan("s", [decision(1, 1, 1, action="no_spray")], 10, 8,
                          self.cfg, self.treatments)
        assert plan["status"] == "no_targets"
        assert plan["metrics"]["sorties"] == 0

    def test_blocked_when_nothing_is_mapped(self):
        plan = build_plan("s", [decision(1, 1, 1, mapped=False)], 10, 8,
                          self.cfg, self.treatments)
        assert plan["status"] == "blocked_uncalibrated"
        assert plan["unmapped_spray_targets"] == [1]

    def test_programmes_sorties_and_capacity(self):
        ds = [decision(i, i, 1, dose=8) for i in range(1, 6)]
        ds += [decision(10, 5, 5, dose=10, treatment="late_blight_fungicide",
                        label="late_blight")]
        ds += [decision(20, 3, 3, action="no_spray", label="healthy")]
        plan = build_plan("s", ds, 10, 8, self.cfg, self.treatments)
        assert plan["status"] == "ready"
        names = [p["treatment"] for p in plan["programmes"]]
        assert names == ["protectant_fungicide", "late_blight_fungicide"]
        for p in plan["programmes"]:
            for s in p["sorties"]:
                assert s["load_ml"] <= self.cfg.usable_capacity_ml
                assert s["path"][0] == [0.0, 0.0] and s["path"][-1] == [0.0, 0.0]
        visited = sorted(pid for p in plan["programmes"] for s in p["sorties"]
                         for pid in s["plant_ids"])
        assert visited == [1, 2, 3, 4, 5, 10]
        m = plan["metrics"]
        assert m["sorties"] == 4          # 40 mL at 20 mL/tank + 1 late-blight sortie
        assert m["optimised_distance_m"] <= m["id_order_distance_m"] + 1e-6
        assert m["blanket_distance_m"] > m["optimised_distance_m"]

    def test_partial_when_some_targets_unmapped(self):
        ds = [decision(1, 1, 1), decision(2, 0, 0, mapped=False)]
        plan = build_plan("s", ds, 10, 8, self.cfg, self.treatments)
        assert plan["status"] == "ready_partial"
        assert plan["unmapped_spray_targets"] == [2]


def test_capacitated_route_never_worse_than_baseline():
    from app.mission_planner.route_optimizer import plan_capacitated_route
    pts = [((i * 37) % 11 * 0.9, (i * 53) % 7 * 1.1) for i in range(40)]
    demands = [((i * 7) % 5) + 5 for i in range(40)]
    baseline = list(range(40))
    base_len = sum(tour_length(pts, c, (0, 0)) for c in split_into_sorties(baseline, demands, 60))
    chunks = plan_capacitated_route(pts, demands, 60, (0, 0), baseline_order=baseline)
    assert sorted(i for c in chunks for i in c) == baseline
    assert all(sum(demands[i] for i in c) <= 60 for c in chunks)
    assert sum(tour_length(pts, c, (0, 0)) for c in chunks) <= base_len + 1e-9
