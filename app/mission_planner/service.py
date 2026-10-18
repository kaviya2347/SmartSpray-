"""
app/mission_planner/service.py

Modules 7 + 8 orchestration: turn the Module 6 decisions into a flyable,
capacity-aware spray mission and compare it with blanket spraying.

    spray targets (M6) --group by treatment (M7)--> programmes
      --NN + 2-opt route (M8)--> split into tank loads (M7) --2-opt again-->
      sorties with paths, distances, times, energy
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from app.db import get_connection
from app.decision import rules_engine
from app.decision.database import (
    ensure_decision_tables,
    get_decisions,
    get_mission_plan,
    save_mission_plan,
)
from app.field_survey.service import parse_field_size
from app.mission_planner.route_optimizer import (
    lawnmower_length,
    plan_capacitated_route,
    split_into_sorties,
    tour_length,
)
from app.resource_optimization.grouping import (
    group_by_treatment,
    load_mission_config,
    mix_plan,
)
from app.spatial_mapping.database import (
    ensure_spatial_tables,
    get_latest_calibration_for_survey,
)

logger = logging.getLogger("mission_planner.service")


def _field_dimensions(conn, survey_id: str) -> tuple[float, float, str]:
    cal = get_latest_calibration_for_survey(conn, survey_id)
    if cal:
        return float(cal["field_width_m"]), float(cal["field_height_m"]), "calibration"
    row = conn.execute("SELECT field_size FROM surveys WHERE survey_id = ?",
                       (survey_id,)).fetchone()
    if row:
        w, h = parse_field_size(row[0])
        return w, h, "survey.field_size"
    return 10.0, 10.0, "default"


def build_plan(survey_id: str, decisions: list[dict], field_w: float, field_h: float,
               cfg, treatments: dict) -> dict:
    """Pure planning step (no database) — easy to unit-test."""
    depot = (cfg.depot_x_m, cfg.depot_y_m)
    sprays = [d for d in decisions if d["action"] == rules_engine.SPRAY]
    mapped = [d for d in sprays if d.get("is_mapped")]
    unmapped = [d["plant_id"] for d in sprays if not d.get("is_mapped")]

    plan: dict = {
        "survey_id": survey_id,
        "config": cfg.to_dict(),
        "field": {"width_m": field_w, "height_m": field_h},
        "spray_targets": len(sprays),
        "mapped_spray_targets": len(mapped),
        "unmapped_spray_targets": unmapped,
        "programmes": [],
        "mix_plan": [],
    }
    if not sprays:
        plan["status"] = "no_targets"
        plan["metrics"] = _metrics(plan, [], field_w, field_h, cfg, decisions)
        return plan
    if not mapped:
        plan["status"] = "blocked_uncalibrated"
        plan["message"] = ("Spray targets exist but none has field coordinates. "
                           "Calibrate the survey (POST /spatial/calibrate) and re-run "
                           "Module 5 — positions are never guessed.")
        plan["metrics"] = _metrics(plan, [], field_w, field_h, cfg, decisions)
        # Nothing was planned, so no saving can be claimed yet.
        for key in ("distance_saved_vs_blanket_pct", "time_saved_vs_blanket_pct",
                    "route_improvement_vs_id_order_pct"):
            plan["metrics"][key] = None
        return plan

    groups = group_by_treatment(mapped)
    plan["mix_plan"] = mix_plan(groups, cfg, treatments)
    all_sorties = []
    sortie_no = 0
    for treatment, items in groups.items():
        pts = [(float(t["field_x_m"]), float(t["field_y_m"])) for t in items]
        demands = [float(t["dosage_ml"]) for t in items]
        # Naive comparison: same targets in plant_id order, same tank splits.
        id_order = sorted(range(len(items)), key=lambda i: items[i]["plant_id"])
        naive_chunks = split_into_sorties(id_order, demands, cfg.usable_capacity_ml)
        naive_len = sum(tour_length(pts, c, depot) for c in naive_chunks)
        chunks = plan_capacitated_route(pts, demands, cfg.usable_capacity_ml, depot,
                                        baseline_order=id_order)
        sorties = []
        for chunk in chunks:
            sortie_no += 1
            load = sum(demands[i] for i in chunk)
            spray_s = load / cfg.spray_flow_ml_per_s
            length = tour_length(pts, chunk, depot)
            sortie = {
                "sortie": sortie_no,
                "treatment": treatment,
                "plant_ids": [items[i]["plant_id"] for i in chunk],
                "load_ml": round(load, 1),
                "distance_m": round(length, 2),
                "spray_time_s": round(spray_s, 1),
                "flight_time_s": round(length / cfg.cruise_speed_mps + spray_s
                                       + cfg.takeoff_land_time_s, 1),
                "path": [list(depot)] + [[round(pts[i][0], 3), round(pts[i][1], 3)]
                                         for i in chunk] + [list(depot)],
                "stops": [{"plant_id": items[i]["plant_id"],
                           "x_m": round(pts[i][0], 3), "y_m": round(pts[i][1], 3),
                           "dosage_ml": demands[i],
                           "spray_s": round(demands[i] / cfg.spray_flow_ml_per_s, 2),
                           "disease": items[i]["disease_label"],
                           "severity": items[i].get("severity_category")}
                          for i in chunk],
            }
            sorties.append(sortie)
            all_sorties.append(sortie)
        plan["programmes"].append({
            "treatment": treatment,
            "plants": len(items),
            "sorties": sorties,
            "optimised_distance_m": round(sum(s["distance_m"] for s in sorties), 2),
            "id_order_distance_m": round(naive_len, 2),
        })
    plan["status"] = "ready" if not unmapped else "ready_partial"
    if unmapped:
        plan["message"] = (f"{len(unmapped)} spray target(s) have no field position "
                           "and were left out of the flight plan.")
    plan["metrics"] = _metrics(plan, all_sorties, field_w, field_h, cfg, decisions)
    return plan


def _metrics(plan, sorties, field_w, field_h, cfg, decisions) -> dict:
    n_sorties = len(sorties)
    dist_opt = sum(s["distance_m"] for s in sorties)
    dist_naive = sum(p["id_order_distance_m"] for p in plan["programmes"])
    spray_s = sum(s["spray_time_s"] for s in sorties)
    flight_s = (dist_opt / cfg.cruise_speed_mps + spray_s
                + n_sorties * cfg.takeoff_land_time_s
                + max(0, n_sorties - 1) * cfg.refill_time_s)
    energy_wh = ((dist_opt / cfg.cruise_speed_mps) * cfg.cruise_power_w
                 + spray_s * cfg.hover_power_w) / 3600.0

    blanket_len = lawnmower_length(field_w, field_h, cfg.spray_swath_m,
                                   (cfg.depot_x_m, cfg.depot_y_m))
    rules = rules_engine.load_rules()
    blanket_ml = len(decisions) * rules.blanket_dose_ml
    blanket_loads = max(1, math.ceil(blanket_ml / cfg.usable_capacity_ml)) if blanket_ml else 0
    # Blanket spraying dispenses continuously while flying the lawn-mower
    # pattern, so no extra hover time is added (conservative for SmartSpray).
    blanket_flight_s = (blanket_len / cfg.cruise_speed_mps
                        + blanket_loads * cfg.takeoff_land_time_s
                        + max(0, blanket_loads - 1) * cfg.refill_time_s)
    blanket_energy = (blanket_len / cfg.cruise_speed_mps) * cfg.cruise_power_w / 3600.0

    def pct(saved, base):
        return round(100.0 * saved / base, 1) if base else 0.0

    return {
        "sorties": n_sorties,
        "tank_refills": max(0, n_sorties - 1),
        "optimised_distance_m": round(dist_opt, 2),
        "id_order_distance_m": round(dist_naive, 2),
        "route_improvement_vs_id_order_pct": pct(dist_naive - dist_opt, dist_naive),
        "blanket_distance_m": round(blanket_len, 2),
        "distance_saved_vs_blanket_pct": pct(blanket_len - dist_opt, blanket_len),
        "flight_time_s": round(flight_s, 1),
        "blanket_flight_time_s": round(blanket_flight_s, 1),
        "time_saved_vs_blanket_pct": pct(blanket_flight_s - flight_s, blanket_flight_s),
        "energy_wh": round(energy_wh, 2),
        "battery_used_pct": round(100.0 * energy_wh / cfg.battery_wh, 1),
        "blanket_energy_wh": round(blanket_energy, 2),
        "blanket_tank_loads": blanket_loads,
        "within_single_battery": energy_wh <= cfg.battery_wh,
    }


def plan_mission(survey_id: str, **config_overrides) -> dict:
    """
    Build and store the mission plan for a survey.

    Raises:
        ValueError: if Module 6 has not produced decisions for the survey.
    """
    cfg = load_mission_config(**config_overrides)
    rules = rules_engine.load_rules()
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        ensure_spatial_tables(conn)
        decisions = get_decisions(conn, survey_id)
        if not decisions:
            raise ValueError(
                f"No treatment decisions for survey '{survey_id}'. Run Module 6 first.")
        field_w, field_h, source = _field_dimensions(conn, survey_id)
        plan = build_plan(survey_id, decisions, field_w, field_h, cfg, rules.treatments)
        plan["field"]["source"] = source
        save_mission_plan(conn, survey_id, plan["status"], plan)
    finally:
        conn.close()
    m = plan["metrics"]
    logger.info("Mission '%s': %s, %d sorties, %.1f m (blanket %.1f m)",
                survey_id, plan["status"], m["sorties"], m["optimised_distance_m"],
                m["blanket_distance_m"])
    return plan


def read_mission(survey_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        return get_mission_plan(conn, survey_id)
    finally:
        conn.close()
