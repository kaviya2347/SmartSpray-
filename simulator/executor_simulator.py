"""
simulator/executor_simulator.py

Software-in-the-loop executor for the Module 9 command queue.

It "flies" the queued commands the way a spot-spraying drone would and
records what happened in execution_logs:

    * point-to-point flight at cruise speed, with seeded GNSS position noise
      (default sigma 0.05 m, typical of an RTK-less small drone hovering low)
    * spraying = hover for dosage / flow-rate seconds; tank level tracked
    * energy model (cruise / hover power) against the battery capacity
    * safety checks: tank never goes negative, battery reserve respected,
      spray point must lie inside the field

Usage:
    python -m simulator.executor_simulator <survey_id> [--noise 0.05] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.db import get_connection  # noqa: E402
from app.decision.database import (  # noqa: E402
    add_execution_log,
    clear_execution_logs,
    ensure_decision_tables,
    get_commands,
    get_mission_plan,
    mark_command_executed,
)

BATTERY_RESERVE_PCT = 20.0


def simulate(commands: list[dict], plan: dict, noise_m: float = 0.05,
             seed: int = 7) -> tuple[dict, list[dict]]:
    """Pure simulation. Returns (summary, events)."""
    cfg = plan["config"]
    field_w = plan["field"]["width_m"]
    field_h = plan["field"]["height_m"]
    rng = random.Random(seed)
    depot = (cfg["depot_x_m"], cfg["depot_y_m"])
    pos = depot
    t = 0.0
    tank = 0.0
    energy_wh = 0.0
    distance = 0.0
    sprayed_ml = 0.0
    events: list[dict] = []
    errors: list[str] = []
    placement_errors: list[float] = []
    battery_wh = cfg["battery_wh"]

    def fly_to(target):
        nonlocal pos, t, energy_wh, distance
        d = math.hypot(target[0] - pos[0], target[1] - pos[1])
        dt = d / cfg["cruise_speed_mps"]
        t += dt
        distance += d
        energy_wh += dt * cfg["cruise_power_w"] / 3600.0
        pos = (target[0], target[1])
        return d, dt

    for c in commands:
        kind = c["type"]
        detail: dict = {"type": kind, "t_s": None}
        if kind == "load_tank":
            tank = min(cfg["tank_capacity_ml"], float(c["volume_ml"]) *
                       (1 + cfg["tank_reserve_pct"] / 100.0))
            if pos != depot:
                errors.append(f"seq {c['seq']}: tank loaded away from depot")
            t += cfg["refill_time_s"] if c["sortie"] > 1 else 0
            detail.update(tank_ml=round(tank, 1), treatment=c["treatment"])
        elif kind in ("takeoff", "land"):
            t += cfg["takeoff_land_time_s"] / 2
            energy_wh += (cfg["takeoff_land_time_s"] / 2) * cfg["hover_power_w"] / 3600.0
        elif kind == "spray":
            target = tuple(c["location"])
            d, dt = fly_to(target)
            actual = (target[0] + rng.gauss(0, noise_m), target[1] + rng.gauss(0, noise_m))
            err = math.hypot(actual[0] - target[0], actual[1] - target[1])
            placement_errors.append(err)
            if not (0 <= target[0] <= field_w and 0 <= target[1] <= field_h):
                errors.append(f"seq {c['seq']}: spray point outside field {target}")
            dose = float(c["dosage"])
            if dose > tank + 1e-6:
                errors.append(f"seq {c['seq']}: tank empty (need {dose}, have {tank:.1f})")
                dose = max(0.0, tank)
            spray_s = dose / cfg["spray_flow_ml_per_s"]
            t += spray_s
            energy_wh += spray_s * cfg["hover_power_w"] / 3600.0
            tank -= dose
            sprayed_ml += dose
            detail.update(plant_id=c["plant_id"], flew_m=round(d, 2),
                          placement_error_m=round(err, 3), dosage_ml=dose,
                          tank_left_ml=round(tank, 1))
        elif kind == "return_to_depot":
            d, _ = fly_to(depot)
            detail.update(flew_m=round(d, 2), tank_left_ml=round(tank, 1))
        detail["t_s"] = round(t, 1)
        detail["battery_used_pct"] = round(100 * energy_wh / battery_wh, 1)
        if 100 * energy_wh / battery_wh > 100 - BATTERY_RESERVE_PCT:
            errors.append(f"seq {c['seq']}: battery reserve breached")
        events.append({"seq": c["seq"], "event": kind, "detail": detail})

    planned_ml = sum(float(c["dosage"]) for c in commands if c["type"] == "spray")
    summary = {
        "commands_executed": len(commands),
        "spray_actions": sum(1 for c in commands if c["type"] == "spray"),
        "mission_time_s": round(t, 1),
        "distance_flown_m": round(distance, 2),
        "chemical_sprayed_ml": round(sprayed_ml, 2),
        "chemical_planned_ml": round(planned_ml, 2),
        "energy_wh": round(energy_wh, 2),
        "battery_used_pct": round(100 * energy_wh / battery_wh, 1),
        "gnss_noise_sigma_m": noise_m,
        "mean_placement_error_m": round(sum(placement_errors) / len(placement_errors), 3)
        if placement_errors else 0.0,
        "max_placement_error_m": round(max(placement_errors), 3) if placement_errors else 0.0,
        "safety_violations": errors,
        "status": "completed" if not errors else "completed_with_warnings",
    }
    return summary, events


def run(survey_id: str, noise_m: float = 0.05, seed: int = 7) -> dict:
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        plan = get_mission_plan(conn, survey_id)
        commands = get_commands(conn, survey_id)
        if plan is None or not commands:
            raise ValueError(f"No commands queued for '{survey_id}'. Run Module 9 first.")
        summary, events = simulate(commands, plan, noise_m, seed)
        clear_execution_logs(conn, survey_id)
        for e in events:
            add_execution_log(conn, survey_id, e["seq"], e["event"], e["detail"])
            mark_command_executed(conn, survey_id, e["seq"], "simulated")
        add_execution_log(conn, survey_id, -1, "summary", summary)
        conn.commit()
    finally:
        conn.close()
    out_dir = REPO_ROOT / "data" / "outputs" / survey_id / "mission"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "simulation_log.json").write_text(
        json.dumps({"summary": summary, "events": events}, indent=2), encoding="utf-8")
    return {"survey_id": survey_id, **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate the SmartSpray drone executor")
    ap.add_argument("survey_id")
    ap.add_argument("--noise", type=float, default=0.05, help="GNSS noise sigma (m)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(json.dumps(run(args.survey_id, args.noise, args.seed), indent=2))


if __name__ == "__main__":
    main()
