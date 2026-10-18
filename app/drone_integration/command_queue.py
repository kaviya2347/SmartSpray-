"""
app/drone_integration/command_queue.py

Module 9 — drone command generation.

Turns a mission plan (Modules 7-8) into:

1. An ordered JSON command queue (stored in drone_commands). Spray commands
   keep the format from the project design:
       {"plant_id": 17, "location": [3.4, 2.2], "action": "spray", "dosage": 9}
   plus the fields an executor needs (seq, sortie, treatment, spray_seconds).

2. One QGroundControl / Mission Planner waypoint file per sortie
   ("QGC WPL 110" text format, MAVLink commands), so the plan can be opened
   in a free ground station and flown in ArduPilot SITL:
       22  MAV_CMD_NAV_TAKEOFF
       16  MAV_CMD_NAV_WAYPOINT   (param1 = hover seconds while spraying)
       216 MAV_CMD_DO_SPRAYER     (param1 = 1 on / 0 off)
       20  MAV_CMD_NAV_RETURN_TO_LAUNCH
   Field metres are converted to latitude/longitude around the configured
   field origin (x = east, y = north). SIMULATION ONLY until validated on
   real hardware.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from app.db import get_connection
from app.decision.database import (
    ensure_decision_tables,
    get_commands,
    get_mission_plan,
    replace_commands,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_ROOT = REPO_ROOT / "data" / "outputs"

EARTH_M_PER_DEG_LAT = 111_320.0
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
CMD_WAYPOINT, CMD_RTL, CMD_TAKEOFF, CMD_SPRAYER = 16, 20, 22, 216


def local_to_gps(x_m: float, y_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat = lat0 + y_m / EARTH_M_PER_DEG_LAT
    lon = lon0 + x_m / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return round(lat, 8), round(lon, 8)


def build_commands(plan: dict) -> list[dict]:
    """Ordered command list for an executor (pure function)."""
    cfg = plan["config"]
    depot = [cfg["depot_x_m"], cfg["depot_y_m"]]
    cmds: list[dict] = []

    def add(**c):
        c["seq"] = len(cmds)
        cmds.append(c)

    for prog in plan.get("programmes", []):
        for s in prog["sorties"]:
            add(type="load_tank", sortie=s["sortie"], treatment=s["treatment"],
                volume_ml=s["load_ml"], location=depot)
            add(type="takeoff", sortie=s["sortie"], altitude_m=cfg["cruise_altitude_m"],
                location=depot)
            for stop in s["stops"]:
                add(type="spray", sortie=s["sortie"], plant_id=stop["plant_id"],
                    location=[stop["x_m"], stop["y_m"]], action="spray",
                    dosage=stop["dosage_ml"], treatment=s["treatment"],
                    spray_seconds=stop["spray_s"], disease=stop["disease"],
                    severity=stop.get("severity"))
            add(type="return_to_depot", sortie=s["sortie"], location=depot)
            add(type="land", sortie=s["sortie"], location=depot)
    return cmds


def to_qgc_waypoints(sortie: dict, cfg: dict) -> str:
    """One sortie -> 'QGC WPL 110' mission text."""
    lat0, lon0 = cfg["origin_lat"], cfg["origin_lon"]
    alt = cfg["cruise_altitude_m"]
    home_lat, home_lon = local_to_gps(cfg["depot_x_m"], cfg["depot_y_m"], lat0, lon0)
    rows = []

    def row(cmd, p1=0.0, lat=0.0, lon=0.0, z=0.0, frame=MAV_FRAME_GLOBAL_RELATIVE_ALT,
            current=0):
        rows.append(f"{len(rows)}\t{current}\t{frame}\t{cmd}\t{p1:.2f}\t0\t0\t0\t"
                    f"{lat:.8f}\t{lon:.8f}\t{z:.2f}\t1")

    row(CMD_WAYPOINT, 0, home_lat, home_lon, 0, frame=0, current=1)   # home
    row(CMD_TAKEOFF, 0, home_lat, home_lon, alt)
    for stop in sortie["stops"]:
        lat, lon = local_to_gps(stop["x_m"], stop["y_m"], lat0, lon0)
        row(CMD_WAYPOINT, 0.0, lat, lon, alt)          # fly to plant
        row(CMD_SPRAYER, 1)                            # sprayer on
        row(CMD_WAYPOINT, stop["spray_s"], lat, lon, alt)  # hover while spraying
        row(CMD_SPRAYER, 0)                            # sprayer off
    row(CMD_RTL)
    return "QGC WPL 110\n" + "\n".join(rows) + "\n"


def generate_commands(survey_id: str, write_files: bool = True) -> dict:
    """
    Build + store the command queue for a survey and export waypoint files.

    Raises:
        ValueError: when no mission plan exists or it is not flyable.
    """
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        plan = get_mission_plan(conn, survey_id)
        if plan is None:
            raise ValueError(f"No mission plan for '{survey_id}'. Run Modules 7-8 first.")
        if plan["status"] not in ("ready", "ready_partial", "no_targets"):
            raise ValueError(f"Mission for '{survey_id}' is not flyable: {plan['status']}. "
                             f"{plan.get('message', '')}".strip())
        commands = build_commands(plan)
        replace_commands(conn, survey_id, commands)
    finally:
        conn.close()

    files = []
    if write_files and commands:
        out_dir = OUTPUTS_ROOT / survey_id / "mission"
        out_dir.mkdir(parents=True, exist_ok=True)
        for prog in plan["programmes"]:
            for s in prog["sorties"]:
                path = out_dir / f"sortie_{s['sortie']:02d}_{s['treatment']}.waypoints"
                path.write_text(to_qgc_waypoints(s, plan["config"]), encoding="utf-8")
                files.append(path.relative_to(REPO_ROOT).as_posix())
        import json
        cmd_path = out_dir / "drone_commands.json"
        cmd_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")
        files.append(cmd_path.relative_to(REPO_ROOT).as_posix())

    spray_cmds = [c for c in commands if c["type"] == "spray"]
    return {
        "status": "success",
        "survey_id": survey_id,
        "total_commands": len(commands),
        "spray_commands": len(spray_cmds),
        "sorties": plan["metrics"]["sorties"],
        "files": files,
        "commands": commands,
    }


def read_commands(survey_id: str) -> list[dict]:
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        return get_commands(conn, survey_id)
    finally:
        conn.close()


def waypoint_text(survey_id: str, sortie_no: int) -> Optional[str]:
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        plan = get_mission_plan(conn, survey_id)
    finally:
        conn.close()
    if not plan:
        return None
    for prog in plan.get("programmes", []):
        for s in prog["sorties"]:
            if s["sortie"] == sortie_no:
                return to_qgc_waypoints(s, plan["config"])
    return None
