"""
app/resource_optimization/grouping.py

Module 7 — resource optimisation (pure functions).

* Groups spray targets by treatment programme — one tank mix at a time.
* Works out how much mix to prepare per programme (with a small safety
  reserve), how many tank loads (sorties) are needed, and the cost.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MISSION_CONFIG = REPO_ROOT / "rules" / "mission_config.json"


@dataclass
class MissionConfig:
    tank_capacity_ml: float = 250.0
    tank_reserve_pct: float = 5.0          # never plan to empty the tank completely
    cruise_speed_mps: float = 3.0
    spray_flow_ml_per_s: float = 5.0
    refill_time_s: float = 60.0
    takeoff_land_time_s: float = 20.0
    cruise_altitude_m: float = 2.5
    spray_swath_m: float = 1.0             # for the blanket lawn-mower baseline
    depot_x_m: float = 0.0
    depot_y_m: float = 0.0
    origin_lat: float = 12.8406            # field origin for waypoint export
    origin_lon: float = 80.1534            # (placeholder: VIT Chennai campus)
    battery_wh: float = 90.0
    cruise_power_w: float = 180.0
    hover_power_w: float = 160.0

    @property
    def usable_capacity_ml(self) -> float:
        return self.tank_capacity_ml * (1 - self.tank_reserve_pct / 100.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usable_capacity_ml"] = self.usable_capacity_ml
        return d


def load_mission_config(path: Path | None = None, **overrides) -> MissionConfig:
    path = Path(path) if path else DEFAULT_MISSION_CONFIG
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = {k: v for k, v in json.load(f).items()
                    if k in MissionConfig.__dataclass_fields__}
    data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = MissionConfig(**data)
    if cfg.tank_capacity_ml <= 0 or cfg.cruise_speed_mps <= 0 or cfg.spray_flow_ml_per_s <= 0:
        raise ValueError("tank capacity, speed and flow rate must be positive")
    return cfg


def group_by_treatment(targets: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for t in targets:
        groups.setdefault(t.get("treatment") or "unspecified", []).append(t)
    # Deterministic order: largest programme first.
    return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def mix_plan(groups: dict[str, list[dict]], cfg: MissionConfig,
             treatments_catalogue: dict) -> list[dict]:
    plan = []
    for name, items in groups.items():
        total = sum(float(i["dosage_ml"]) for i in items)
        sorties = max(1, math.ceil(total / cfg.usable_capacity_ml)) if total > 0 else 0
        prepare = math.ceil(total * (1 + cfg.tank_reserve_pct / 100.0))
        price = treatments_catalogue.get(name, {}).get("cost_per_litre_mix_inr", 0.0)
        plan.append({
            "treatment": name,
            "description": treatments_catalogue.get(name, {}).get("description", ""),
            "plants": len(items),
            "mix_needed_ml": round(total, 1),
            "mix_to_prepare_ml": prepare,
            "min_tank_loads": sorties,
            "estimated_cost_inr": round(total / 1000.0 * price, 2),
        })
    return plan
