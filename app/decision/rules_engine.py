"""
app/decision/rules_engine.py

Module 6 — Treatment Decision Engine (pure logic, no database, no HTTP).

Turns per-plant evidence (disease + confidence from M3, severity from M4,
field position from M5) into an explicit, auditable decision:

    {plant_id, location, action, dosage_ml, treatment, reason}

Decision order (first matching rule wins, and the reason says which):
    1. no classification            -> inspect
    2. healthy                      -> no_spray
    3. confidence < threshold       -> inspect   (never spray on a guess)
    4. severity estimate invalid    -> policy (late blight: default spray,
                                       others: inspect)
    5. disease x severity rule      -> spray / monitor / no_spray + dosage
    6. re-treatment interval check  -> hold, if the same field position was
                                       sprayed recently (mapped plants only)

Actions: spray | no_spray | monitor | inspect | hold
The rule table lives in rules/dosage_rules.json so agronomists can audit and
change it without touching code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "dosage_rules.json"

SPRAY, NO_SPRAY, MONITOR, INSPECT, HOLD = "spray", "no_spray", "monitor", "inspect", "hold"
ACTIONS = (SPRAY, NO_SPRAY, MONITOR, INSPECT, HOLD)
SEVERITY_LEVELS = ("low", "medium", "high")


@dataclass
class RuleSet:
    version: str
    rules: dict
    treatments: dict
    blanket_dose_ml: float
    uncertainty_threshold: float
    min_retreatment_days: int
    retreatment_radius_m: float
    invalid_policy: dict
    dosage_unit: str = "mL per plant"
    status: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RuleSet":
        rules = data["rules"]
        for disease, table in rules.items():
            for level in SEVERITY_LEVELS:
                if level not in table:
                    raise ValueError(f"rules[{disease}] missing severity level '{level}'")
                entry = table[level]
                if entry.get("action") not in ACTIONS:
                    raise ValueError(f"rules[{disease}][{level}] has invalid action")
                if float(entry.get("dosage_ml", 0)) < 0:
                    raise ValueError(f"rules[{disease}][{level}] negative dosage")
                if entry["action"] == SPRAY and not entry.get("treatment"):
                    raise ValueError(f"rules[{disease}][{level}] spray without treatment")
        return cls(
            version=data.get("version", "unversioned"),
            rules=rules,
            treatments=data.get("treatments", {}),
            blanket_dose_ml=float(data.get("blanket_dose_ml_per_plant", 10.0)),
            uncertainty_threshold=float(data.get("uncertainty_threshold", 0.6)),
            min_retreatment_days=int(data.get("min_retreatment_interval_days", 7)),
            retreatment_radius_m=float(data.get("retreatment_match_radius_m", 0.35)),
            invalid_policy=data.get("invalid_severity_policy", {"default": INSPECT}),
            dosage_unit=data.get("dosage_unit", "mL per plant"),
            status=data.get("status", ""),
            raw=data,
        )


def load_rules(path: Optional[Path] = None) -> RuleSet:
    path = Path(path) if path else DEFAULT_RULES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dosage rules not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return RuleSet.from_dict(json.load(f))


def _base(evidence: dict) -> dict:
    mapped = bool(evidence.get("is_mapped")) and evidence.get("field_x_m") is not None
    return {
        "plant_id": int(evidence["plant_id"]),
        "image_filename": evidence.get("image_filename"),
        "disease_label": evidence.get("disease_label") or "unclassified",
        "confidence": evidence.get("confidence"),
        "is_uncertain": bool(evidence.get("is_uncertain") or False),
        "severity_score": evidence.get("severity_score"),
        "severity_category": evidence.get("severity_category"),
        "severity_valid": (None if evidence.get("severity_valid") is None
                           else int(bool(evidence.get("severity_valid")))),
        "field_x_m": evidence.get("field_x_m") if mapped else None,
        "field_y_m": evidence.get("field_y_m") if mapped else None,
        "is_mapped": int(mapped),
    }


def _finish(d: dict, action: str, dosage: float, treatment: Optional[str],
            reason: str, rules: RuleSet) -> dict:
    d.update(action=action, dosage_ml=float(dosage), treatment=treatment,
             reason=reason, rules_version=rules.version)
    d["location"] = ([round(d["field_x_m"], 3), round(d["field_y_m"], 3)]
                     if d["is_mapped"] else None)
    return d


def decide_plant(evidence: dict, rules: RuleSet,
                 recent_sprays: Optional[list[dict]] = None) -> dict:
    """Decide the treatment for one plant. Never raises for missing evidence."""
    d = _base(evidence)
    label = evidence.get("disease_label")
    conf = evidence.get("confidence")

    if not label:
        return _finish(d, INSPECT, 0, None,
                       "No disease classification available — manual inspection.", rules)
    if label not in rules.rules:
        return _finish(d, INSPECT, 0, None,
                       f"Unknown label '{label}' — manual inspection.", rules)
    if label == "healthy":
        return _finish(d, NO_SPRAY, 0, None, rules.rules[label]["low"]["reason"], rules)

    uncertain = bool(evidence.get("is_uncertain")) or (
        conf is not None and float(conf) < rules.uncertainty_threshold)
    d["is_uncertain"] = uncertain
    if uncertain:
        return _finish(
            d, INSPECT, 0, None,
            f"Model confidence {float(conf or 0):.2f} < {rules.uncertainty_threshold:.2f} "
            f"for '{label}' — flagged uncertain, no automatic spray.", rules)

    level = evidence.get("severity_category")
    valid = evidence.get("severity_valid")
    if level not in SEVERITY_LEVELS or valid in (0, False):
        policy = rules.invalid_policy.get(label, rules.invalid_policy.get("default", INSPECT))
        if policy == "spray_default":
            entry = rules.rules[label]["medium"]
            d["severity_category"] = level
            return _finish(d, entry["action"], entry["dosage_ml"], entry["treatment"],
                           "Severity could not be measured; "
                           f"{label} is treated by default. " + entry["reason"], rules)
        return _finish(d, INSPECT, 0, None,
                       f"{label} detected but severity could not be measured "
                       "— manual inspection before treating.", rules)

    entry = rules.rules[label][level]
    action, dosage, treatment = entry["action"], entry["dosage_ml"], entry.get("treatment")
    reason = entry["reason"]
    if evidence.get("severity_score") is not None:
        reason = f"{reason} (measured severity {float(evidence['severity_score']):.0%})"

    if action == SPRAY and d["is_mapped"] and recent_sprays:
        for prev in recent_sprays:
            if prev.get("field_x_m") is None:
                continue
            dist = math.hypot(prev["field_x_m"] - d["field_x_m"],
                              prev["field_y_m"] - d["field_y_m"])
            if dist <= rules.retreatment_radius_m:
                return _finish(
                    d, HOLD, 0, treatment,
                    f"Would spray ({reason}) but this position was sprayed on "
                    f"{prev['decided_at']} (survey {prev['survey_id']}), inside the "
                    f"{rules.min_retreatment_days}-day re-treatment interval.", rules)

    return _finish(d, action, dosage, treatment, reason, rules)


def decide_survey(evidence_rows: list[dict], rules: RuleSet,
                  recent_sprays: Optional[list[dict]] = None) -> list[dict]:
    return [decide_plant(e, rules, recent_sprays) for e in evidence_rows]


def summarise_decisions(decisions: list[dict], rules: RuleSet) -> dict[str, Any]:
    """
    Pesticide use vs. the uniform (blanket) baseline in which every detected
    plant receives the blanket dose — the savings metric from the design.
    """
    n = len(decisions)
    counts = {a: sum(1 for d in decisions if d["action"] == a) for a in ACTIONS}
    by_disease: dict[str, int] = {}
    for d in decisions:
        by_disease[d["disease_label"]] = by_disease.get(d["disease_label"], 0) + 1
    actual_ml = sum(d["dosage_ml"] for d in decisions if d["action"] == SPRAY)
    baseline_ml = n * rules.blanket_dose_ml

    cost_actual = 0.0
    for d in decisions:
        if d["action"] == SPRAY and d.get("treatment"):
            price = rules.treatments.get(d["treatment"], {}).get("cost_per_litre_mix_inr", 0.0)
            cost_actual += d["dosage_ml"] / 1000.0 * price
    # Blanket spraying uses the most expensive programme on every plant
    # (farmers spray a broad programme when they cannot see which disease is where).
    max_price = max([t.get("cost_per_litre_mix_inr", 0.0)
                     for t in rules.treatments.values()] or [0.0])
    cost_baseline = baseline_ml / 1000.0 * max_price

    return {
        "total_plants": n,
        "actions": counts,
        "plants_by_label": by_disease,
        "plants_sprayed": counts[SPRAY],
        "plants_needing_inspection": counts[INSPECT],
        "chemical_used_ml": round(actual_ml, 2),
        "blanket_baseline_ml": round(baseline_ml, 2),
        "chemical_saved_ml": round(baseline_ml - actual_ml, 2),
        "chemical_saved_pct": round(100.0 * (1 - actual_ml / baseline_ml), 1) if baseline_ml else 0.0,
        "cost_actual_inr": round(cost_actual, 2),
        "cost_blanket_inr": round(cost_baseline, 2),
        "cost_saved_inr": round(cost_baseline - cost_actual, 2),
        "dosage_unit": rules.dosage_unit,
        "rules_version": rules.version,
        "baseline_definition": (f"uniform spraying: every detected plant receives "
                                f"{rules.blanket_dose_ml:g} {rules.dosage_unit}"),
    }
