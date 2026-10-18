"""
app/decision/service.py

Module 6 orchestration: read evidence from the database, run the rules
engine, persist one treatment decision per plant, return a summary.
"""

import logging
from typing import Optional

from app.db import get_connection
from app.classification.database import ensure_classifications_table
from app.decision import rules_engine
from app.decision.database import (
    ensure_decision_tables,
    get_decisions,
    get_plant_evidence,
    get_recent_sprays,
    replace_decisions,
)
from app.severity.database import ensure_severity_table
from app.spatial_mapping.database import ensure_spatial_tables

logger = logging.getLogger("decision.service")


def _ensure_all(conn) -> None:
    from app.detection.router import _ensure_detections_table

    _ensure_detections_table(conn)
    ensure_classifications_table(conn)
    ensure_severity_table(conn)
    ensure_spatial_tables(conn)
    ensure_decision_tables(conn)


def generate_decisions(survey_id: str, rules_path: Optional[str] = None,
                       ignore_history: bool = False) -> dict:
    """
    Run Module 6 for a survey.

    ignore_history: skip the re-treatment check against other surveys. Used
        by the mission simulator, whose synthetic fields share coordinates with
        each other but are different fields.

    Raises:
        ValueError: if the survey has no detections, or no plant has been
                    classified yet (Module 3 must run first).
    """
    survey_id = (survey_id or "").strip()
    if not survey_id:
        raise ValueError("survey_id must be a non-empty string")
    rules = rules_engine.load_rules(rules_path)

    conn = get_connection()
    try:
        _ensure_all(conn)
        try:
            evidence = get_plant_evidence(conn, survey_id)
        except Exception as exc:  # e.g. detections table missing
            raise ValueError(f"No detections available for '{survey_id}': {exc}") from exc
        if not evidence:
            raise ValueError(
                f"No detections found for survey '{survey_id}'. Run Modules 2-5 first.")
        if not any(e.get("disease_label") for e in evidence):
            raise ValueError(
                f"No classifications found for survey '{survey_id}'. Run Module 3 first.")
        recent = ([] if ignore_history
                  else get_recent_sprays(conn, survey_id, rules.min_retreatment_days))
        decisions = rules_engine.decide_survey(evidence, rules, recent)
        replace_decisions(conn, survey_id, decisions)
    finally:
        conn.close()

    summary = rules_engine.summarise_decisions(decisions, rules)
    logger.info(
        "Decisions for '%s': %s | chemical %.1f mL vs blanket %.1f mL (%.1f%% saved)",
        survey_id, summary["actions"], summary["chemical_used_ml"],
        summary["blanket_baseline_ml"], summary["chemical_saved_pct"])
    return {
        "status": "success",
        "survey_id": survey_id,
        "rules_version": rules.version,
        "rules_status": rules.status,
        "summary": summary,
        "decisions": decisions,
    }


def read_decisions(survey_id: str) -> list[dict]:
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        return get_decisions(conn, survey_id)
    finally:
        conn.close()
