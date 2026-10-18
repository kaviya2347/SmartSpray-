"""
app/decision/database.py

Tables owned by Modules 6-9 (decision, resource optimisation, mission
planning, drone command queue) and their data-access helpers.

    treatment_decisions  one row per (survey_id, plant_id)       Module 6
    mission_plans        one row per survey (route + tank plan)  Modules 7-8
    drone_commands       ordered command queue per survey        Module 9
    execution_logs       what the (simulated) executor did       Module 9
"""

import json
import sqlite3
from typing import Any

from app.db import get_connection  # noqa: F401  (re-exported for callers)


def ensure_decision_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS treatment_decisions (
            decision_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id          TEXT    NOT NULL,
            plant_id           INTEGER NOT NULL,
            image_filename     TEXT,
            disease_label      TEXT    NOT NULL,
            confidence         REAL,
            is_uncertain       INTEGER NOT NULL DEFAULT 0,
            severity_score     REAL,
            severity_category  TEXT,
            severity_valid     INTEGER,
            field_x_m          REAL,
            field_y_m          REAL,
            is_mapped          INTEGER NOT NULL DEFAULT 0,
            action             TEXT    NOT NULL,
            dosage_ml          REAL    NOT NULL DEFAULT 0,
            treatment          TEXT,
            reason             TEXT    NOT NULL,
            rules_version      TEXT    NOT NULL,
            decided_at         TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE (survey_id, plant_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mission_plans (
            survey_id        TEXT PRIMARY KEY,
            status           TEXT NOT NULL,
            plan_json        TEXT NOT NULL,
            created_at       TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drone_commands (
            command_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id      TEXT    NOT NULL,
            seq            INTEGER NOT NULL,
            command_json   TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'queued',
            created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            executed_at    TEXT,
            UNIQUE (survey_id, seq)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_logs (
            log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id      TEXT    NOT NULL,
            seq            INTEGER NOT NULL,
            event          TEXT    NOT NULL,
            detail_json    TEXT    NOT NULL,
            logged_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


# ── Inputs for Module 6 ───────────────────────────────────────────────────────

def get_plant_evidence(conn: sqlite3.Connection, survey_id: str) -> list[dict[str, Any]]:
    """
    One row per detected plant with everything Module 6 needs:
    classification (M3), severity (M4) and location (M5). LEFT JOINs so a
    missing downstream result is visible as NULL instead of dropping the plant.
    """
    rows = conn.execute(
        """
        SELECT d.plant_id, d.image_filename,
               c.label, c.confidence, c.is_uncertain,
               s.severity_score, s.severity_category, s.is_valid,
               l.field_x_m, l.field_y_m, l.is_mapped, l.bbox_cx_px, l.bbox_cy_px
        FROM detections d
        LEFT JOIN classifications c
               ON c.survey_id = d.survey_id AND c.plant_id = d.plant_id
        LEFT JOIN plant_severity s
               ON s.survey_id = d.survey_id AND s.plant_id = d.plant_id
        LEFT JOIN plant_locations l
               ON l.survey_id = d.survey_id AND l.plant_id = d.plant_id
        WHERE d.survey_id = ?
        ORDER BY d.plant_id
        """,
        (survey_id,),
    ).fetchall()
    keys = ["plant_id", "image_filename", "disease_label", "confidence",
            "is_uncertain", "severity_score", "severity_category",
            "severity_valid", "field_x_m", "field_y_m", "is_mapped",
            "bbox_cx_px", "bbox_cy_px"]
    return [dict(zip(keys, r)) for r in rows]


def get_recent_sprays(conn: sqlite3.Connection, exclude_survey_id: str,
                      since_days: int) -> list[dict[str, Any]]:
    """
    Mapped spray decisions from OTHER surveys in the last `since_days` days.

    Simulated missions (survey ids starting with "sim_") are synthetic,
    separate fields, so their sprays never count as a previous treatment of
    a real survey's plants.
    """
    rows = conn.execute(
        """
        SELECT survey_id, plant_id, field_x_m, field_y_m, decided_at, treatment
        FROM treatment_decisions
        WHERE survey_id != ? AND action = 'spray' AND is_mapped = 1
          AND substr(survey_id, 1, 4) != 'sim_'
          AND julianday('now') - julianday(decided_at) <= ?
        """,
        (exclude_survey_id, since_days),
    ).fetchall()
    keys = ["survey_id", "plant_id", "field_x_m", "field_y_m", "decided_at", "treatment"]
    return [dict(zip(keys, r)) for r in rows]


# ── Module 6 persistence ──────────────────────────────────────────────────────

DECISION_COLUMNS = [
    "survey_id", "plant_id", "image_filename", "disease_label", "confidence",
    "is_uncertain", "severity_score", "severity_category", "severity_valid",
    "field_x_m", "field_y_m", "is_mapped", "action", "dosage_ml", "treatment",
    "reason", "rules_version",
]


def replace_decisions(conn: sqlite3.Connection, survey_id: str,
                      decisions: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM treatment_decisions WHERE survey_id = ?", (survey_id,))
    placeholders = ", ".join("?" for _ in DECISION_COLUMNS)
    conn.executemany(
        f"INSERT INTO treatment_decisions ({', '.join(DECISION_COLUMNS)}) "
        f"VALUES ({placeholders})",
        [tuple(survey_id if k == "survey_id" else d.get(k) for k in DECISION_COLUMNS)
         for d in decisions],
    )
    conn.commit()


def get_decisions(conn: sqlite3.Connection, survey_id: str) -> list[dict[str, Any]]:
    cols = DECISION_COLUMNS + ["decided_at"]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM treatment_decisions "
        "WHERE survey_id = ? ORDER BY plant_id",
        (survey_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        d["is_uncertain"] = bool(d["is_uncertain"])
        d["is_mapped"] = bool(d["is_mapped"])
        out.append(d)
    return out


# ── Modules 7-9 persistence ───────────────────────────────────────────────────

def save_mission_plan(conn: sqlite3.Connection, survey_id: str, status: str,
                      plan: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO mission_plans (survey_id, status, plan_json, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (survey_id, status, json.dumps(plan)),
    )
    conn.commit()


def get_mission_plan(conn: sqlite3.Connection, survey_id: str) -> dict | None:
    row = conn.execute(
        "SELECT status, plan_json, created_at FROM mission_plans WHERE survey_id = ?",
        (survey_id,),
    ).fetchone()
    if row is None:
        return None
    plan = json.loads(row[1])
    plan.setdefault("status", row[0])
    plan["created_at"] = row[2]
    return plan


def replace_commands(conn: sqlite3.Connection, survey_id: str,
                     commands: list[dict]) -> None:
    conn.execute("DELETE FROM drone_commands WHERE survey_id = ?", (survey_id,))
    conn.executemany(
        "INSERT INTO drone_commands (survey_id, seq, command_json) VALUES (?, ?, ?)",
        [(survey_id, c["seq"], json.dumps(c)) for c in commands],
    )
    conn.commit()


def get_commands(conn: sqlite3.Connection, survey_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT seq, command_json, status, executed_at FROM drone_commands "
        "WHERE survey_id = ? ORDER BY seq",
        (survey_id,),
    ).fetchall()
    out = []
    for seq, cj, status, executed_at in rows:
        c = json.loads(cj)
        c.update(status=status, executed_at=executed_at)
        out.append(c)
    return out


def mark_command_executed(conn: sqlite3.Connection, survey_id: str, seq: int,
                          status: str = "executed") -> None:
    conn.execute(
        "UPDATE drone_commands SET status = ?, executed_at = datetime('now') "
        "WHERE survey_id = ? AND seq = ?",
        (status, survey_id, seq),
    )


def add_execution_log(conn: sqlite3.Connection, survey_id: str, seq: int,
                      event: str, detail: dict) -> None:
    conn.execute(
        "INSERT INTO execution_logs (survey_id, seq, event, detail_json) VALUES (?, ?, ?, ?)",
        (survey_id, seq, event, json.dumps(detail)),
    )


def get_execution_logs(conn: sqlite3.Connection, survey_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT seq, event, detail_json, logged_at FROM execution_logs "
        "WHERE survey_id = ? ORDER BY log_id",
        (survey_id,),
    ).fetchall()
    return [{"seq": s, "event": e, "detail": json.loads(d), "logged_at": t}
            for s, e, d, t in rows]


def clear_execution_logs(conn: sqlite3.Connection, survey_id: str) -> None:
    conn.execute("DELETE FROM execution_logs WHERE survey_id = ?", (survey_id,))
    conn.commit()
