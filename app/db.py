"""
app/db.py

The single point of contact between SmartSpray and its SQLite database.
"""

import sqlite3
from pathlib import Path

# Path(__file__) = the location of THIS file (app/db.py) on your disk.
# .resolve() turns it into a full, unambiguous path.
# .parent.parent walks up two folders: app/ -> SmartSpray/
# This means the database file always lands in the same place,
# no matter which folder you happen to run a command from.
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "smartspray.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """
    Opens the database file. If smartspray.db doesn't exist yet,
    sqlite3.connect() creates it automatically — empty, with no tables.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)


def init_db() -> None:
    """
    Makes sure every table in schema.sql actually exists in the database.
    Safe to call every single time the app starts, because schema.sql
    uses "CREATE TABLE IF NOT EXISTS" — it will never delete your data.
    """
    connection = get_connection()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()
    connection.executescript(schema_sql)   # runs the whole SQL file at once
    # Module 1 per-image quality table.
    from app.field_survey.service import _ensure_images_table
    _ensure_images_table(connection)
    # Module 2 detections table.
    from app.detection.router import _ensure_detections_table
    _ensure_detections_table(connection)
    # Module 3 classifications table (+ is_uncertain migration).
    from app.classification.database import ensure_classifications_table
    ensure_classifications_table(connection)
    # Module 4 severity columns/table (idempotent; also asserted at runtime).
    from app.severity.database import ensure_severity_table
    ensure_severity_table(connection)
    # Module 5 spatial mapping tables (idempotent; also asserted at runtime).
    from app.spatial_mapping.database import ensure_spatial_tables
    ensure_spatial_tables(connection)
    # Module 6-9 decision / mission / drone tables.
    from app.decision.database import ensure_decision_tables
    ensure_decision_tables(connection)
    connection.commit()                     # saves the change permanently
    connection.close()                      # always close what you open