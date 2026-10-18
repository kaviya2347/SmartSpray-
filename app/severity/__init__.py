"""
app/severity/__init__.py

Module 4 — Disease Severity Estimation.

Package marker + selective exports. No business logic.
"""
from app.severity.database import (  # noqa: F401
    ensure_severity_table,
    save_severity,
    get_severity_for_survey,
    get_detections_for_severity,
)

__all__ = [
    "ensure_severity_table",
    "save_severity",
    "get_severity_for_survey",
    "get_detections_for_severity",
]