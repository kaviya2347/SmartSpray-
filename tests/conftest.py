"""Shared pytest setup: import the app package first (applies its import-order guard)."""
import pytest

import app  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _ensure_schema():
    """Create every table from database/schema.sql before any test runs.

    On a fresh clone database/smartspray.db does not exist yet (it is
    git-ignored), so tests that write to the DB directly need the schema first.
    """
    from app.db import init_db

    init_db()
