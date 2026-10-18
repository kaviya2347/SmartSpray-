"""
tests/test_field_survey.py

Automated tests for Module 1.
"""

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.db import get_connection, init_db

client = TestClient(app)

TEST_SURVEY_ID = "survey_test_001"
TEST_IMAGES_FOLDER = Path("data/images") / TEST_SURVEY_ID


def setup_module(module):
    """Runs once before these tests. Creates a fake survey folder with
    3 dummy files, so we don't need real photos to test the logic."""
    init_db()
    TEST_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (TEST_IMAGES_FOLDER / f"photo_{i}.jpg").write_bytes(b"fake image data")


def teardown_module(module):
    """Runs once after these tests finish. Cleans up the fake folder and
    test rows, so re-running the test suite doesn't fail with
    'already registered'."""
    shutil.rmtree(TEST_IMAGES_FOLDER, ignore_errors=True)
    connection = get_connection()
    for table in ("surveys", "survey_images"):
        connection.execute(
            f"DELETE FROM {table} WHERE survey_id IN (?, ?)",
            (TEST_SURVEY_ID, TEST_SURVEY_ID + "_dup"),
        )
    connection.commit()
    connection.close()


def test_register_survey_success():
    response = client.post("/survey/register", json={"survey_id": TEST_SURVEY_ID})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["survey_id"] == TEST_SURVEY_ID
    assert body["num_images"] == 3


def test_register_survey_missing_folder():
    response = client.post("/survey/register", json={"survey_id": "does_not_exist"})
    assert response.status_code == 404


def test_register_survey_duplicate():
    dup_id = TEST_SURVEY_ID + "_dup"
    (Path("data/images") / dup_id).mkdir(parents=True, exist_ok=True)
    (Path("data/images") / dup_id / "photo_0.jpg").write_bytes(b"fake")

    first = client.post("/survey/register", json={"survey_id": dup_id})
    second = client.post("/survey/register", json={"survey_id": dup_id})

    assert first.status_code == 200
    assert second.status_code == 400

    shutil.rmtree(Path("data/images") / dup_id, ignore_errors=True)


def test_quality_check_reports_unreadable_and_blurry(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from app.field_survey import service

    monkeypatch.setattr(service, "IMAGES_ROOT", tmp_path)
    folder = tmp_path / "survey_quality_001"
    folder.mkdir()
    sharp = (np.indices((200, 200)).sum(axis=0) % 2 * 255).astype(np.uint8)   # checkerboard
    cv2.imwrite(str(folder / "sharp.png"), sharp)
    cv2.imwrite(str(folder / "flat.png"), np.full((200, 200), 128, np.uint8))
    (folder / "broken.jpg").write_bytes(b"not an image")

    assert service.assess_image(folder / "sharp.png")["is_blurry"] is False
    assert service.assess_image(folder / "flat.png")["is_blurry"] is True
    assert service.assess_image(folder / "broken.jpg")["is_readable"] is False
    try:
        result = service.register_survey("survey_quality_001", field_width_m=12, field_height_m=6)
        assert result["num_images"] == 3
        assert result["readable_images"] == 2
        assert result["blurry_images"] == ["flat.png"]
        assert result["unreadable_images"] == ["broken.jpg"]
        assert result["field_size"] == "12m x 6m"
        assert service.parse_field_size(result["field_size"]) == (12.0, 6.0)
    finally:
        conn = get_connection()
        conn.execute("DELETE FROM surveys WHERE survey_id = 'survey_quality_001'")
        conn.execute("DELETE FROM survey_images WHERE survey_id = 'survey_quality_001'")
        conn.commit()
        conn.close()


def test_register_rejects_path_like_ids():
    response = client.post("/survey/register", json={"survey_id": "../etc"})
    assert response.status_code == 400
