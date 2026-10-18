"""
tests/test_decision.py

Module 6 — Treatment Decision Engine.

    pytest tests/test_decision.py -v
"""

import json
import sqlite3

import pytest

from app.decision import rules_engine
from app.decision.rules_engine import (
    HOLD, INSPECT, MONITOR, NO_SPRAY, SPRAY,
    RuleSet, decide_plant, decide_survey, load_rules, summarise_decisions,
)


@pytest.fixture(scope="module")
def rules() -> RuleSet:
    return load_rules()


def evidence(**kw):
    base = {"plant_id": 1, "image_filename": "img.jpg", "disease_label": "early_blight",
            "confidence": 0.95, "is_uncertain": 0, "severity_score": 0.35,
            "severity_category": "medium", "severity_valid": 1,
            "field_x_m": 2.0, "field_y_m": 3.0, "is_mapped": 1}
    base.update(kw)
    return base


class TestRuleFile:
    def test_rules_file_loads_and_is_complete(self, rules):
        for disease in ("healthy", "early_blight", "late_blight", "septoria_leaf_spot"):
            assert set(rules.rules[disease]) >= {"low", "medium", "high"}
        assert rules.uncertainty_threshold == pytest.approx(0.60)
        assert rules.blanket_dose_ml > 0

    def test_invalid_rule_table_is_rejected(self, rules):
        bad = json.loads(json.dumps(rules.raw))
        bad["rules"]["early_blight"]["high"]["action"] = "explode"
        with pytest.raises(ValueError):
            RuleSet.from_dict(bad)

    def test_spray_without_treatment_is_rejected(self, rules):
        bad = json.loads(json.dumps(rules.raw))
        bad["rules"]["early_blight"]["high"]["treatment"] = None
        with pytest.raises(ValueError):
            RuleSet.from_dict(bad)

    def test_missing_rules_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rules(tmp_path / "nope.json")


class TestDecisions:
    def test_healthy_is_never_sprayed(self, rules):
        d = decide_plant(evidence(disease_label="healthy", severity_category="high"), rules)
        assert d["action"] == NO_SPRAY and d["dosage_ml"] == 0

    @pytest.mark.parametrize("level,action,dose", [
        ("low", MONITOR, 0), ("medium", SPRAY, 5), ("high", SPRAY, 9)])
    def test_early_blight_table(self, rules, level, action, dose):
        d = decide_plant(evidence(severity_category=level), rules)
        assert (d["action"], d["dosage_ml"]) == (action, dose)

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_late_blight_any_severity_full_dose(self, rules, level):
        d = decide_plant(evidence(disease_label="late_blight", severity_category=level), rules)
        assert d["action"] == SPRAY and d["dosage_ml"] == 10
        assert d["treatment"] == "late_blight_fungicide"

    def test_septoria_medium_and_high(self, rules):
        assert decide_plant(evidence(disease_label="septoria_leaf_spot",
                                     severity_category="medium"), rules)["dosage_ml"] == 6
        assert decide_plant(evidence(disease_label="septoria_leaf_spot",
                                     severity_category="high"), rules)["dosage_ml"] == 8

    def test_low_confidence_is_inspected_not_sprayed(self, rules):
        d = decide_plant(evidence(confidence=0.41, severity_category="high"), rules)
        assert d["action"] == INSPECT and d["dosage_ml"] == 0
        assert d["is_uncertain"] is True
        assert "uncertain" in d["reason"].lower()

    def test_uncertain_flag_from_classifier_is_respected(self, rules):
        d = decide_plant(evidence(confidence=0.9, is_uncertain=1), rules)
        assert d["action"] == INSPECT

    def test_missing_classification_is_inspected(self, rules):
        d = decide_plant(evidence(disease_label=None, confidence=None), rules)
        assert d["action"] == INSPECT

    def test_invalid_severity_policy(self, rules):
        eb = decide_plant(evidence(severity_valid=0, severity_category="low"), rules)
        assert eb["action"] == INSPECT
        lb = decide_plant(evidence(disease_label="late_blight", severity_valid=0,
                                   severity_category="low"), rules)
        assert lb["action"] == SPRAY and lb["dosage_ml"] == 10

    def test_location_only_for_mapped_plants(self, rules):
        mapped = decide_plant(evidence(), rules)
        assert mapped["location"] == [2.0, 3.0]
        unmapped = decide_plant(evidence(is_mapped=0, field_x_m=None, field_y_m=None), rules)
        assert unmapped["location"] is None and unmapped["is_mapped"] == 0

    def test_retreatment_interval_holds_recent_position(self, rules):
        recent = [{"survey_id": "old", "plant_id": 9, "field_x_m": 2.1, "field_y_m": 3.05,
                   "decided_at": "2026-09-14 10:00:00", "treatment": "protectant_fungicide"}]
        d = decide_plant(evidence(severity_category="high"), rules, recent)
        assert d["action"] == HOLD and d["dosage_ml"] == 0
        far = [{**recent[0], "field_x_m": 8.0}]
        assert decide_plant(evidence(severity_category="high"), rules, far)["action"] == SPRAY

    def test_every_decision_has_a_reason(self, rules):
        rows = [evidence(plant_id=i, disease_label=lbl)
                for i, lbl in enumerate(["healthy", "early_blight", "late_blight",
                                         "septoria_leaf_spot", None], start=1)]
        for d in decide_survey(rows, rules):
            assert d["reason"] and d["rules_version"] == rules.version


class TestSummary:
    def test_savings_against_blanket_baseline(self, rules):
        rows = [evidence(plant_id=1, disease_label="late_blight"),
                evidence(plant_id=2, disease_label="healthy"),
                evidence(plant_id=3, disease_label="healthy"),
                evidence(plant_id=4, severity_category="high")]
        s = summarise_decisions(decide_survey(rows, rules), rules)
        assert s["chemical_used_ml"] == 19
        assert s["blanket_baseline_ml"] == 4 * rules.blanket_dose_ml
        assert s["chemical_saved_pct"] == pytest.approx(100 * (1 - 19 / 40), abs=0.1)
        assert s["cost_saved_inr"] > 0
        assert s["plants_sprayed"] == 2


class TestServiceAndApi:
    def _seed(self, conn, survey_id):
        from app.classification.database import ensure_classifications_table
        from app.decision.database import ensure_decision_tables
        from app.detection.router import _ensure_detections_table
        from app.severity.database import ensure_severity_table
        from app.spatial_mapping.database import ensure_spatial_tables

        _ensure_detections_table(conn)
        ensure_classifications_table(conn)
        ensure_severity_table(conn)
        ensure_spatial_tables(conn)
        ensure_decision_tables(conn)
        for pid, label, sev in [(1, "late_blight", "high"), (2, "healthy", "low"),
                                (3, "early_blight", "medium")]:
            conn.execute("INSERT INTO detections (survey_id,image_filename,plant_id,bbox_x1,"
                         "bbox_y1,bbox_x2,bbox_y2,confidence,model_version) "
                         "VALUES (?,?,?,?,?,?,?,?,?)",
                         (survey_id, "f.jpg", pid, 0, 0, 10, 10, 0.9, "t"))
            conn.execute("INSERT INTO classifications (survey_id,image_filename,plant_id,bbox_x1,"
                         "bbox_y1,bbox_x2,bbox_y2,label,confidence,all_probs_json,model_version) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (survey_id, "f.jpg", pid, 0, 0, 10, 10, label, 0.97, "{}", "t"))
            conn.execute("INSERT INTO plant_severity (survey_id,plant_id,image_filename,"
                         "disease_label,severity_score,severity_category,leaf_px,lesion_px,"
                         "is_valid,model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (survey_id, pid, "f.jpg", label, 0.3, sev, 100, 30, 1, "t"))
            conn.execute("INSERT INTO plant_locations (survey_id,plant_id,image_filename,"
                         "bbox_cx_px,bbox_cy_px,field_x_m,field_y_m,is_mapped,calibration_id,"
                         "mapping_mode,model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (survey_id, pid, "f.jpg", 5, 5, float(pid), 1.0, 1, "c",
                          "calibrated", "t"))
        conn.commit()

    @pytest.fixture
    def temp_db(self, tmp_path, monkeypatch):
        db = tmp_path / "t.db"
        import app.db as appdb

        monkeypatch.setattr(appdb, "DB_PATH", db)
        conn = sqlite3.connect(db)
        self._seed(conn, "dec_s1")
        conn.close()
        return db

    def test_generate_decisions_persists_rows(self, temp_db):
        from app.decision.service import generate_decisions, read_decisions

        out = generate_decisions("dec_s1")
        assert out["status"] == "success"
        actions = {d["plant_id"]: d["action"] for d in out["decisions"]}
        assert actions == {1: SPRAY, 2: NO_SPRAY, 3: SPRAY}
        assert len(read_decisions("dec_s1")) == 3

    def test_generate_decisions_unknown_survey_raises(self, temp_db):
        from app.decision.service import generate_decisions

        with pytest.raises(ValueError):
            generate_decisions("nope")

    def test_api_generate_and_read(self, temp_db):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        r = client.post("/decisions/generate", json={"survey_id": "dec_s1"})
        assert r.status_code == 200
        assert r.json()["summary"]["plants_sprayed"] == 2
        r2 = client.get("/decisions/dec_s1")
        assert r2.status_code == 200 and len(r2.json()["decisions"]) == 3
        assert client.post("/decisions/generate", json={"survey_id": "zzz"}).status_code == 400
        assert client.get("/decisions/rules/current").status_code == 200
