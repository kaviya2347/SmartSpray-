"""
app/simulation/service.py

Mission steps for the drone simulator. The browser flies the drone; each
step below is what the drone's camera and onboard software do at that point
of the flight. Every step goes through the real SmartSpray modules and
writes to the same SQLite tables, so a simulated mission can also be opened
in the dashboard, the API and the QGroundControl export afterwards.

    1. new_session        synthetic field + home pad           (world.py)
    2. drop_marker x4     drone drops an ArUco marker at each field corner
    3. capture_overview   high-altitude photo -> M1 register + quality check,
                          M5 ArUco calibration (boundary and size),
                          M2 plant detection -> plant positions in metres,
                          row/gap check for plants the detector may have missed
    4. capture_plant      low-altitude photo of one plant -> M2 locate,
                          M3 disease + calibrated confidence (+ Grad-CAM),
                          M4 severity, M6 decision and dose for that plant
    5. plan               M6 decisions for the field, M7 tank mixes,
                          M8 shortest capacity-aware route, M9 commands
    6. replan             after a manual override: shortest route from where
                          the drone is now, with what is left in the tank
    7. finish             flight log + summary, checked against ground truth
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.db import get_connection, init_db
from app.simulation.world import (
    CLOSEUP_VIEW_M,
    TEXTURE_PPM,
    FieldRenderer,
    make_world,
)

logger = logging.getLogger("simulation.service")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_ROOT = REPO_ROOT / "data" / "images"
OUTPUTS_ROOT = REPO_ROOT / "data" / "outputs"
OVERVIEW_NAME = "overview.jpg"

SIM_ID_RE = re.compile(r"^sim_[A-Za-z0-9_]{1,60}$")
MATCH_RADIUS_M = 0.45          # detection <-> true plant matching radius
CLOSEUP_SEARCH_M = 0.35        # plant must be this close to the photo centre
OVERVIEW_DETECTOR_KERNEL = 0.004

_SESSIONS: dict[str, "Session"] = {}
_LOCK = threading.Lock()


# ── Session state ────────────────────────────────────────────────────────────

@dataclass
class Session:
    world: dict
    drops: list = field(default_factory=list)
    overview: Optional[dict] = None
    plants: dict = field(default_factory=dict)       # plant_id -> info
    targets: dict = field(default_factory=dict)      # target_id -> scan target
    captures: dict = field(default_factory=dict)     # plant_id -> capture summary
    plan: Optional[dict] = None
    replans: list = field(default_factory=list)
    summary: Optional[dict] = None
    renderer: Optional[FieldRenderer] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def sim_id(self) -> str:
        return self.world["sim_id"]

    def out_dir(self) -> Path:
        d = OUTPUTS_ROOT / self.sim_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self) -> None:
        state = {k: getattr(self, k) for k in
                 ("world", "drops", "overview", "plants", "targets", "captures",
                  "plan", "replans", "summary")}
        state["plants"] = {str(k): v for k, v in self.plants.items()}
        state["captures"] = {str(k): v for k, v in self.captures.items()}
        (self.out_dir() / "sim_state.json").write_text(json.dumps(state, indent=1),
                                                       encoding="utf-8")

    @classmethod
    def load(cls, sim_id: str) -> Optional["Session"]:
        path = OUTPUTS_ROOT / sim_id / "sim_state.json"
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        s = cls(world=state["world"])
        for k in ("drops", "overview", "targets", "plan", "replans", "summary"):
            setattr(s, k, state.get(k) or ([] if k in ("drops", "replans") else
                                           ({} if k == "targets" else None)))
        s.plants = {int(k): v for k, v in (state.get("plants") or {}).items()}
        s.captures = {int(k): v for k, v in (state.get("captures") or {}).items()}
        return s

    def render(self) -> FieldRenderer:
        if self.renderer is None:
            self.renderer = FieldRenderer(self.world)
        return self.renderer


def validate_sim_id(sim_id: str) -> str:
    if not isinstance(sim_id, str) or not SIM_ID_RE.match(sim_id):
        raise ValueError("invalid simulation id")
    return sim_id


def get_session(sim_id: str) -> Session:
    validate_sim_id(sim_id)
    with _LOCK:
        s = _SESSIONS.get(sim_id)
        if s is None:
            s = Session.load(sim_id)
            if s is None:
                raise KeyError(f"simulation '{sim_id}' not found")
            _SESSIONS[sim_id] = s
        return s


def _jpeg_data_url(img_bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _nearest_true(world: dict, x: float, y: float) -> Optional[dict]:
    best, best_d = None, MATCH_RADIUS_M
    for p in world["plants"]:
        d = math.hypot(p["x_m"] - x, p["y_m"] - y)
        if d < best_d:
            best, best_d = p, d
    return best


# ── 1. New field ─────────────────────────────────────────────────────────────

def _warm_up() -> None:
    """Load the classifier and compile its fast paths while the drone takes off."""
    try:
        from app.classification.classifier import get_classifier
        from app.classification.explain import grad_cam

        clf = get_classifier()
        dummy = np.full((190, 190, 3), 120, np.uint8)
        clf.predict(dummy)
        grad_cam(clf.model, dummy, 0)
    except Exception as exc:  # the first capture will report any real problem
        logger.warning("Classifier warm-up failed: %s", exc)


def new_session(seed: int = 7, preset: str = "small") -> dict:
    init_db()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    sim_id = f"sim_{stamp}"
    world = make_world(sim_id, int(seed), preset)
    s = Session(world=world)
    with _LOCK:
        _SESSIONS[sim_id] = s
    s.save()
    threading.Thread(target=_warm_up, daemon=True).start()
    logger.info("Simulation %s: %s field, seed %s, %d plants", sim_id, preset, seed,
                len(world["plants"]))
    from app.decision.rules_engine import load_rules
    from app.resource_optimization.grouping import load_mission_config

    rules = load_rules()
    cfg = load_mission_config()
    return {
        "sim_id": sim_id,
        "world": world,
        "mission_config": cfg.to_dict() | {"usable_capacity_ml": cfg.usable_capacity_ml},
        "rules": {"version": rules.version, "blanket_dose_ml": rules.blanket_dose_ml,
                  "uncertainty_threshold": rules.uncertainty_threshold,
                  "treatments": rules.treatments},
    }


# ── 2. Markers ───────────────────────────────────────────────────────────────

def drop_marker(sim_id: str, marker_id: int, x_m: float, y_m: float) -> dict:
    s = get_session(sim_id)
    if marker_id not in (0, 1, 2, 3):
        raise ValueError("marker_id must be 0-3")
    with s.lock:
        s.drops = [d for d in s.drops if d["marker_id"] != marker_id]
        s.drops.append({"marker_id": int(marker_id), "x_m": round(float(x_m), 4),
                        "y_m": round(float(y_m), 4)})
        s.drops.sort(key=lambda d: d["marker_id"])
        s.save()
        planned = s.world["planned_corners"][marker_id]
        return {"marker_id": marker_id, "placed": len(s.drops),
                "offset_from_planned_m": round(math.hypot(x_m - planned[0], y_m - planned[1]), 3)}


def _field_size_from_drops(drops: list[dict]) -> tuple[float, float]:
    d = {m["marker_id"]: (m["x_m"], m["y_m"]) for m in drops}
    width = (math.dist(d[0], d[1]) + math.dist(d[3], d[2])) / 2.0
    height = (math.dist(d[0], d[3]) + math.dist(d[1], d[2])) / 2.0
    return round(width, 3), round(height, 3)


# ── 3. Overview: boundary + plant positions ──────────────────────────────────

def _rows(points: list[tuple[float, float, Any]], tol: float = 0.35) -> list[list]:
    rows: list[list] = []
    for p in sorted(points, key=lambda q: (q[1], q[0])):
        if rows and abs(p[1] - np.mean([q[1] for q in rows[-1]])) <= tol:
            rows[-1].append(p)
        else:
            rows.append([p])
    return [sorted(r, key=lambda q: q[0]) for r in rows]


def find_row_gaps(points: list[tuple[float, float]], width: float, height: float) -> list[tuple[float, float]]:
    """
    Row check: plants are planted in rows at regular spacing, so a hole in a
    row (or a short row) is probably a plant the detector missed. Returns
    suspected positions for the low-altitude pass to look at.
    """
    rows = _rows([(x, y, None) for x, y in points])
    rows = [r for r in rows if len(r) >= 3]
    if not rows:
        return []
    gaps = [b[0] - a[0] for r in rows for a, b in zip(r[:-1], r[1:])]
    spacing = float(np.median(gaps)) if gaps else 1.0
    if spacing <= 0.2:
        return []
    starts = float(np.median([r[0][0] for r in rows]))
    ends = float(np.median([r[-1][0] for r in rows]))
    out = []
    for r in rows:
        y = float(np.mean([q[1] for q in r]))
        xs = [q[0] for q in r]
        for a, b in zip(xs[:-1], xs[1:]):
            n = int(round((b - a) / spacing)) - 1
            if n >= 1 and (b - a) > 1.5 * spacing:
                out += [(a + (b - a) * k / (n + 1), y) for k in range(1, n + 1)]
        x = xs[0] - spacing
        while x >= starts - 0.25 * spacing:
            out.append((x, y))
            x -= spacing
        x = xs[-1] + spacing
        while x <= ends + 0.25 * spacing:
            out.append((x, y))
            x += spacing
    return [(round(x, 3), round(y, 3)) for x, y in out
            if 0.2 < x < width - 0.2 and 0.2 < y < height - 0.2]


def _serpentine(targets: list[dict]) -> list[dict]:
    """Scan order: row by row from the south (nearest the home pad), alternating direction."""
    rows = _rows([(t["x_m"], t["y_m"], t) for t in targets], tol=0.4)
    order = []
    for i, r in enumerate(rows):
        seq = r if i % 2 == 0 else list(reversed(r))
        order += [q[2] for q in seq]
    return order


def capture_overview(sim_id: str) -> dict:
    from app.detection.detector import ClassicalPlantDetector
    from app.detection.identity_resolver import resolve_survey_detections
    from app.detection.router import _ensure_detections_table, _save_detections
    from app.field_survey.service import assess_image, register_survey, survey_exists
    from app.spatial_mapping.calibration import get_calibrator
    from app.spatial_mapping.database import ensure_spatial_tables, save_calibration
    from app.spatial_mapping.mapper import map_survey

    s = get_session(sim_id)
    with s.lock:
        if len(s.drops) < 4:
            raise ValueError("Place all four corner markers before mapping the field.")
        renderer = s.render()
        img = renderer.overview(s.drops)
        folder = IMAGES_ROOT / sim_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / OVERVIEW_NAME
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 93])

        # M1 — register the survey and check the photo is usable.
        width, height = _field_size_from_drops(s.drops)
        if not survey_exists(sim_id):
            register_survey(sim_id, field_width_m=width, field_height_m=height)
        quality = assess_image(path)

        # M5 — ArUco markers -> pixel-to-metre homography (field boundary).
        cal = get_calibrator().calibrate(sim_id, OVERVIEW_NAME, img, width, height)
        conn = get_connection()
        try:
            ensure_spatial_tables(conn)
            conn.execute("DELETE FROM spatial_calibrations WHERE survey_id = ?", (sim_id,))
            conn.commit()
            if cal.is_valid:
                save_calibration(conn, cal.calibration_id, sim_id, OVERVIEW_NAME,
                                 cal.field_width_m, cal.field_height_m, cal.aruco_dict,
                                 cal.marker_count, cal.homography_json,
                                 cal.reprojection_error, 1)
        finally:
            conn.close()
        markers_px = []
        try:
            for m in get_calibrator().detect_markers(img):
                md = m.as_dict() if hasattr(m, "as_dict") else dict(m)
                markers_px.append(md)
        except Exception:  # marker listing is only for display
            pass

        # M2 — plant detection on the overview photo.
        detector = ClassicalPlantDetector(kernel_frac=OVERVIEW_DETECTOR_KERNEL)
        raw = detector.detect_array(img)
        resolved = resolve_survey_detections([(OVERVIEW_NAME, raw)])
        conn = get_connection()
        try:
            _ensure_detections_table(conn)
            _save_detections(conn, sim_id, resolved["all_detections"],
                             model_version=detector.model_version)
            for table in ("classifications", "plant_severity", "plant_locations",
                          "treatment_decisions"):
                try:
                    conn.execute(f"DELETE FROM {table} WHERE survey_id = ?", (sim_id,))
                except Exception:
                    pass
            conn.commit()
        finally:
            conn.close()

        # M5 — every detection -> field metres.
        records = map_survey(sim_id) if resolved["all_detections"] else []
        by_id = {d["plant_id"]: d for d in resolved["all_detections"]}
        s.plants, s.targets, s.captures = {}, {}, {}
        for rec in records:
            pid = int(rec["plant_id"])
            if not rec["is_mapped"]:
                continue
            x, y = float(rec["field_x_m"]), float(rec["field_y_m"])
            gt = _nearest_true(s.world, x, y)
            s.plants[pid] = {"plant_id": pid, "x_m": round(x, 3), "y_m": round(y, 3),
                             "bbox": by_id[pid]["bbox"], "confidence": by_id[pid]["confidence"],
                             "source": "overview", "gid": gt["gid"] if gt else None}

        # Row check for missed plants.
        gaps = find_row_gaps([(p["x_m"], p["y_m"]) for p in s.plants.values()], width, height)
        targets = [{"target_id": f"p{pid}", "kind": "plant", "plant_id": pid,
                    "x_m": p["x_m"], "y_m": p["y_m"]} for pid, p in s.plants.items()]
        targets += [{"target_id": f"g{i + 1}", "kind": "gap", "plant_id": None,
                     "x_m": x, "y_m": y} for i, (x, y) in enumerate(gaps)]
        ordered = _serpentine(targets)
        s.targets = {t["target_id"]: t for t in ordered}

        # Ground-truth check of the mapping (synthetic field only).
        matched = {p["gid"] for p in s.plants.values() if p["gid"]}
        errs = []
        for p in s.plants.values():
            if p["gid"]:
                t = s.world["plants"][p["gid"] - 1]
                errs.append(math.hypot(t["x_m"] - p["x_m"], t["y_m"] - p["y_m"]))
        n_true = len(s.world["plants"])
        s.overview = {
            "image": OVERVIEW_NAME,
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "image_quality": {k: quality[k] for k in ("blur_score", "is_readable", "is_blurry")},
            "markers_found": int(cal.marker_count),
            "markers": markers_px,
            "calibration_valid": bool(cal.is_valid),
            "reprojection_error_px": (None if cal.reprojection_error is None
                                      else round(float(cal.reprojection_error), 3)),
            "field_width_m": width,
            "field_height_m": height,
            "boundary": [[d["x_m"], d["y_m"]] for d in s.drops],
            "detector": detector.model_version,
            "plants_detected": len(s.plants),
            "suspected_gaps": len(gaps),
            "eval": {
                "true_plants": n_true,
                "matched": len(matched),
                "false_detections": sum(1 for p in s.plants.values() if not p["gid"]),
                "recall": round(len(matched) / n_true, 4) if n_true else None,
                "mean_position_error_m": round(float(np.mean(errs)), 4) if errs else None,
            },
        }
        s.save()

        overlay = img.copy()
        for p in s.plants.values():
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 230, 255), 2)
        return {
            **s.overview,
            "image_url": f"/sim/{sim_id}/files/{OVERVIEW_NAME}",
            "image_annotated": _jpeg_data_url(cv2.resize(overlay, None, fx=0.5, fy=0.5), 80),
            "plants": list(s.plants.values()),
            "scan_targets": ordered,
        }


# ── 4. Per-plant photo -> model -> decision ──────────────────────────────────

class _Box:
    def __init__(self, bbox):
        self.bbox = [int(v) for v in bbox]


def _plant_mask(img: np.ndarray) -> np.ndarray:
    """
    Plant-versus-soil segmentation for the low-altitude photo. At 2.5 m the
    soil fills most of the frame and has a narrow colour range (brown: hue
    3-24, moderate saturation), so everything that is not soil is plant.
    Unlike the vegetation index used on the overview, this keeps brown,
    necrotic and pale leaf tissue, so the whole plant is cropped.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    soil = (h >= 3) & (h <= 24) & (s >= 60) & (v >= 40) & (v <= 190)
    mask = (~soil).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _locate_in_closeup(img: np.ndarray):
    """The plant nearest the photo centre (None if there is none)."""
    from app.detection.detector import ClassicalPlantDetector

    h, w = img.shape[:2]
    limit = CLOSEUP_SEARCH_M * TEXTURE_PPM
    n, _, stats, cents = cv2.connectedComponentsWithStats(_plant_mask(img))
    best, best_d = None, limit
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.05 * h * w:
            continue
        d = math.hypot(cents[i][0] - w / 2, cents[i][1] - h / 2)
        if d < best_d:
            best, best_d = _Box((x, y, x + bw, y + bh)), d
    if best is not None:
        return best
    # Fall back to the vegetation detector used on the overview photo.
    det = ClassicalPlantDetector(min_area_frac=0.03, max_area_frac=0.95)
    for d in det.detect_array(img):
        x1, y1, x2, y2 = d.bbox
        dd = math.hypot((x1 + x2) / 2 - w / 2, (y1 + y2) / 2 - h / 2)
        if dd < best_d:
            best, best_d = d, dd
    return best


def _insert_recovered_plant(sim_id: str, s: Session, x: float, y: float) -> int:
    """A plant found on the low pass where the overview missed one."""
    from app.spatial_mapping.calibration import homography_from_json
    from app.spatial_mapping.database import get_latest_calibration_for_survey, save_location

    pid = max(list(s.plants) + [0]) + 1
    sizes = [p["bbox"][2] - p["bbox"][0] for p in s.plants.values()] or [90]
    half = float(np.median(sizes)) / 2.0
    conn = get_connection()
    try:
        cal = get_latest_calibration_for_survey(conn, sim_id)
        cx = cy = 0.0
        cal_id = None
        if cal:
            H = homography_from_json(cal["homography_json"])
            v = np.linalg.inv(H) @ np.array([x, y, 1.0])
            cx, cy = float(v[0] / v[2]), float(v[1] / v[2])
            cal_id = cal["calibration_id"]
        bbox = [round(cx - half), round(cy - half), round(cx + half), round(cy + half)]
        conn.execute(
            "INSERT INTO detections (survey_id, image_filename, plant_id, bbox_x1, bbox_y1, "
            "bbox_x2, bbox_y2, confidence, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sim_id, OVERVIEW_NAME, pid, *bbox, 0.5, "low-pass-recovered-v1"))
        conn.commit()
        save_location(conn, sim_id, pid, OVERVIEW_NAME, cx, cy, x, y, 1, cal_id,
                      "calibrated", "low-pass-closeup-v1")
    finally:
        conn.close()
    gt = _nearest_true(s.world, x, y)
    s.plants[pid] = {"plant_id": pid, "x_m": round(x, 3), "y_m": round(y, 3), "bbox": bbox,
                     "confidence": 0.5, "source": "low_pass", "gid": gt["gid"] if gt else None}
    return pid


def capture_plant(sim_id: str, target_id: str, drone_x: float, drone_y: float,
                  gradcam: bool = True) -> dict:
    from app.classification.classifier import get_classifier
    from app.classification.database import ensure_classifications_table, save_classification
    from app.classification.prediction import ClassificationResult
    from app.decision.rules_engine import decide_plant, load_rules
    from app.severity.database import ensure_severity_table, save_severity
    from app.severity.estimator import get_estimator

    s = get_session(sim_id)
    with s.lock:
        if target_id not in s.targets:
            raise KeyError(f"unknown scan target '{target_id}'")
        t = s.targets[target_id]
        renderer = s.render()
        photo = renderer.closeup(s.drops, drone_x, drone_y,
                                 seed=int(re.sub(r"\D", "", target_id) or 0))
        h, w = photo.shape[:2]
        out = s.out_dir() / "closeups"
        out.mkdir(exist_ok=True)
        det = _locate_in_closeup(photo)
        result: dict[str, Any] = {"target_id": target_id, "kind": t["kind"]}

        if det is None:
            annotated = photo.copy()
            cv2.putText(annotated, "no plant", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (40, 40, 255), 2)
            result.update(found=False, photo=_jpeg_data_url(annotated),
                          plant_id=t.get("plant_id"))
            if t["kind"] == "gap":
                result["message"] = "No plant at this gap — confirmed empty."
            else:
                result["message"] = "Plant not visible in the close-up — marked for inspection."
                result["decision"] = {"action": "inspect", "dosage_ml": 0, "treatment": None,
                                      "reason": "Plant not found in the low-altitude photo."}
            t["captured"] = True
            s.save()
            return result

        x1, y1, x2, y2 = det.bbox
        # Plant position from the photo: offset from the image centre.
        px = drone_x + ((x1 + x2) / 2 - w / 2) / TEXTURE_PPM
        py = drone_y - ((y1 + y2) / 2 - h / 2) / TEXTURE_PPM
        if t["kind"] == "gap":
            pid = _insert_recovered_plant(sim_id, s, px, py)
            t.update(kind="plant", plant_id=pid, recovered=True)
        pid = int(t["plant_id"])
        plant = s.plants[pid]

        crop = photo[y1:y2, x1:x2]
        pred = get_classifier().predict(crop)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        sev = get_estimator().estimate(crop_rgb, pred["label"], pid, sim_id)

        conn = get_connection()
        try:
            ensure_classifications_table(conn)
            ensure_severity_table(conn)
            save_classification(conn, ClassificationResult(
                survey_id=sim_id, image_filename=OVERVIEW_NAME, plant_id=pid,
                bbox=[float(v) for v in plant["bbox"]], label=pred["label"],
                confidence=pred["confidence"], all_probs=pred["all_probs"],
                uncertain=bool(pred.get("uncertain"))))
            save_severity(conn=conn, survey_id=sim_id, plant_id=pid,
                          severity_score=sev.severity_score,
                          severity_category=sev.severity_category, leaf_px=sev.leaf_px,
                          lesion_px=sev.lesion_px, is_valid=sev.is_valid,
                          model_version=sev.model_version)
            conn.commit()
        finally:
            conn.close()

        rules = load_rules()
        decision = decide_plant({
            "plant_id": pid, "image_filename": OVERVIEW_NAME,
            "disease_label": pred["label"], "confidence": pred["confidence"],
            "is_uncertain": pred.get("uncertain"), "severity_score": sev.severity_score,
            "severity_category": sev.severity_category, "severity_valid": sev.is_valid,
            "field_x_m": plant["x_m"], "field_y_m": plant["y_m"], "is_mapped": 1,
        }, rules, recent_sprays=[])

        annotated = photo.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 230, 255), 2)
        cv2.imwrite(str(out / f"plant_{pid:03d}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        heat_url = None
        if gradcam:
            try:
                from app.classification.explain import grad_cam, overlay_heatmap

                clf = get_classifier()
                idx = clf.class_order.index(pred["label"])
                heat, _, _ = grad_cam(clf.model, crop_rgb, idx)
                over = cv2.cvtColor(overlay_heatmap(crop_rgb, heat), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / f"gradcam_{pid:03d}.jpg"), over)
                heat_url = _jpeg_data_url(over)
            except Exception as exc:  # explanation is optional
                logger.warning("Grad-CAM failed for plant %s: %s", pid, exc)

        gt = s.world["plants"][plant["gid"] - 1] if plant.get("gid") else None
        summary = {
            "plant_id": pid, "x_m": plant["x_m"], "y_m": plant["y_m"], "gid": plant.get("gid"),
            "label": pred["label"], "confidence": pred["confidence"],
            "uncertain": bool(pred.get("uncertain")),
            "severity_score": sev.severity_score, "severity_category": sev.severity_category,
            "severity_valid": bool(sev.is_valid),
            "action": decision["action"], "dosage_ml": decision["dosage_ml"],
            "treatment": decision["treatment"], "reason": decision["reason"],
            "true_label": gt["true_label"] if gt else None,
        }
        s.captures[pid] = summary
        t["captured"] = True
        s.save()
        result.update(
            found=True, plant_id=pid, recovered=bool(t.get("recovered")),
            position=[round(px, 3), round(py, 3)],
            photo=_jpeg_data_url(annotated), gradcam=heat_url,
            probabilities=pred["all_probs"],
            model_version=get_classifier().model_version,
            **{k: summary[k] for k in ("label", "confidence", "uncertain", "severity_score",
                                       "severity_category", "severity_valid")},
            decision={k: decision[k] for k in ("action", "dosage_ml", "treatment", "reason")},
        )
        return result


# ── 5. Plan the spray mission ────────────────────────────────────────────────

def plan(sim_id: str) -> dict:
    from app.decision.service import generate_decisions
    from app.drone_integration.command_queue import generate_commands
    from app.mission_planner.service import plan_mission
    from app.spatial_mapping.mapper import map_survey

    s = get_session(sim_id)
    with s.lock:
        if not s.captures:
            raise ValueError("No plant has been photographed yet — run the scan first.")
        map_survey(sim_id)                      # include plants recovered on the low pass
        decisions = generate_decisions(sim_id, ignore_history=True)
        home = s.world["home"]
        mission = plan_mission(sim_id, depot_x_m=home["x_m"], depot_y_m=home["y_m"])
        files = []
        if mission["status"].startswith("ready"):
            files = generate_commands(sim_id)["files"]
        s.plan = {"status": mission["status"], "metrics": mission["metrics"],
                  "summary": decisions["summary"],
                  "sorties": [sortie for p in mission["programmes"] for sortie in p["sorties"]],
                  "mix_plan": mission["mix_plan"], "files": files}
        s.save()
        return {
            "status": mission["status"],
            "message": mission.get("message"),
            "rules_version": decisions["rules_version"],
            "summary": decisions["summary"],
            "decisions": [{k: d[k] for k in ("plant_id", "disease_label", "confidence",
                                             "severity_category", "action", "dosage_ml",
                                             "treatment", "reason", "field_x_m", "field_y_m")}
                          for d in decisions["decisions"]],
            "mix_plan": mission["mix_plan"],
            "programmes": mission["programmes"],
            "sorties": s.plan["sorties"],
            "metrics": mission["metrics"],
            "waypoint_files": [Path(f).name for f in files if str(f).endswith(".waypoints")],
        }


# ── 6. Re-plan from the current position ─────────────────────────────────────

def replan(sim_id: str, position: tuple[float, float], tank_ml: float,
           tank_treatment: Optional[str], remaining: list[int]) -> dict:
    from app.decision.database import ensure_decision_tables, get_decisions
    from app.mission_planner.route_optimizer import plan_capacitated_route, plan_from_position
    from app.resource_optimization.grouping import load_mission_config

    s = get_session(sim_id)
    cfg = load_mission_config()
    home = (s.world["home"]["x_m"], s.world["home"]["y_m"])
    conn = get_connection()
    try:
        ensure_decision_tables(conn)
        decisions = {d["plant_id"]: d for d in get_decisions(conn, sim_id)}
    finally:
        conn.close()
    todo = [decisions[p] for p in remaining
            if p in decisions and decisions[p]["action"] == "spray" and decisions[p]["is_mapped"]]

    def stop(d):
        return {"plant_id": d["plant_id"], "x_m": round(d["field_x_m"], 3),
                "y_m": round(d["field_y_m"], 3), "dosage_ml": d["dosage_ml"],
                "spray_s": round(d["dosage_ml"] / cfg.spray_flow_ml_per_s, 2),
                "disease": d["disease_label"], "severity": d["severity_category"]}

    sorties = []
    start = (float(position[0]), float(position[1]))
    same = [d for d in todo if tank_treatment and d["treatment"] == tank_treatment]
    others = [d for d in todo if d not in same]
    if same and tank_ml > 0:
        pts = [(d["field_x_m"], d["field_y_m"]) for d in same]
        r = plan_from_position(pts, [d["dosage_ml"] for d in same], start, tank_ml, home,
                               cfg.usable_capacity_ml)
        first = [same[i] for i in r["first"]]
        if first:
            sorties.append({"treatment": tank_treatment, "from_current_position": True,
                            "stops": [stop(d) for d in first],
                            "load_ml": round(sum(d["dosage_ml"] for d in first), 1)})
        others = [same[i] for chunk in r["later"] for i in chunk] + others
    groups: dict[str, list] = {}
    for d in others:
        groups.setdefault(d["treatment"], []).append(d)
    for treatment, items in groups.items():
        pts = [(d["field_x_m"], d["field_y_m"]) for d in items]
        dem = [d["dosage_ml"] for d in items]
        for chunk in plan_capacitated_route(pts, dem, cfg.usable_capacity_ml, home):
            sorties.append({"treatment": treatment, "from_current_position": False,
                            "stops": [stop(items[i]) for i in chunk],
                            "load_ml": round(sum(dem[i] for i in chunk), 1)})

    def leg(points):
        return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))

    total = 0.0
    for so in sorties:
        pts = [(st["x_m"], st["y_m"]) for st in so["stops"]]
        begin = start if so["from_current_position"] else home
        total += leg([begin] + pts + [home])
    if not sorties or not sorties[0]["from_current_position"]:
        total += math.dist(start, home)
    for i, so in enumerate(sorties, 1):
        so["sortie"] = i
    entry = {"at": datetime.now().isoformat(timespec="seconds"), "position": list(start),
             "tank_ml": tank_ml, "tank_treatment": tank_treatment,
             "remaining_targets": len(todo), "sorties": len(sorties),
             "distance_m": round(total, 2)}
    with s.lock:
        s.replans.append(entry)
        s.save()
    return {"sorties": sorties, "distance_m": round(total, 2), "remaining_targets": len(todo)}


# ── 7. Finish: flight log + summary ──────────────────────────────────────────

def finish(sim_id: str, telemetry: dict) -> dict:
    from app.decision.rules_engine import load_rules
    from app.mission_planner.route_optimizer import lawnmower_length
    from app.resource_optimization.grouping import load_mission_config

    s = get_session(sim_id)
    rules = load_rules()
    cfg = load_mission_config()
    with s.lock:
        sprayed = {int(k): float(v) for k, v in (telemetry.get("sprayed_ml") or {}).items()}
        world_plants = s.world["plants"]
        n_true = len(world_plants)
        diseased = [p for p in world_plants if p["true_label"] != "healthy"]
        gid_to_pid = {p["gid"]: pid for pid, p in s.plants.items() if p.get("gid")}

        cls_total = cls_ok = 0
        confusion: dict[str, dict[str, int]] = {}
        for c in s.captures.values():
            if c.get("true_label"):
                cls_total += 1
                cls_ok += int(c["label"] == c["true_label"])
                row = confusion.setdefault(c["true_label"], {})
                row[c["label"]] = row.get(c["label"], 0) + 1

        def acted(p):
            pid = gid_to_pid.get(p["gid"])
            c = s.captures.get(pid) if pid else None
            return bool(c and (c["action"] in ("spray", "inspect", "monitor")
                               or sprayed.get(pid, 0) > 0))

        healthy_sprayed = sum(1 for p in world_plants if p["true_label"] == "healthy"
                              and sprayed.get(gid_to_pid.get(p["gid"], -1), 0) > 0)
        planned_targets = {pid for pid, c in s.captures.items() if c["action"] == "spray"}
        fully = sum(1 for pid in planned_targets
                    if sprayed.get(pid, 0) >= 0.95 * s.captures[pid]["dosage_ml"])
        used = round(sum(sprayed.values()), 1)
        blanket = n_true * rules.blanket_dose_ml
        width = (s.overview or {}).get("field_width_m", s.world["field_width_m"])
        height = (s.overview or {}).get("field_height_m", s.world["field_height_m"])
        home = (s.world["home"]["x_m"], s.world["home"]["y_m"])
        lawn = lawnmower_length(width, height, cfg.spray_swath_m, home)
        spray_dist = float(telemetry.get("spray_route_m")
                           or (telemetry.get("phases") or {}).get("spray", {}).get("distance_m", 0))
        cost_per_ml = {}
        for name, tinfo in rules.treatments.items():
            cost_per_ml[name] = float(tinfo.get("cost_per_litre_mix_inr", tinfo.get("cost_per_litre_inr", 0))) / 1000.0
        cost_used = 0.0
        for pid, ml in sprayed.items():
            tr = (s.captures.get(pid) or {}).get("treatment") or "protectant_fungicide"
            cost_used += ml * cost_per_ml.get(tr, 0.06)
        # Same baseline as Module 6: blanket spraying uses the most expensive
        # programme on every plant (the farmer cannot see which disease is where).
        blanket_cost = blanket * max(list(cost_per_ml.values()) or [0.06])

        summary = {
            "sim_id": sim_id,
            "field": {"preset": s.world["preset"], "seed": s.world["seed"],
                      "plants": n_true, "width_m": width, "height_m": height},
            "mapping": (s.overview or {}).get("eval"),
            "plants_mapped": len(s.plants),
            "recovered_on_low_pass": sum(1 for p in s.plants.values() if p["source"] == "low_pass"),
            "plants_photographed": len(s.captures),
            "classification": {"correct": cls_ok, "total": cls_total,
                               "accuracy": round(cls_ok / cls_total, 4) if cls_total else None,
                               "confusion": confusion},
            "actions": _count(c["action"] for c in s.captures.values()),
            "spray_targets": len(planned_targets),
            "targets_fully_sprayed": fully,
            "diseased_plants": len(diseased),
            "diseased_acted_on": sum(1 for p in diseased if acted(p)),
            "healthy_plants_sprayed": healthy_sprayed,
            "chemical": {"used_ml": used, "blanket_ml": blanket,
                         "saved_pct": round(100 * (1 - used / blanket), 1) if blanket else None,
                         "cost_inr": round(cost_used, 2), "blanket_cost_inr": round(blanket_cost, 2)},
            "route": {"spray_phase_distance_m": round(spray_dist, 2),
                      "planned_distance_m": (s.plan or {}).get("metrics", {}).get("optimised_distance_m"),
                      "lawnmower_distance_m": round(lawn, 2),
                      "saved_vs_lawnmower_pct": round(100 * (1 - spray_dist / lawn), 1)
                      if lawn and spray_dist else None},
            "flight": {k: telemetry.get(k) for k in ("flight_time_s", "distance_m",
                                                    "battery_used_pct", "energy_wh")},
            "phases": telemetry.get("phases"),
            "manual": telemetry.get("manual"),
            "replans": len(s.replans),
            "note": s.world["note"],
        }
        s.summary = summary
        out = s.out_dir()
        (out / "flight_log.json").write_text(json.dumps({
            "sim_id": sim_id, "events": telemetry.get("events", []),
            "track": telemetry.get("track", []), "sprayed_ml": sprayed,
            "replans": s.replans}, indent=1), encoding="utf-8")
        (out / "sim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        s.save()
        return summary


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def state(sim_id: str) -> dict:
    s = get_session(sim_id)
    return {"sim_id": sim_id, "markers_placed": len(s.drops), "overview": s.overview,
            "plants_mapped": len(s.plants), "plants_photographed": len(s.captures),
            "plan": s.plan, "replans": s.replans, "summary": s.summary}


def file_path(sim_id: str, name: str) -> Path:
    """Safe path to a file the simulator wrote (overview, close-ups, logs)."""
    validate_sim_id(sim_id)
    if name == OVERVIEW_NAME:
        path = (IMAGES_ROOT / sim_id / name).resolve()
        root = (IMAGES_ROOT / sim_id).resolve()
    else:
        root = (OUTPUTS_ROOT / sim_id).resolve()
        path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(name)
    return path


# ── Housekeeping ─────────────────────────────────────────────────────────────

SIM_TABLES = ("survey_images", "detections", "classifications", "plant_severity",
              "plant_locations", "spatial_calibrations", "treatment_decisions",
              "mission_plans", "drone_commands", "execution_logs", "surveys")


def delete_simulation(sim_id: str) -> None:
    """Remove one simulated mission: database rows, images and outputs."""
    import shutil

    validate_sim_id(sim_id)
    conn = get_connection()
    try:
        for table in SIM_TABLES:
            try:
                conn.execute(f"DELETE FROM {table} WHERE survey_id = ?", (sim_id,))
            except Exception:       # table may not exist yet
                pass
        conn.commit()
    finally:
        conn.close()
    for root in (IMAGES_ROOT, OUTPUTS_ROOT):
        shutil.rmtree(root / sim_id, ignore_errors=True)
    with _LOCK:
        _SESSIONS.pop(sim_id, None)


def list_simulations() -> list[str]:
    ids = set()
    for root in (IMAGES_ROOT, OUTPUTS_ROOT):
        if root.exists():
            ids |= {p.name for p in root.iterdir() if p.is_dir() and SIM_ID_RE.match(p.name)}
    conn = get_connection()
    try:
        rows = conn.execute("SELECT survey_id FROM surveys WHERE survey_id LIKE 'sim\\_%' ESCAPE '\\'").fetchall()
        ids |= {r[0] for r in rows if SIM_ID_RE.match(r[0])}
    except Exception:
        pass
    finally:
        conn.close()
    return sorted(ids)
