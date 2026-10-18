"""
FILE PATH: SmartSpray/app/pipeline.py

PURPOSE:
    The SmartSpray processing pipeline. run_pipeline(survey_id) runs every
    module in order, calling each module's service layer directly (no HTTP).
    Every stage writes to the database and a failing stage never aborts the
    stages after it — it is reported as "skipped" (missing input) or "error".

PIPELINE STAGES:
    Stage 1 — Field Survey (M1)          survey lookup
    Stage 2 — Plant Detection (M2)       YOLOv8 (or opt-in OpenCV fallback)
    Stage 3 — Disease Classification (M3) EfficientNetB0 (+ optional Grad-CAM)
    Stage 4 — Severity Estimation (M4)   OpenCV lesion-area ratio
    Stage 5 — Spatial Mapping (M5)       ArUco homography (auto-calibrates
                                         from survey images when possible)
    Stage 6 — Treatment Decision (M6)    auditable rule table
    Stage 7 — Resource Optimisation (M7) tank mixes, loads, cost
    Stage 8 — Mission Planning (M8)      NN + 2-opt route, sorties
    Stage 9 — Drone Commands (M9)        JSON queue + QGC waypoint files

USAGE:
    python -m app.pipeline <survey_id> [--detector auto] [--gradcam] [--simulate]
"""

import argparse
import json
import logging
import sqlite3  # noqa: F401  (kept for backwards compatibility of imports)
from pathlib import Path

from app.db import get_connection
from app.detection.detector import get_detector
from app.detection.identity_resolver import resolve_survey_detections
from app.detection.router import (
    VALID_IMAGE_SUFFIXES,
    _ensure_detections_table,
    _save_detections,
)
import app.spatial_mapping.mapper as spatial_mapper

logger = logging.getLogger("pipeline")

# Where survey images live (same as Module 1 and router.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = REPO_ROOT / "data" / "images"


def _skipped(reason) -> dict:
    return {"status": "skipped", "reason": str(reason)}


def _error(exc) -> dict:
    return {"status": "error", "error": str(exc)}


def _auto_calibrate(survey_id: str, survey_folder, field_size: str) -> dict:
    """
    If the survey has no valid calibration yet, look for the four corner ArUco
    markers in its images and store the first valid calibration found.
    """
    from app.field_survey.service import parse_field_size
    from app.spatial_mapping.calibration import get_calibrator
    from app.spatial_mapping.database import (
        ensure_spatial_tables,
        get_latest_calibration_for_survey,
        save_calibration,
    )
    from app.spatial_mapping.database import get_connection as spatial_conn

    conn = spatial_conn()
    try:
        ensure_spatial_tables(conn)
        existing = get_latest_calibration_for_survey(conn, survey_id)
    finally:
        conn.close()
    if existing:
        return {"calibration": "existing", "calibration_id": existing["calibration_id"]}
    if not survey_folder.exists():
        return {"calibration": "none", "reason": "survey folder not found"}

    width, height = parse_field_size(field_size or "")
    calibrator = get_calibrator()
    for img in sorted(f for f in survey_folder.iterdir()
                      if f.suffix.lower() in VALID_IMAGE_SUFFIXES):
        try:
            result = calibrator.calibrate_from_file(survey_id, img.name, img, width, height)
        except (ValueError, FileNotFoundError, RuntimeError):
            continue
        if not result.is_valid:
            continue
        conn = spatial_conn()
        try:
            save_calibration(conn, result.calibration_id, survey_id, img.name,
                             result.field_width_m, result.field_height_m,
                             result.aruco_dict, result.marker_count,
                             result.homography_json, result.reprojection_error, 1)
        finally:
            conn.close()
        return {"calibration": "auto", "calibration_id": result.calibration_id,
                "image": img.name, "reprojection_error_px": result.reprojection_error}
    return {"calibration": "none",
            "reason": "no image contains the 4 corner ArUco markers (ids 0-3)"}


def run_pipeline(survey_id: str, detector_mode: str | None = None,
                 save_gradcam: bool = False, simulate: bool = False,
                 write_report: bool = False) -> dict:
    """
    Run all pipeline stages for a given survey_id.

    Args:
        survey_id:     must exist in the surveys table (Module 1).
        detector_mode: "yolo" | "auto" | "classical" | None (env default).
        save_gradcam:  write Grad-CAM overlays for every plant (Module 3).
        simulate:      run the software drone executor after Stage 9.
        write_report:  write data/outputs/<survey_id>/report.html.

    Raises:
        ValueError: if the survey_id doesn't exist in the database.
    """
    pipeline_result = {"survey_id": survey_id}

    # ═══ STAGE 1 — FIELD SURVEY (Module 1) ═══════════════════════════════════
    logger.info(f"[Pipeline] Stage 1: Verifying survey '{survey_id}'")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT survey_id, crop, field_size, num_images, created_at FROM surveys WHERE survey_id = ?",
            (survey_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(
            f"Survey '{survey_id}' not found in the database. "
            f"Register it first using POST /survey/register (Module 1)."
        )

    pipeline_result["stage_1_survey"] = {
        "status": "found",
        "survey_id": row[0],
        "crop": row[1],
        "field_size": row[2],
        "num_images": row[3],
        "created_at": row[4]
    }
    logger.info(f"[Pipeline] Stage 1 complete: survey found ({row[3]} images registered)")

    # ═══ STAGE 2 — PLANT DETECTION (Module 2) ════════════════════════════════
    logger.info(f"[Pipeline] Stage 2: Running plant detection for '{survey_id}'")
    survey_folder = IMAGES_ROOT / survey_id
    image_files = sorted([
        f for f in survey_folder.iterdir()
        if f.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]) if survey_folder.exists() else []

    if not image_files:
        pipeline_result["stage_2_detection"] = _skipped(f"No images found in {survey_folder}")
    else:
        try:
            detector = get_detector(detector_mode)
            per_image_results = []
            for img_path in image_files:
                try:
                    raw = detector.detect(img_path)
                    per_image_results.append((img_path.name, raw))
                except Exception as e:
                    logger.warning(f"[Pipeline] Skipped {img_path.name}: {e}")
                    per_image_results.append((img_path.name, []))

            resolved = resolve_survey_detections(per_image_results)
            conn = get_connection()
            try:
                _ensure_detections_table(conn)
                _save_detections(conn, survey_id, resolved["all_detections"],
                                 model_version=detector.model_version)
            finally:
                conn.close()

            pipeline_result["stage_2_detection"] = {
                "status": "success",
                "model_version": detector.model_version,
                "images_processed": len(image_files),
                "plants_detected": resolved["total_plants"],
                "all_detections": resolved["all_detections"],
            }
            logger.info(
                f"[Pipeline] Stage 2 complete: {resolved['total_plants']} plants detected "
                f"across {len(image_files)} images ({detector.model_version})"
            )
        except FileNotFoundError as e:
            pipeline_result["stage_2_detection"] = _skipped(e)
        except Exception as e:
            logger.error(f"[Pipeline] Stage 2 failed: {e}", exc_info=True)
            pipeline_result["stage_2_detection"] = _error(e)

    # ═══ STAGE 3 — DISEASE CLASSIFICATION (Module 3) ═════════════════════════
    logger.info(f"[Pipeline] Stage 3: Classifying plants for '{survey_id}'")
    try:
        from app.classification.service import classify_survey

        kwargs = {"save_gradcam": True} if save_gradcam else {}
        pipeline_result["stage_3_classification"] = classify_survey(survey_id, **kwargs)
        logger.info("[Pipeline] Stage 3 complete: %s classified, %s uncertain",
                    pipeline_result["stage_3_classification"].get("classified"),
                    pipeline_result["stage_3_classification"].get("uncertain"))
    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"[Pipeline] Stage 3 skipped: {e}")
        pipeline_result["stage_3_classification"] = _skipped(e)
    except Exception as e:
        logger.error(f"[Pipeline] Stage 3 failed: {e}", exc_info=True)
        pipeline_result["stage_3_classification"] = _error(e)

    # ═══ STAGE 4 — DISEASE SEVERITY ESTIMATION (Module 4) ════════════════════
    logger.info(f"[Pipeline] Stage 4: Estimating severity for survey '{survey_id}'")
    try:
        from app.severity.service import estimate_survey
        from app.severity import database
        from app.severity.database import save_severity

        severity_results = estimate_survey(survey_id)

        conn = database.get_connection()
        try:
            for result in severity_results:
                save_severity(
                    conn=conn,
                    survey_id=result.survey_id,
                    plant_id=result.plant_id,
                    severity_score=result.severity_score,
                    severity_category=result.severity_category,
                    leaf_px=result.leaf_px,
                    lesion_px=result.lesion_px,
                    is_valid=result.is_valid,
                    model_version=result.model_version,
                )
            conn.commit()
        finally:
            conn.close()

        valid_count = sum(1 for r in severity_results if r.is_valid)
        invalid_count = sum(1 for r in severity_results if not r.is_valid)

        pipeline_result["stage_4_severity"] = {
            "status": "success",
            "total_plants": len(severity_results),
            "valid_estimates": valid_count,
            "invalid_estimates": invalid_count,
            "results": [
                {
                    "plant_id": r.plant_id,
                    "disease_label": r.disease_label,
                    "severity_score": r.severity_score,
                    "severity_category": r.severity_category,
                    "is_valid": r.is_valid,
                }
                for r in severity_results
            ],
        }
        logger.info(
            f"[Pipeline] Stage 4 complete: {len(severity_results)} plants, "
            f"{valid_count} valid, {invalid_count} invalid."
        )
    except ValueError as e:
        logger.warning(f"[Pipeline] Stage 4 skipped: {e}")
        pipeline_result["stage_4_severity"] = _skipped(e)
    except Exception as e:
        logger.error(f"[Pipeline] Stage 4 failed: {e}", exc_info=True)
        pipeline_result["stage_4_severity"] = _error(e)

    # ═══ STAGE 5 — SPATIAL MAPPING (Module 5) ════════════════════════════════
    # Pixel bbox centres -> field metres via a stored ArUco calibration. If the
    # survey has none, the pipeline first tries to calibrate from its own
    # images. Without a valid calibration plants stay 'uncalibrated' (NULL
    # coordinates) — coordinates are never invented.
    try:
        try:
            calibration_info = _auto_calibrate(survey_id, survey_folder, row[2])
        except Exception as e:  # calibration problems must not stop mapping
            calibration_info = {"calibration": "none", "reason": str(e)}
        location_records = spatial_mapper.map_survey(survey_id)
        mapping_summary = spatial_mapper.summarise(location_records)
        mapping_mode = location_records[0]["mapping_mode"]

        pipeline_result["stage_5_spatial_mapping"] = {
            "status": "success",
            "mapping_mode": mapping_mode,
            "calibration_id": location_records[0]["calibration_id"],
            "calibration_source": calibration_info,
            "total_plants": mapping_summary["total_plants"],
            "mapped_plants": mapping_summary["mapped_plants"],
            "unmapped_plants": mapping_summary["unmapped_plants"],
            "results": [
                {
                    "plant_id": record["plant_id"],
                    "field_x_m": record["field_x_m"],
                    "field_y_m": record["field_y_m"],
                    "is_mapped": record["is_mapped"],
                }
                for record in location_records
            ],
        }
        logger.info(
            f"[Pipeline] Stage 5 complete: {mapping_summary['total_plants']} plants, "
            f"{mapping_summary['mapped_plants']} mapped, "
            f"{mapping_summary['unmapped_plants']} unmapped, mode={mapping_mode}."
        )
    except ValueError as e:
        logger.warning(f"[Pipeline] Stage 5 skipped: {e}")
        pipeline_result["stage_5_spatial_mapping"] = _skipped(e)
    except Exception as e:
        logger.error(f"[Pipeline] Stage 5 failed: {e}", exc_info=True)
        pipeline_result["stage_5_spatial_mapping"] = _error(e)

    # ═══ STAGE 6 — TREATMENT DECISION (Module 6) ═════════════════════════════
    try:
        from app.decision.service import generate_decisions

        decisions = generate_decisions(survey_id)
        pipeline_result["stage_6_treatment_decision"] = {
            "status": "success",
            "rules_version": decisions["rules_version"],
            "summary": decisions["summary"],
            "decisions": [
                {k: d[k] for k in ("plant_id", "disease_label", "confidence",
                                   "severity_category", "location", "action",
                                   "dosage_ml", "treatment", "reason")}
                for d in decisions["decisions"]
            ],
        }
    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"[Pipeline] Stage 6 skipped: {e}")
        pipeline_result["stage_6_treatment_decision"] = _skipped(e)
    except Exception as e:
        logger.error(f"[Pipeline] Stage 6 failed: {e}", exc_info=True)
        pipeline_result["stage_6_treatment_decision"] = _error(e)

    # ═══ STAGES 7 + 8 — RESOURCE OPTIMISATION + MISSION PLANNING ═════════════
    plan = None
    if pipeline_result["stage_6_treatment_decision"]["status"] == "success":
        try:
            from app.mission_planner.service import plan_mission

            plan = plan_mission(survey_id)
            pipeline_result["stage_7_resource_optimization"] = {
                "status": "success",
                "mix_plan": plan["mix_plan"],
                "tank_loads": plan["metrics"]["sorties"],
                "chemical": pipeline_result["stage_6_treatment_decision"]["summary"],
            }
            pipeline_result["stage_8_mission_planning"] = {
                "status": "success" if plan["status"].startswith("ready")
                or plan["status"] == "no_targets" else "skipped",
                "plan_status": plan["status"],
                "message": plan.get("message"),
                "metrics": plan["metrics"],
                "sorties": [
                    {k: s[k] for k in ("sortie", "treatment", "plant_ids", "load_ml",
                                       "distance_m", "flight_time_s")}
                    for p in plan["programmes"] for s in p["sorties"]
                ],
            }
        except (ValueError, FileNotFoundError) as e:
            pipeline_result["stage_7_resource_optimization"] = _skipped(e)
            pipeline_result["stage_8_mission_planning"] = _skipped(e)
        except Exception as e:
            logger.error(f"[Pipeline] Stage 7/8 failed: {e}", exc_info=True)
            pipeline_result["stage_7_resource_optimization"] = _error(e)
            pipeline_result["stage_8_mission_planning"] = _error(e)
    else:
        reason = "Module 6 produced no decisions"
        pipeline_result["stage_7_resource_optimization"] = _skipped(reason)
        pipeline_result["stage_8_mission_planning"] = _skipped(reason)

    # ═══ STAGE 9 — DRONE COMMAND GENERATION (Module 9) ═══════════════════════
    if pipeline_result["stage_8_mission_planning"]["status"] == "success":
        try:
            from app.drone_integration.command_queue import generate_commands

            cmds = generate_commands(survey_id)
            pipeline_result["stage_9_drone_commands"] = {
                "status": "success",
                "total_commands": cmds["total_commands"],
                "spray_commands": cmds["spray_commands"],
                "files": cmds["files"],
                "sample": [c for c in cmds["commands"] if c["type"] == "spray"][:3],
            }
            if simulate:
                from simulator.executor_simulator import run as run_simulation

                pipeline_result["simulation"] = run_simulation(survey_id)
        except (ValueError, FileNotFoundError) as e:
            pipeline_result["stage_9_drone_commands"] = _skipped(e)
        except Exception as e:
            logger.error(f"[Pipeline] Stage 9 failed: {e}", exc_info=True)
            pipeline_result["stage_9_drone_commands"] = _error(e)
    else:
        pipeline_result["stage_9_drone_commands"] = _skipped("no flyable mission plan")

    if write_report:
        try:
            from app.reporting import write_survey_report

            pipeline_result["report"] = write_survey_report(survey_id, pipeline_result)
        except Exception as e:
            logger.error(f"[Pipeline] Report failed: {e}", exc_info=True)
            pipeline_result["report"] = _error(e)

    logger.info(f"[Pipeline] All stages complete for survey '{survey_id}'")
    return pipeline_result


def stage_statuses(result: dict) -> dict:
    return {k: v.get("status") for k, v in result.items()
            if isinstance(v, dict) and k.startswith("stage_")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SmartSpray pipeline for a survey")
    ap.add_argument("survey_id")
    ap.add_argument("--detector", choices=["yolo", "auto", "classical"], default=None)
    ap.add_argument("--gradcam", action="store_true", help="save Grad-CAM overlays")
    ap.add_argument("--simulate", action="store_true", help="run the drone simulator")
    ap.add_argument("--report", action="store_true", help="write report.html")
    ap.add_argument("--json", action="store_true", help="print the full JSON result")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from app.db import init_db

    init_db()
    result = run_pipeline(args.survey_id, args.detector, args.gradcam,
                          args.simulate, args.report)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(json.dumps(stage_statuses(result), indent=2))
        s6 = result.get("stage_6_treatment_decision", {})
        if s6.get("status") == "success":
            print(json.dumps(s6["summary"], indent=2))
        s8 = result.get("stage_8_mission_planning", {})
        if s8.get("status") == "success":
            print(json.dumps(s8["metrics"], indent=2))
        if "report" in result:
            print("report:", result["report"])


if __name__ == "__main__":
    main()
