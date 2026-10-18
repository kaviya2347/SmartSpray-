"""
FILE PATH: SmartSpray/scripts/run_demo.py

One-command end-to-end demo of SmartSpray (Modules 1 -> 9 + simulator).

    python scripts/run_demo.py                 # synthetic demo field
    python scripts/run_demo.py --fresh         # rebuild the field + DB rows first

Steps:
    1. build data/images/demo_survey_001/ (synthetic orthomosaic + ground truth)
    2. register the survey (Module 1)
    3. run the pipeline with the OpenCV fallback detector (the YOLO model is
       trained on real field photos, not on this synthetic mosaic), Grad-CAM,
       the drone simulator and the HTML report
    4. print a summary
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey-id", default="demo_survey_001")
    ap.add_argument("--detector", default="classical", choices=["classical", "auto", "yolo"])
    ap.add_argument("--fresh", action="store_true", help="regenerate the field and re-register")
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from app.db import get_connection, init_db

    survey_dir = REPO_ROOT / "data" / "images" / args.survey_id
    if args.fresh or not (survey_dir / "ground_truth.json").exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_demo_field.py"),
                        "--survey-id", args.survey_id], check=True)

    init_db()
    conn = get_connection()
    try:
        exists = conn.execute("SELECT 1 FROM surveys WHERE survey_id = ?",
                              (args.survey_id,)).fetchone()
        if exists and args.fresh:
            for table in ("surveys", "survey_images", "detections", "classifications",
                          "plant_severity", "plant_locations", "spatial_calibrations",
                          "treatment_decisions", "mission_plans", "drone_commands",
                          "execution_logs"):
                conn.execute(f"DELETE FROM {table} WHERE survey_id = ?", (args.survey_id,))
            conn.commit()
            exists = None
    finally:
        conn.close()

    gt = json.loads((survey_dir / "ground_truth.json").read_text(encoding="utf-8"))
    if not exists:
        from app.field_survey.service import register_survey

        reg = register_survey(args.survey_id, field_width_m=gt["field_width_m"],
                              field_height_m=gt["field_height_m"])
        print(f"[M1] registered {args.survey_id}: {reg['num_images']} image(s), "
              f"field {reg['field_size']}")

    from app.pipeline import run_pipeline, stage_statuses

    result = run_pipeline(args.survey_id, detector_mode=args.detector,
                          save_gradcam=not args.no_gradcam, simulate=True,
                          write_report=True)

    print("\nStage status:")
    for k, v in stage_statuses(result).items():
        print(f"  {k:32s} {v}")
    s2 = result["stage_2_detection"]
    s3 = result["stage_3_classification"]
    s5 = result["stage_5_spatial_mapping"]
    print(f"\n[M2] plants detected : {s2.get('plants_detected')} ({s2.get('model_version')})")
    if s3.get("status") == "success":
        print(f"[M3] classified      : {s3['classified']} (uncertain: {s3['uncertain']})")
    if s5.get("status") == "success":
        print(f"[M5] mapping         : {s5['mapping_mode']} "
              f"({s5['mapped_plants']}/{s5['total_plants']} mapped)")
    s6 = result["stage_6_treatment_decision"]
    if s6.get("status") == "success":
        sm = s6["summary"]
        print(f"[M6] actions         : {sm['actions']}")
        print(f"[M6] chemical        : {sm['chemical_used_ml']} mL vs "
              f"{sm['blanket_baseline_ml']} mL blanket -> {sm['chemical_saved_pct']}% saved "
              f"(₹{sm['cost_saved_inr']} saved)")
    s8 = result["stage_8_mission_planning"]
    if s8.get("status") == "success":
        m = s8["metrics"]
        print(f"[M8] route           : {m['optimised_distance_m']} m in {m['sorties']} sortie(s) "
              f"vs {m['blanket_distance_m']} m blanket pass "
              f"({m['distance_saved_vs_blanket_pct']}% shorter); "
              f"{m['route_improvement_vs_id_order_pct']}% shorter than plant-id order")
    if "simulation" in result:
        sim = result["simulation"]
        print(f"[SIM] flight         : {sim['mission_time_s']} s, {sim['distance_flown_m']} m, "
              f"battery {sim['battery_used_pct']}%, placement error "
              f"{sim['mean_placement_error_m']} m, status {sim['status']}")
    gt_eval_path = REPO_ROOT / "data" / "outputs" / args.survey_id / "demo_ground_truth_evaluation.json"
    if gt_eval_path.exists():
        ev = json.loads(gt_eval_path.read_text(encoding="utf-8"))
        print(f"[GT] detection recall {ev['detection_recall']}, precision "
              f"{ev['detection_precision']}, classification accuracy "
              f"{ev['classification_accuracy_on_matched']}, mean localisation error "
              f"{ev['mean_localisation_error_m']} m")
    print(f"\nReport: {REPO_ROOT / result.get('report', '')}")
    print("Open it in a browser, or start the API and visit "
          f"http://127.0.0.1:8000/reports/{args.survey_id}")


if __name__ == "__main__":
    main()
