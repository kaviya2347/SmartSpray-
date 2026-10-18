"""
FILE PATH: SmartSpray/scripts/inspect_classifier.py

Shows, step by step, how the backend classifies ONE plant image — the same
code path the API, the pipeline and the simulator use (Modules 3, 4 and 6).
Useful in a review: run it on any photo and every intermediate value is
printed.

USAGE
    python scripts/inspect_classifier.py                       # a random test-split photo
    python scripts/inspect_classifier.py path/to/leaf.jpg
    python scripts/inspect_classifier.py --label late_blight   # random test photo of that class
    python scripts/inspect_classifier.py --survey demo_survey_001 --plant 12
                                                               # a plant crop from a survey

Writes the Grad-CAM overlay next to the run: data/outputs/inspect/gradcam_<name>.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import app  # noqa: E402,F401  (import-order guard for TensorFlow)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

LINE = "─" * 72


def step(n, title):
    print(f"\n{LINE}\nSTEP {n} · {title}\n{LINE}")


def pick_test_image(label: str | None, seed: int | None) -> Path:
    manifest = json.loads((REPO_ROOT / "models/classification/split_manifest.json").read_text())
    rows = [r for r in manifest["images"] if r["split"] == "test" and (not label or r["label"] == label)]
    r = random.Random(seed).choice(rows)
    return REPO_ROOT / "data/datasets/plantvillage_tomato" / r["path"], r["label"]


def crop_from_survey(survey: str, plant: int):
    from app.db import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT image_filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM detections "
                           "WHERE survey_id = ? AND plant_id = ?", (survey, plant)).fetchone()
    finally:
        conn.close()
    if not row:
        sys.exit(f"plant {plant} not found in survey '{survey}' — run the pipeline first")
    img = cv2.imread(str(REPO_ROOT / "data/images" / survey / row[0]))
    x1, y1, x2, y2 = [int(round(v)) for v in row[1:]]
    return img[y1:y2, x1:x2], f"{survey}/{row[0]} plant {plant} bbox {[x1, y1, x2, y2]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--label", help="pick a random test photo of this class")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--survey")
    ap.add_argument("--plant", type=int)
    args = ap.parse_args()

    true_label = None
    if args.survey:
        bgr, source = crop_from_survey(args.survey, args.plant or 1)
    else:
        if args.image:
            path = Path(args.image)
        else:
            path, true_label = pick_test_image(args.label, args.seed)
        bgr = cv2.imread(str(path))
        if bgr is None:
            sys.exit(f"cannot read image {path}")
        source = str(path.relative_to(REPO_ROOT)) if path.is_absolute() and REPO_ROOT in path.parents else str(path)

    from app.classification.classifier import (
        DEFAULT_WEIGHTS_PATH, IMAGE_SIZE, _softmax_with_temperature, get_classifier)
    from app.classification.explain import grad_cam, overlay_heatmap
    from app.decision.rules_engine import decide_plant, load_rules
    from app.severity.estimator import get_estimator

    step(1, "Load the trained model (once per server start)")
    clf = get_classifier()
    model = clf.model
    sha = hashlib.sha256(DEFAULT_WEIGHTS_PATH.read_bytes()).hexdigest()
    print(f"model file      : {DEFAULT_WEIGHTS_PATH.relative_to(REPO_ROOT)}")
    print(f"SHA-256         : {sha}")
    print(f"model version   : {clf.model_version}")
    print(f"parameters      : {model.count_params():,}")
    print("layers          : " + " -> ".join(f"{l.name}" for l in model.layers))
    print(f"class order     : {clf.class_order}")
    print(f"inference config: temperature T = {clf.temperature:.4f}, flip-TTA = {clf.tta}, "
          f"uncertainty threshold = {clf.uncertainty_threshold}")

    step(2, "Input image")
    print(f"source          : {source}")
    if true_label:
        print(f"true label      : {true_label}   (from the dataset folder; the model never sees this)")
    print(f"shape (H, W, C) : {bgr.shape}, dtype {bgr.dtype}, OpenCV channel order BGR")

    step(3, "Pre-processing  (DiseaseClassifier._prepare)")
    x = clf._prepare(bgr)
    print(f"BGR -> RGB, resize to {IMAGE_SIZE}x{IMAGE_SIZE}, float32 -> {x.shape}, "
          f"range {x.min():.0f}..{x.max():.0f}")
    print("pixel scaling and ImageNet normalisation happen INSIDE the model "
          "(EfficientNetB0's own Rescaling + Normalization layers)")

    step(4, "Forward pass  (EfficientNetB0 backbone -> GAP -> Dense(256) -> Dense(4) softmax)")
    batch = x[None]
    p_orig = np.asarray(model(batch, training=False))[0]
    p_flip = np.asarray(model(batch[:, :, ::-1, :], training=False))[0]
    for name, p in (("original image ", p_orig), ("mirrored image ", p_flip)):
        print(f"{name} : " + "  ".join(f"{c} {v:.4f}" for c, v in zip(clf.class_order, p)))
    p_avg = (p_orig + p_flip) / 2 if clf.tta else p_orig
    print(f"averaged (TTA)  : " + "  ".join(f"{c} {v:.4f}" for c, v in zip(clf.class_order, p_avg)))

    step(5, "Confidence calibration (temperature scaling, T fitted on the validation split)")
    p_cal = _softmax_with_temperature(p_avg[None], clf.temperature)[0]
    print("p_cal = softmax(log(p) / T)")
    print("calibrated      : " + "  ".join(f"{c} {v:.4f}" for c, v in zip(clf.class_order, p_cal)))

    step(6, "Prediction  (DiseaseClassifier.predict — what the API returns)")
    pred = clf.predict(bgr)
    print(json.dumps(pred, indent=2))
    verdict = "UNCERTAIN -> manual inspection, never sprayed automatically" if pred["uncertain"] \
        else "confident (>= threshold)"
    print(f"=> {pred['label']} at {pred['confidence']:.1%} : {verdict}")
    if true_label:
        print(f"=> {'CORRECT' if pred['label'] == true_label else 'WRONG'} (true label {true_label})")

    step(7, "Severity  (Module 4, OpenCV lesion-area estimate)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sev = get_estimator().estimate(rgb, pred["label"], 1, "inspect")
    print(f"leaf pixels {sev.leaf_px}, lesion pixels {sev.lesion_px}, "
          f"score {sev.severity_score:.3f} -> {sev.severity_category} (valid={sev.is_valid})")
    print("bands: < 0.20 low, 0.20-0.50 medium, > 0.50 high")

    step(8, "Treatment decision  (Module 6, rules/dosage_rules.json)")
    rules = load_rules()
    d = decide_plant({"plant_id": 1, "disease_label": pred["label"], "confidence": pred["confidence"],
                      "is_uncertain": pred["uncertain"], "severity_score": sev.severity_score,
                      "severity_category": sev.severity_category, "severity_valid": sev.is_valid,
                      "is_mapped": 0}, rules)
    print(f"action {d['action'].upper()} | dose {d['dosage_ml']} mL | treatment {d['treatment']}")
    print(f"reason: {d['reason']}")

    step(9, "Explanation  (Grad-CAM on the last convolutional block)")
    idx = clf.class_order.index(pred["label"])
    heat, _, _ = grad_cam(model, cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)), idx)
    out_dir = REPO_ROOT / "data/outputs/inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(source.split(" ")[0]).stem.replace(" ", "_")[:40]
    over = overlay_heatmap(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)), heat)
    side = np.hstack([cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)), over])
    out = out_dir / f"gradcam_{name}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"red = regions that pushed the score for '{pred['label']}' up")
    print(f"saved: {out.relative_to(REPO_ROOT)}  (left: input, right: heatmap)")


if __name__ == "__main__":
    main()
