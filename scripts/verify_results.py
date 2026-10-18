"""
Re-check the Module 3 headline numbers from the shipped model weights.

    python scripts/verify_results.py

Needs the PlantVillage tomato images in data/datasets/plantvillage_tomato/
(see "Datasets" in the README). The script:

1. checks the SHA-256 of models/classification/best.keras against the
   evaluation record, so you know you are testing the promoted model;
2. reads the locked 70/15/15 split from split_manifest.csv and checks every
   test image's SHA-256, so the test set is exactly the one reported;
3. classifies the 752 test images twice: raw (single pass, no calibration) and
   production mode (horizontal-flip TTA + temperature, as the API serves it);
4. prints accuracy, macro F1, confident coverage, and accuracy on confident
   predictions next to the values recorded in models/classification/eval/metrics.json.

It never writes to models/ and does not touch the TEST_EVALUATED lock.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "datasets" / "plantvillage_tomato"
MODEL = ROOT / "models" / "classification" / "best.keras"
MANIFEST = ROOT / "models" / "classification" / "split_manifest.csv"
METRICS = ROOT / "models" / "classification" / "eval" / "metrics.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main() -> int:
    recorded = json.loads(METRICS.read_text())

    model_hash = sha256(MODEL)
    ok_model = model_hash == recorded["evaluated_model_sha256"]
    print(f"model sha256 matches evaluation record: {ok_model}")

    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "test"]
    if not DATA.exists():
        print(f"Dataset not found at {DATA}. See README → Datasets.")
        return 2
    missing = [r["path"] for r in rows if not (DATA / r["path"]).exists()]
    if missing:
        print(f"{len(missing)} test images missing, e.g. {missing[:3]}")
        return 2
    bad_hash = [r["path"] for r in rows if sha256(DATA / r["path"]) != r["sha256"]]
    print(f"test images: {len(rows)}; sha256 mismatches: {len(bad_hash)}")

    from app.classification.classifier import DiseaseClassifier

    clf = DiseaseClassifier()
    classes = clf.class_order
    crops = [cv2.imread(str(DATA / r["path"])) for r in rows]
    y_true = [r["label"] for r in rows]

    # Raw: one forward pass, no TTA, no temperature (how the test set was first scored).
    batch = np.stack([clf._prepare(c) for c in crops])
    raw = np.asarray(clf.model.predict(batch, batch_size=64, verbose=0))
    y_raw = [classes[i] for i in raw.argmax(1)]

    # Production: exactly what the API returns.
    prod = []
    for i in range(0, len(crops), 64):
        prod.extend(clf.predict_batch(crops[i:i + 64]))
    y_prod = [p["label"] for p in prod]
    confident = [i for i, p in enumerate(prod) if not p["uncertain"]]

    acc_raw = np.mean([t == p for t, p in zip(y_true, y_raw)])
    acc_prod = np.mean([t == p for t, p in zip(y_true, y_prod)])
    cov = len(confident) / len(rows)
    acc_conf = np.mean([y_true[i] == y_prod[i] for i in confident])
    pm = recorded["production_mode"]

    print()
    print(f"{'metric':38s} {'re-computed':>12s} {'recorded':>10s}")
    for name, got, want in [
        ("test accuracy (raw)", acc_raw, recorded["test_accuracy"]),
        ("macro F1 (raw)", macro_f1(y_true, y_raw, classes), recorded["macro_f1"]),
        ("accuracy (production: TTA + temp.)", acc_prod, pm["accuracy"]),
        ("confident coverage (>= 0.60)", cov, pm["coverage_confident"]),
        ("accuracy on confident predictions", acc_conf, pm["accuracy_on_confident"]),
    ]:
        print(f"{name:38s} {got:12.4f} {want:10.4f}")
    print(f"correct (raw): {sum(t == p for t, p in zip(y_true, y_raw))}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
