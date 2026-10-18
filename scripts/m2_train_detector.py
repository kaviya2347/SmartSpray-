"""
FILE PATH: SmartSpray/scripts/m2_train_detector.py

Module 2 — train / evaluate the YOLOv8n tomato-plant detector.

Dataset: LeafPest "Plant Detection" v1 (Roboflow, CC BY 4.0), YOLOv8 export,
546 images (train 460 / valid 57 / test 29), one class "Tomato-Plant".
Put the export at data/datasets/plant_detection/ (folders train/ valid/ test/).

    python scripts/m2_train_detector.py                    # train + test eval + install
    python scripts/m2_train_detector.py --epochs 50 --imgsz 640 --device mps   # Apple GPU
    python scripts/m2_train_detector.py --eval-only

Outputs:
    models/yolov8_plant_detector/best.pt          (used by app/detection/detector.py)
    models/yolov8_plant_detector/eval/metrics.json
    models/yolov8_plant_detector/runs/...         (Ultralytics run folder, git-ignored)

Model selection uses the validation split (Ultralytics keeps best.pt by
validation fitness); the test split is evaluated once at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "datasets" / "plant_detection"
MODEL_DIR = REPO_ROOT / "models" / "yolov8_plant_detector"
BASE_WEIGHTS = REPO_ROOT / "notebooks" / "yolov8n.pt"      # COCO-pretrained YOLOv8n


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_yaml() -> Path:
    for split in ("train", "valid", "test"):
        if not (DATASET / split / "images").is_dir():
            raise SystemExit(f"missing {DATASET / split / 'images'} — copy the Roboflow export there")
    yaml_path = DATASET / "data_smartspray.yaml"
    yaml_path.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames: ['Tomato-Plant']\n", encoding="utf-8")
    return yaml_path


def evaluate(weights: Path, yaml_path: Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for split in ("val", "test"):
        m = model.val(data=str(yaml_path), split=split, imgsz=imgsz, device=device,
                      plots=False, verbose=False, project=str(MODEL_DIR / "runs"),
                      name=f"eval_{split}", exist_ok=True)
        out[split] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                      "precision": float(m.box.mp), "recall": float(m.box.mr)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple) | 0 (CUDA)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml_path = write_yaml()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = MODEL_DIR / "runs" / "train_smartspray"
    best = run_dir / "weights" / "best.pt"

    if not args.eval_only:
        base = str(BASE_WEIGHTS) if BASE_WEIGHTS.exists() else "yolov8n.pt"
        model = YOLO(base)
        model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, patience=args.patience, optimizer="AdamW",
                    lr0=0.001, seed=42, deterministic=True, device=args.device,
                    workers=args.workers, project=str(MODEL_DIR / "runs"),
                    name="train_smartspray", exist_ok=True, plots=False, verbose=True)
    if not best.exists():
        raise SystemExit(f"{best} not found — train first")

    metrics = evaluate(best, yaml_path, args.imgsz, args.device)
    target = MODEL_DIR / "best.pt"
    if target.exists():
        backup = MODEL_DIR / f"best.pt.bak-{datetime.now():%Y%m%d-%H%M%S}"
        shutil.copy2(target, backup)
    shutil.copy2(best, target)
    import yaml

    run_args = {}
    if (run_dir / "args.yaml").exists():
        run_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
    epochs_done = None
    if (run_dir / "results.csv").exists():
        epochs_done = max(0, len((run_dir / "results.csv").read_text().strip().splitlines()) - 1)
    record = {
        "dataset": "LeafPest Plant Detection v1 (Roboflow, CC BY 4.0) — 460/57/29 images",
        "base_weights": BASE_WEIGHTS.name,
        "epochs_trained": epochs_done,
        "imgsz": run_args.get("imgsz", args.imgsz), "batch": run_args.get("batch", args.batch),
        "optimizer": run_args.get("optimizer"), "seed": run_args.get("seed"),
        "selection": "Ultralytics best.pt (validation fitness)",
        "metrics": metrics,
        "model_sha256": sha256(target),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "eval").mkdir(exist_ok=True)
    (MODEL_DIR / "eval" / "metrics.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
