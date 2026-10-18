"""
FILE PATH: SmartSpray/scripts/m3_train_classifier.py

Module 3 (Disease Classification): reproducible rebuild with a REAL,
verifiable 70 / 15 / 15 train / validation / test split.

Stages (run all with `--stage all`, or one at a time):

    split     Scan the dataset, hash every file, drop exact duplicates,
              check every image decodes, make a stratified 70/15/15 split with
              sklearn (random_state=RANDOM_SEED on BOTH calls), verify zero
              overlap by path AND by SHA-256, write the split manifest.
    train     Phase 1 (frozen backbone) + Phase 2 (top-30 layers fine-tuned).
              Checkpoint selection, early stopping, temperature scaling and the
              TTA on/off decision use the VALIDATION split only.
    deep      Phase 3 (pre-registered before any test evaluation): continue
              from the selected checkpoint with blocks 5-7 unfrozen
              (118 layers, BatchNorm frozen) at lr 1e-4. The result replaces
              the candidate ONLY if it beats it on validation.
    evaluate  The single, final pass over the untouched TEST split. Writes
              every metric/artifact and a lock file so it is not silently
              re-run.
    promote   Back up the current best.keras (SHA-256 recorded), verify the
              candidate, copy it to models/classification/best.keras.

USAGE (from the repo root):
    python scripts/m3_train_classifier.py --stage all
    python scripts/m3_train_classifier.py --stage split
    python scripts/m3_train_classifier.py --stage train
    python scripts/m3_train_classifier.py --stage deep
    python scripts/m3_train_classifier.py --stage evaluate
    python scripts/m3_train_classifier.py --stage promote

The dataset root defaults to data/datasets/plantvillage_tomato with one
sub-folder per canonical class name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.classification.prediction import CLASS_NAMES, MODEL_VERSION  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42                 # passed as random_state= to BOTH split calls
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 20                 # Phase 1 max epochs (backbone frozen)
EPOCHS_FINETUNE = 10             # Phase 2 max epochs
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_LAYERS = 30
DROPOUT_RATE = 0.3
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
# Phase 3 (deeper fine-tune) — fixed before any test-set evaluation.
DEEP_UNFREEZE_FROM = "block5a_expand_conv"   # blocks 5, 6, 7 + top (118 layers)
EPOCHS_DEEP = 8
LR_DEEP = 1e-4
DEEP_EARLY_STOP_PATIENCE = 3
DEEP_REDUCE_LR_PATIENCE = 2
UNCERTAINTY_THRESHOLD = 0.60     # conf < 0.60 -> flagged uncertain (PPT spec)

VALID_EXT = {".jpg", ".jpeg", ".png"}

DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "plantvillage_tomato"
MODEL_DIR = REPO_ROOT / "models" / "classification"
EVAL_DIR = MODEL_DIR / "eval"
CKPT_DIR = EVAL_DIR / "checkpoints"
MANIFEST_JSON = MODEL_DIR / "split_manifest.json"
MANIFEST_CSV = MODEL_DIR / "split_manifest.csv"
CLASS_ORDER_JSON = MODEL_DIR / "class_order.json"
INFERENCE_CONFIG_JSON = MODEL_DIR / "inference_config.json"
PROD_MODEL = MODEL_DIR / "best.keras"
CANDIDATE_MODEL = EVAL_DIR / "candidate_best.keras"
TEST_LOCK = EVAL_DIR / "TEST_EVALUATED.lock"

logger = logging.getLogger("m3_train")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(EVAL_DIR / "run_log.txt", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", force=True)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import keras
    keras.utils.set_random_seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def banner(title: str) -> None:
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def load_image_rgb(path: Path) -> np.ndarray:
    """
    Exactly the production preprocessing in app/classification/classifier.py:
    cv2.imread (BGR) -> RGB -> cv2.resize(224, 224) -> uint8.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — split
# ══════════════════════════════════════════════════════════════════════════════

def stage_split(dataset_root: Path) -> dict:
    from sklearn.model_selection import train_test_split

    banner("STAGE 1 — Dataset scan, de-duplication and 70/15/15 split")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    folders = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    logger.info(f"Dataset root   : {dataset_root}")
    logger.info(f"Class folders  : {folders}")
    if folders != sorted(CLASS_NAMES):
        raise ValueError(f"class folders {folders} != CLASS_NAMES {CLASS_NAMES}")
    if sorted(CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("CLASS_NAMES must be alphabetical (Keras convention)")
    label_to_index = {c: i for i, c in enumerate(CLASS_NAMES)}

    files = []
    ignored = []
    for cls in CLASS_NAMES:
        for p in sorted((dataset_root / cls).iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VALID_EXT:
                ignored.append(str(p.relative_to(dataset_root)))
                continue
            files.append((cls, p))
    raw_counts = Counter(c for c, _ in files)
    logger.info(f"Image files found (case-insensitive extension match): {len(files)}")
    for cls in CLASS_NAMES:
        logger.info(f"   {cls:22s}: {raw_counts[cls]}")
    if ignored:
        logger.info(f"Ignored non-image files: {ignored}")

    # Hash + decode check
    records = []
    unreadable = []
    for cls, p in files:
        digest = sha256_file(p)
        img = cv2.imread(str(p))
        if img is None:
            unreadable.append(str(p.relative_to(dataset_root)))
            continue
        records.append({
            "path": p.relative_to(dataset_root).as_posix(),
            "label": cls,
            "label_index": label_to_index[cls],
            "sha256": digest,
            "height": int(img.shape[0]),
            "width": int(img.shape[1]),
        })

    # Exact-duplicate handling (same bytes, different file name).
    by_hash = defaultdict(list)
    for r in records:
        by_hash[r["sha256"]].append(r)
    unique, duplicates, conflicts = [], [], []
    for digest, group in by_hash.items():
        group = sorted(group, key=lambda r: r["path"])
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            conflicts.extend(group)          # same image, different labels
            continue
        unique.append(group[0])
        for dup in group[1:]:
            duplicates.append({**dup, "duplicate_of": group[0]["path"]})
    unique.sort(key=lambda r: r["path"])
    logger.info(f"Unreadable images            : {len(unreadable)}")
    logger.info(f"Exact duplicates removed     : {len(duplicates)}")
    logger.info(f"Cross-class label conflicts  : {len(conflicts)}")
    logger.info(f"Unique usable images         : {len(unique)}")

    paths = [r["path"] for r in unique]
    labels = [r["label"] for r in unique]

    # Split 1: 70% train / 30% holdout (stratified, seeded)
    train_p, hold_p, train_y, hold_y = train_test_split(
        paths, labels,
        test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    # Split 2: holdout -> 50% validation / 50% test (stratified, seeded)
    val_p, test_p, val_y, test_y = train_test_split(
        hold_p, hold_y,
        test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        stratify=hold_y,
        random_state=RANDOM_SEED,
    )
    split_of = {**{p: "train" for p in train_p},
                **{p: "validation" for p in val_p},
                **{p: "test" for p in test_p}}
    for r in unique:
        r["split"] = split_of[r["path"]]

    # Leakage checks — by path AND by content hash.
    sets_path = {s: {r["path"] for r in unique if r["split"] == s}
                 for s in ("train", "validation", "test")}
    sets_hash = {s: {r["sha256"] for r in unique if r["split"] == s}
                 for s in ("train", "validation", "test")}
    checks = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        checks[f"path_overlap_{a}_{b}"] = len(sets_path[a] & sets_path[b])
        checks[f"hash_overlap_{a}_{b}"] = len(sets_hash[a] & sets_hash[b])
    union = sets_path["train"] | sets_path["validation"] | sets_path["test"]
    checks["every_unique_image_in_exactly_one_split"] = (
        len(union) == len(unique)
        and sum(len(v) for v in sets_path.values()) == len(unique)
    )
    # Duplicates: the dropped copy must not sneak into any split.
    checks["dropped_duplicates_absent_from_splits"] = all(
        d["path"] not in union for d in duplicates)
    leak_free = all(v == 0 for k, v in checks.items() if "overlap" in k)
    if not leak_free or not checks["every_unique_image_in_exactly_one_split"]:
        raise RuntimeError(f"Split verification FAILED: {checks}")

    per_split = {s: Counter(r["label"] for r in unique if r["split"] == s)
                 for s in ("train", "validation", "test")}
    counts = {s: {c: int(per_split[s][c]) for c in CLASS_NAMES}
              for s in per_split}
    totals = {s: sum(v.values()) for s, v in counts.items()}

    logger.info("")
    logger.info(f"{'class':22s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for c in CLASS_NAMES:
        tot = sum(counts[s][c] for s in counts)
        logger.info(f"{c:22s} {tot:6d} {counts['train'][c]:6d} "
                    f"{counts['validation'][c]:6d} {counts['test'][c]:6d}")
    logger.info(f"{'TOTAL':22s} {len(unique):6d} {totals['train']:6d} "
                f"{totals['validation']:6d} {totals['test']:6d}")
    logger.info("")
    logger.info("LEAKAGE CHECKS")
    for k, v in checks.items():
        logger.info(f"   {k:48s}: {v}")
    logger.info(f"   RESULT: {'PASS — zero overlap' if leak_free else 'FAIL'}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": dataset_root.relative_to(REPO_ROOT).as_posix()
        if dataset_root.is_relative_to(REPO_ROOT) else str(dataset_root),
        "dataset_source": "PlantVillage (tomato subset: early_blight, healthy, "
                          "late_blight, septoria_leaf_spot)",
        "class_order": CLASS_NAMES,
        "random_seed": RANDOM_SEED,
        "split_method": (
            "sklearn.model_selection.train_test_split called twice, "
            "stratify=labels, random_state=RANDOM_SEED: "
            "(1) 70% train vs 30% holdout; (2) holdout 50/50 -> validation/test"
        ),
        "fractions": {"train": TRAIN_FRACTION, "validation": VAL_FRACTION,
                      "test": TEST_FRACTION},
        "raw_file_counts": {c: int(raw_counts[c]) for c in CLASS_NAMES},
        "raw_file_total": len(files),
        "unique_images": len(unique),
        "split_counts": counts,
        "split_totals": totals,
        "leakage_checks": checks,
        "excluded": {
            "exact_duplicates": duplicates,
            "unreadable": unreadable,
            "label_conflicts": conflicts,
            "non_image_files": ignored,
        },
        "images": unique,
    }
    write_json(MANIFEST_JSON, manifest)
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "label_index",
                                          "split", "sha256", "height", "width"])
        w.writeheader()
        for r in unique:
            w.writerow({k: r[k] for k in w.fieldnames})
    write_json(EVAL_DIR / "split_summary.json",
               {k: manifest[k] for k in manifest if k not in ("images",)})
    write_json(CLASS_ORDER_JSON, {"class_order": CLASS_NAMES})
    logger.info(f"Split manifest written: {MANIFEST_JSON.relative_to(REPO_ROOT)}")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_JSON.exists():
        raise FileNotFoundError("split_manifest.json missing — run --stage split")
    with open(MANIFEST_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_split_arrays(manifest: dict, split: str, dataset_root: Path):
    rows = [r for r in manifest["images"] if r["split"] == split]
    x = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.empty((len(rows),), dtype=np.int32)
    for i, r in enumerate(rows):
        x[i] = load_image_rgb(dataset_root / r["path"])
        y[i] = r["label_index"]
    return x, y, rows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — train (validation-only model selection)
# ══════════════════════════════════════════════════════════════════════════════

def make_datasets(x_train, y_train, x_val, y_val):
    import tensorflow as tf
    import keras
    from keras import layers

    # Augmentation is applied ONLY to the training pipeline.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        layers.RandomRotation(20 / 360, fill_mode="nearest", seed=RANDOM_SEED),
        layers.RandomTranslation(0.15, 0.15, fill_mode="nearest", seed=RANDOM_SEED),
        layers.RandomZoom(0.15, fill_mode="nearest", seed=RANDOM_SEED),
        layers.RandomBrightness(0.2, value_range=(0, 255), seed=RANDOM_SEED),
    ], name="train_augmentation")

    def aug_fn(img, lbl):
        img = tf.cast(img, tf.float32)
        img = augment(img, training=True)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, lbl

    def plain_fn(img, lbl):
        return tf.cast(img, tf.float32), lbl

    autotune = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train), seed=RANDOM_SEED, reshuffle_each_iteration=True)
                .batch(BATCH_SIZE)
                .map(aug_fn, num_parallel_calls=autotune)
                .prefetch(autotune))
    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .batch(BATCH_SIZE)
              .map(plain_fn, num_parallel_calls=autotune)
              .prefetch(autotune))
    return train_ds, val_ds, [l.name for l in augment.layers]


class ValidationSelector:
    """
    Keras callback factory: keeps the weights of the best epoch according to
    (highest val_accuracy, then lowest val_loss). Validation data only.
    """

    def __init__(self, path: Path, phase: str, state: dict):
        import keras

        selector = self

        class _CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                key = (round(float(logs["val_accuracy"]), 6),
                       -round(float(logs["val_loss"]), 6))
                if state.get("best_key") is None or key > state["best_key"]:
                    state.update(best_key=key, phase=phase, epoch=epoch + 1,
                                 val_accuracy=float(logs["val_accuracy"]),
                                 val_loss=float(logs["val_loss"]))
                    self.model.save_weights(str(selector.path))
                    logger.info(f"   [select] new best ({phase} epoch {epoch + 1}): "
                                f"val_acc={logs['val_accuracy']:.4f} "
                                f"val_loss={logs['val_loss']:.4f}")

        self.path = path
        self.callback = _CB()


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017) fitted by NLL on validation."""
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(probs, 1e-12, 1.0))

    def nll(log_t):
        t = np.exp(log_t)
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_of(probs, y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))))


def ece_of(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def predict_probs(model, x: np.ndarray, tta: bool) -> np.ndarray:
    xf = x.astype(np.float32)
    p = model.predict(xf, batch_size=64, verbose=0)
    if tta:
        p2 = model.predict(xf[:, :, ::-1, :], batch_size=64, verbose=0)
        p = (p + p2) / 2.0
    return p


def save_clean_model(trained_model, path: Path) -> None:
    """Save an uncompiled copy (no optimizer state) -> smaller .keras file."""
    from app.classification.model_factory import build_model

    clean, clean_base, _ = build_model(len(CLASS_NAMES), DROPOUT_RATE)
    clean.set_weights(trained_model.get_weights())
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(str(path))


def stage_train(dataset_root: Path) -> dict:
    import keras
    import tensorflow as tf
    from app.classification.model_factory import build_model, unfreeze_top_layers

    banner("STAGE 2 — Training (validation-only model selection)")
    set_seeds()
    manifest = load_manifest()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x_train, y_train, _ = load_split_arrays(manifest, "train", dataset_root)
    x_val, y_val, _ = load_split_arrays(manifest, "validation", dataset_root)
    logger.info(f"Loaded train={len(x_train)} validation={len(x_val)} "
                f"in {time.time() - t0:.1f}s (test split NOT loaded)")

    train_ds, val_ds, aug_layers = make_datasets(x_train, y_train, x_val, y_val)
    logger.info(f"Train-only augmentation: {aug_layers}")
    logger.info("Validation pipeline   : cast to float32 only (no augmentation)")

    model, base, weights_source = build_model(len(CLASS_NAMES), DROPOUT_RATE)
    logger.info(f"Backbone weights source: {weights_source}")
    best_path = CKPT_DIR / "best_by_validation.weights.h5"
    state: dict = {}

    def callbacks(phase):
        return [
            ValidationSelector(best_path, phase, state).callback,
            keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                          patience=EARLY_STOP_PATIENCE,
                                          mode="max", restore_best_weights=False,
                                          verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=REDUCE_LR_PATIENCE,
                                              min_lr=1e-7, verbose=1),
            keras.callbacks.CSVLogger(str(EVAL_DIR / f"history_{phase}.csv")),
        ]

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable_p1 = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    frozen_p1 = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    logger.info(f"Phase 1: backbone frozen | trainable params={trainable_p1:,} "
                f"frozen params={frozen_p1:,} | Adam lr={LR_HEAD} | max {EPOCHS_HEAD} epochs")
    t1 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                   callbacks=callbacks("phase1"), verbose=2)
    phase1_time = time.time() - t1
    phase1_best = dict(state)
    logger.info(f"Phase 1 done in {phase1_time/60:.1f} min; best so far: {phase1_best}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    model.load_weights(str(best_path))            # start from Phase-1 best
    n_unfrozen = unfreeze_top_layers(base, FINE_TUNE_LAYERS, keep_bn_frozen=True)
    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable_p2 = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    logger.info(f"Phase 2: top {FINE_TUNE_LAYERS} backbone layers unfrozen "
                f"({n_unfrozen} weighted, BatchNorm kept frozen) | trainable params="
                f"{trainable_p2:,} | Adam lr={LR_FINETUNE} | max {EPOCHS_FINETUNE} epochs")
    t2 = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
                   callbacks=callbacks("phase2"), verbose=2)
    phase2_time = time.time() - t2

    # ── Selection (validation only) ──────────────────────────────────────────
    model.load_weights(str(best_path))
    selected = {k: v for k, v in state.items() if k != "best_key"}
    logger.info(f"SELECTED CHECKPOINT (validation only): {selected}")

    decision = finalize_candidate(model, x_val, y_val)
    val_probs_plain = decision.pop("_val_probs_plain")

    save_clean_model(model, CANDIDATE_MODEL)
    logger.info(f"Candidate model saved: {CANDIDATE_MODEL.relative_to(REPO_ROOT)} "
                f"({CANDIDATE_MODEL.stat().st_size/1e6:.1f} MB)")

    def hist(h):
        return {k: [float(v) for v in vals] for k, vals in h.history.items()}

    training_record = {
        "model_version": MODEL_VERSION,
        "architecture": "EfficientNetB0(ImageNet) -> GAP -> Dropout(0.3) -> "
                        "Dense(256, relu) -> Dropout(0.2) -> Dense(4, softmax)",
        "backbone_weights_source": weights_source,
        "hyperparameters": {
            "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
            "epochs_head_max": EPOCHS_HEAD, "epochs_finetune_max": EPOCHS_FINETUNE,
            "lr_head": LR_HEAD, "lr_finetune": LR_FINETUNE,
            "fine_tune_layers": FINE_TUNE_LAYERS, "batchnorm_frozen_in_phase2": True,
            "dropout": DROPOUT_RATE, "early_stop_patience": EARLY_STOP_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE, "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy", "random_seed": RANDOM_SEED,
            "augmentation_train_only": aug_layers,
        },
        "parameters": {"phase1_trainable": trainable_p1, "phase1_frozen": frozen_p1,
                       "phase2_trainable": trainable_p2,
                       "phase2_weighted_layers_unfrozen": n_unfrozen},
        "phase1": {"epochs_run": len(h1.history["loss"]), "minutes": phase1_time / 60,
                   "history": hist(h1), "best": {k: v for k, v in phase1_best.items()
                                                 if k != "best_key"}},
        "phase2": {"epochs_run": len(h2.history["loss"]), "minutes": phase2_time / 60,
                   "history": hist(h2)},
        "selected_checkpoint": selected,
        "selection_rule": "max val_accuracy, ties broken by min val_loss "
                          "(validation split only)",
        "validation_inference_decision": decision["validation_inference_decision"],
        "calibration": decision["calibration"],
        "environment": {"python": platform.python_version(),
                        "tensorflow": tf.__version__, "keras": keras.__version__,
                        "platform": platform.platform()},
        "test_split_used_during_training": False,
    }
    write_json(EVAL_DIR / "training_history.json", training_record)
    write_inference_config(decision)
    plot_training_curves(training_record)
    del val_probs_plain
    return training_record


def finalize_candidate(model, x_val, y_val) -> dict:
    """
    Validation-only post-processing of the selected weights: decide whether
    horizontal-flip TTA helps and fit the calibration temperature.
    """
    val_probs_plain = predict_probs(model, x_val, tta=False)
    val_probs_tta = predict_probs(model, x_val, tta=True)
    acc_plain = float((val_probs_plain.argmax(1) == y_val).mean())
    acc_tta = float((val_probs_tta.argmax(1) == y_val).mean())
    nll_plain, nll_tta = nll_of(val_probs_plain, y_val), nll_of(val_probs_tta, y_val)
    use_tta = bool(acc_tta >= acc_plain and nll_tta <= nll_plain)
    val_probs = val_probs_tta if use_tta else val_probs_plain
    temperature = fit_temperature(val_probs, y_val)
    val_cal = apply_temperature(val_probs, temperature)
    logger.info(f"Validation acc plain={acc_plain:.4f} tta={acc_tta:.4f}; "
                f"NLL plain={nll_plain:.4f} tta={nll_tta:.4f} -> use_tta={use_tta}")
    logger.info(f"Temperature (fit on validation) T={temperature:.4f}; "
                f"val NLL {nll_of(val_probs, y_val):.4f} -> {nll_of(val_cal, y_val):.4f}; "
                f"val ECE {ece_of(val_probs, y_val):.4f} -> {ece_of(val_cal, y_val):.4f}")
    return {
        "_val_probs_plain": val_probs_plain,
        "validation_inference_decision": {
            "acc_plain": acc_plain, "acc_tta_hflip": acc_tta,
            "nll_plain": nll_plain, "nll_tta_hflip": nll_tta, "use_tta": use_tta,
        },
        "calibration": {
            "method": "temperature scaling fitted on validation NLL",
            "temperature": temperature,
            "val_nll_before": nll_of(val_probs, y_val),
            "val_nll_after": nll_of(val_cal, y_val),
            "val_ece_before": ece_of(val_probs, y_val),
            "val_ece_after": ece_of(val_cal, y_val),
        },
    }


def write_inference_config(decision: dict) -> None:
    write_json(EVAL_DIR / "candidate_inference_config.json", {
        "model_version": MODEL_VERSION,
        "class_order": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "input": "RGB uint8 0-255, resized with cv2.resize (bilinear); "
                 "EfficientNet preprocess_input is a pass-through",
        "temperature": decision["calibration"]["temperature"],
        "tta_hflip": decision["validation_inference_decision"]["use_tta"],
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
    })


def stage_deep(dataset_root: Path) -> dict:
    """Phase 3 — deeper fine-tuning, accepted only if validation improves."""
    import keras

    banner("STAGE 2b — Phase 3 deeper fine-tune (validation-gated)")
    if TEST_LOCK.exists():
        raise RuntimeError("Test set already evaluated — refusing to train further "
                           "(that would turn the test set into a selection set).")
    set_seeds()
    manifest = load_manifest()
    with open(EVAL_DIR / "training_history.json", encoding="utf-8") as f:
        record = json.load(f)
    prev = record["selected_checkpoint"]
    state = {"best_key": (round(prev["val_accuracy"], 6), -round(prev["val_loss"], 6)),
             **prev}
    logger.info(f"Current candidate (validation): {prev}")

    x_train, y_train, _ = load_split_arrays(manifest, "train", dataset_root)
    x_val, y_val, _ = load_split_arrays(manifest, "validation", dataset_root)
    train_ds, val_ds, _ = make_datasets(x_train, y_train, x_val, y_val)

    model = keras.models.load_model(str(CANDIDATE_MODEL))
    base = model.get_layer("efficientnetb0")
    names = [l.name for l in base.layers]
    start = names.index(DEEP_UNFREEZE_FROM)
    base.trainable = True
    n_unfrozen = 0
    for i, layer in enumerate(base.layers):
        trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = trainable
        n_unfrozen += int(trainable and bool(layer.weights))
    model.compile(optimizer=keras.optimizers.Adam(LR_DEEP),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable_p3 = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    logger.info(f"Phase 3: unfrozen from {DEEP_UNFREEZE_FROM} ({len(names) - start} layers, "
                f"{n_unfrozen} weighted, BatchNorm frozen) | trainable params={trainable_p3:,} "
                f"| Adam lr={LR_DEEP} | max {EPOCHS_DEEP} epochs")

    best_path = CKPT_DIR / "best_by_validation_phase3.weights.h5"
    selector = ValidationSelector(best_path, "phase3", state)
    t3 = time.time()
    h3 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_DEEP, verbose=2,
        callbacks=[
            selector.callback,
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                          patience=DEEP_EARLY_STOP_PATIENCE, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=DEEP_REDUCE_LR_PATIENCE,
                                              min_lr=1e-7, verbose=1),
            keras.callbacks.CSVLogger(str(EVAL_DIR / "history_phase3.csv")),
        ])
    minutes = (time.time() - t3) / 60
    improved = state.get("phase") == "phase3"
    record["phase3"] = {
        "epochs_run": len(h3.history["loss"]), "minutes": minutes,
        "unfreeze_from": DEEP_UNFREEZE_FROM, "lr": LR_DEEP,
        "weighted_layers_unfrozen": n_unfrozen, "trainable_params": trainable_p3,
        "history": {k: [float(v) for v in vals] for k, vals in h3.history.items()},
        "improved_validation": improved,
    }
    if improved:
        model.load_weights(str(best_path))
        record["selected_checkpoint"] = {k: v for k, v in state.items() if k != "best_key"}
        decision = finalize_candidate(model, x_val, y_val)
        decision.pop("_val_probs_plain")
        record["validation_inference_decision"] = decision["validation_inference_decision"]
        record["calibration"] = decision["calibration"]
        save_clean_model(model, CANDIDATE_MODEL)
        write_inference_config(decision)
        logger.info(f"Phase 3 IMPROVED validation -> new candidate {record['selected_checkpoint']}")
    else:
        logger.info("Phase 3 did not beat the Phase 1/2 checkpoint on validation; "
                    "candidate unchanged.")
    write_json(EVAL_DIR / "training_history.json", record)
    plot_training_curves(record)
    return record


def plot_training_curves(rec: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h1, h2 = rec["phase1"]["history"], rec["phase2"]["history"]
    h3 = rec.get("phase3", {}).get("history", {"loss": [], "accuracy": [],
                                               "val_loss": [], "val_accuracy": []})
    n1, n2 = len(h1["loss"]), len(h2["loss"])
    ep = np.arange(1, n1 + n2 + len(h3["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(ep, h1[key] + h2[key] + h3[key], color="#2a6fdb", label=f"train {key}")
        ax.plot(ep, h1["val_" + key] + h2["val_" + key] + h3["val_" + key],
                color="#d9480f", label=f"validation {key}")
        ax.axvline(n1 + 0.5, color="#888", ls="--", lw=1, label="phase 2 (top-30 fine-tune)")
        if h3["loss"]:
            ax.axvline(n1 + n2 + 0.5, color="#555", ls="-.", lw=1,
                       label="phase 3 (blocks 5-7 fine-tune)")
        sel = rec["selected_checkpoint"]
        offset = {"phase1": 0, "phase2": n1, "phase3": n1 + n2}[sel["phase"]]
        sel_ep = offset + sel["epoch"]
        ax.axvline(sel_ep, color="#2b8a3e", ls=":", lw=1.5, label="selected epoch")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Module 3 training — validation-selected checkpoint marked in green")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "training_curves.png", dpi=140)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — final, single test evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stage_evaluate(dataset_root: Path, force: bool = False) -> dict:
    import keras
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    banner("STAGE 3 — Independent TEST evaluation (single pass)")
    if TEST_LOCK.exists() and not force:
        raise RuntimeError(
            f"Test set already evaluated ({TEST_LOCK.read_text().strip()}). "
            "Refusing to evaluate again. Use --force-reeval only for a documented "
            "re-run of the SAME model.")
    manifest = load_manifest()
    with open(EVAL_DIR / "candidate_inference_config.json", encoding="utf-8") as f:
        icfg = json.load(f)
    with open(EVAL_DIR / "training_history.json", encoding="utf-8") as f:
        trec = json.load(f)

    model = keras.models.load_model(str(CANDIDATE_MODEL))
    x_test, y_test, rows = load_split_arrays(manifest, "test", dataset_root)
    logger.info(f"Test images loaded: {len(x_test)} (untouched until now)")

    # Loss/accuracy of the plain model (Keras evaluate) — the headline numbers.
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    test_loss, test_acc_keras = model.evaluate(x_test.astype(np.float32), y_test,
                                               batch_size=64, verbose=0)
    probs_plain = predict_probs(model, x_test, tta=False)
    probs_tta = predict_probs(model, x_test, tta=True) if icfg["tta_hflip"] else None
    TEST_LOCK.write_text(f"test evaluated at {datetime.now(timezone.utc).isoformat()} "
                         f"model={CANDIDATE_MODEL.name}\n")

    y_pred = probs_plain.argmax(1)
    acc = accuracy_score(y_test, y_pred)
    report_txt = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                        digits=4, output_dict=True)
    p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred,
                                                    labels=list(range(len(CLASS_NAMES))))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    n = len(y_test)
    correct = int((y_pred == y_test).sum())

    # Production inference mode (as decided on validation): TTA + temperature.
    prod_probs = probs_tta if probs_tta is not None else probs_plain
    prod_cal = apply_temperature(prod_probs, icfg["temperature"])
    prod_pred = prod_cal.argmax(1)
    prod_acc = float((prod_pred == y_test).mean())
    conf = prod_cal.max(1)
    confident = conf >= icfg["uncertainty_threshold"]

    counts = manifest["split_totals"]
    lines = [
        "",
        f"Total images:            {manifest['unique_images']}  "
        f"(raw files {manifest['raw_file_total']}, "
        f"{len(manifest['excluded']['exact_duplicates'])} exact duplicates removed)",
        f"Training:                {counts['train']}",
        f"Validation:              {counts['validation']}",
        f"Test:                    {counts['test']}",
        "",
        f"Test Loss:               {test_loss:.4f}",
        f"Test Accuracy:           {acc:.4f}  ({acc*100:.2f}%)",
        "",
        "Classification Report:",
        report_txt,
        f"Macro F1:                {macro_f1:.4f}",
        f"Weighted F1:             {weighted_f1:.4f}",
        "",
        "Confusion Matrix (rows = true, cols = predicted):",
        f"{'':22s}" + "".join(f"{c[:12]:>14s}" for c in CLASS_NAMES),
    ]
    for i, c in enumerate(CLASS_NAMES):
        lines.append(f"{c:22s}" + "".join(f"{v:14d}" for v in cm[i]))
    lines += [
        "",
        f"Total test samples:      {n}",
        f"Correct predictions:     {correct}",
        f"Incorrect predictions:   {n - correct}",
        "",
        "Production inference mode (chosen on validation): "
        f"TTA={icfg['tta_hflip']}, temperature={icfg['temperature']:.4f}",
        f"   accuracy                         : {prod_acc:.4f}",
        f"   ECE raw -> calibrated            : {ece_of(prod_probs, y_test):.4f} -> "
        f"{ece_of(prod_cal, y_test):.4f}",
        f"   plants with conf >= {icfg['uncertainty_threshold']:.2f}        : "
        f"{int(confident.sum())}/{n} ({confident.mean()*100:.1f}%)",
        f"   accuracy on confident plants     : "
        f"{(prod_pred[confident] == y_test[confident]).mean():.4f}",
        f"   flagged uncertain                : {int((~confident).sum())} "
        f"(of which wrong: {int((prod_pred[~confident] != y_test[~confident]).sum())})",
    ]
    for line in lines:
        logger.info(line)

    metrics = {
        "model_version": MODEL_VERSION,
        "evaluated_model": CANDIDATE_MODEL.relative_to(REPO_ROOT).as_posix(),
        "evaluated_model_sha256": sha256_file(CANDIDATE_MODEL),
        "split_manifest": MANIFEST_JSON.relative_to(REPO_ROOT).as_posix(),
        "split_manifest_sha256": sha256_file(MANIFEST_JSON),
        "counts": {"total_unique": manifest["unique_images"], **counts},
        "test_loss": float(test_loss),
        "test_accuracy": float(acc),
        "test_accuracy_keras_evaluate": float(test_acc_keras),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
        "total_test_samples": n,
        "correct": correct,
        "incorrect": n - correct,
        "production_mode": {
            "tta_hflip": icfg["tta_hflip"], "temperature": icfg["temperature"],
            "accuracy": prod_acc,
            "nll_raw": nll_of(prod_probs, y_test), "nll_calibrated": nll_of(prod_cal, y_test),
            "ece_raw": ece_of(prod_probs, y_test), "ece_calibrated": ece_of(prod_cal, y_test),
            "uncertainty_threshold": icfg["uncertainty_threshold"],
            "coverage_confident": float(confident.mean()),
            "accuracy_on_confident": float((prod_pred[confident] == y_test[confident]).mean()),
            "n_flagged_uncertain": int((~confident).sum()),
        },
        "selected_checkpoint": trec["selected_checkpoint"],
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(EVAL_DIR / "metrics.json", metrics)
    (EVAL_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    (EVAL_DIR / "test_evaluation_printout.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(EVAL_DIR / "classification_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for i, c in enumerate(CLASS_NAMES):
            w.writerow([c, f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
        w.writerow(["macro avg", f"{report_dict['macro avg']['precision']:.4f}",
                    f"{report_dict['macro avg']['recall']:.4f}", f"{macro_f1:.4f}", n])
        w.writerow(["weighted avg", f"{report_dict['weighted avg']['precision']:.4f}",
                    f"{report_dict['weighted avg']['recall']:.4f}", f"{weighted_f1:.4f}", n])
    with open(EVAL_DIR / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + CLASS_NAMES)
        for i, c in enumerate(CLASS_NAMES):
            w.writerow([c] + [int(v) for v in cm[i]])
    write_json(EVAL_DIR / "confusion_matrix.json",
               {"labels": CLASS_NAMES, "rows": "true", "cols": "predicted",
                "matrix": cm.tolist()})

    with open(EVAL_DIR / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "sha256", "true_label", "pred_label", "confidence",
                    "correct", "prod_pred_label", "prod_calibrated_confidence",
                    "prod_uncertain"] + [f"p_{c}" for c in CLASS_NAMES])
        for i, row in enumerate(rows):
            w.writerow([row["path"], row["sha256"], CLASS_NAMES[y_test[i]],
                        CLASS_NAMES[y_pred[i]], f"{probs_plain[i].max():.6f}",
                        int(y_pred[i] == y_test[i]), CLASS_NAMES[prod_pred[i]],
                        f"{conf[i]:.6f}", int(not confident[i])]
                       + [f"{v:.6f}" for v in probs_plain[i]])
    with open(EVAL_DIR / "misclassified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "pred_label", "confidence"])
        for i, row in enumerate(rows):
            if y_pred[i] != y_test[i]:
                w.writerow([row["path"], CLASS_NAMES[y_test[i]],
                            CLASS_NAMES[y_pred[i]], f"{probs_plain[i].max():.4f}"])

    plot_confusion(cm)
    plot_reliability(prod_probs, prod_cal, y_test)
    try:
        plot_gradcam_examples(model, x_test, y_test, prod_pred)
    except Exception as exc:  # visual only; never block evaluation
        logger.warning(f"Grad-CAM figure skipped: {exc}")
    return metrics


def plot_confusion(cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#222", fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — held-out test set (n={int(cm.sum())})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_reliability(raw: np.ndarray, cal: np.ndarray, y: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5))
    edges = np.linspace(0, 1, 11)
    for probs, name, color in ((raw, "raw", "#adb5bd"), (cal, "calibrated", "#2a6fdb")):
        conf, pred = probs.max(1), probs.argmax(1)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean())
                ys.append((pred[m] == y[m]).mean())
        ax.plot(xs, ys, "o-", color=color,
                label=f"{name} (ECE {ece_of(probs, y):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="perfect calibration")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram — test set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "reliability_diagram.png", dpi=150)
    plt.close(fig)


def plot_gradcam_examples(model, x, y, pred, per_class: int = 2) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from app.classification.explain import grad_cam, overlay_heatmap

    rng = np.random.default_rng(RANDOM_SEED)
    picks = []
    for k in range(len(CLASS_NAMES)):
        idx = np.where(y == k)[0]
        picks += list(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    fig, axes = plt.subplots(2, len(picks) // 2 if len(picks) > 1 else 1,
                             figsize=(2.5 * len(picks) / 2, 6.4))
    axes = np.array(axes).reshape(2, -1)
    for n_i, i in enumerate(picks):
        heat, cls_idx, _ = grad_cam(model, x[i])
        over = overlay_heatmap(x[i], heat)
        ax = axes[n_i % 2, n_i // 2]
        ax.imshow(over)
        ok = "correct" if pred[i] == y[i] else "WRONG"
        ax.set_title(f"true: {CLASS_NAMES[y[i]]}\npred: {CLASS_NAMES[pred[i]]} ({ok})",
                     fontsize=7)
        ax.axis("off")
    fig.suptitle("Grad-CAM — where the model looks (test images)")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "gradcam_examples.png", dpi=140)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — safe promotion
# ══════════════════════════════════════════════════════════════════════════════

def stage_promote() -> dict:
    import keras

    banner("STAGE 4 — Safe promotion to production")
    if not TEST_LOCK.exists():
        raise RuntimeError("Refusing to promote: the candidate has not been evaluated.")
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if PROD_MODEL.exists():
        prev_sha = sha256_file(PROD_MODEL)
        backup = MODEL_DIR / f"best.keras.bak-{stamp}"
        shutil.copy2(PROD_MODEL, backup)
        assert sha256_file(backup) == prev_sha
        record.update(previous_model_sha256=prev_sha,
                      previous_model_backup=backup.relative_to(REPO_ROOT).as_posix())
        logger.info(f"Previous best.keras SHA256 {prev_sha} backed up to {backup.name}")

    model = keras.models.load_model(str(CANDIDATE_MODEL))
    out_shape = tuple(model.output_shape)
    probe = np.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.float32)
    probs = model.predict(probe, verbose=0)
    ok = out_shape[-1] == len(CLASS_NAMES) and probs.shape == (1, len(CLASS_NAMES)) \
        and abs(float(probs.sum()) - 1.0) < 1e-3
    record.update(candidate_output_shape=list(out_shape),
                  candidate_probe_sum=float(probs.sum()), verification_passed=bool(ok))
    if not ok:
        raise RuntimeError(f"Candidate verification failed: {record}")

    shutil.copy2(CANDIDATE_MODEL, PROD_MODEL)
    shutil.copy2(EVAL_DIR / "candidate_inference_config.json", INFERENCE_CONFIG_JSON)
    write_json(CLASS_ORDER_JSON, {"class_order": CLASS_NAMES})
    new_sha = sha256_file(PROD_MODEL)
    record.update(new_model_sha256=new_sha,
                  candidate_sha256=sha256_file(CANDIDATE_MODEL),
                  model_path=PROD_MODEL.relative_to(REPO_ROOT).as_posix(),
                  inference_config=INFERENCE_CONFIG_JSON.relative_to(REPO_ROOT).as_posix(),
                  class_order=CLASS_NAMES)
    with open(EVAL_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    record.update(test_accuracy=metrics["test_accuracy"], macro_f1=metrics["macro_f1"])
    write_json(EVAL_DIR / "promotion_record.json", record)
    logger.info(f"Promoted. New best.keras SHA256 {new_sha}")
    return record


# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["split", "train", "deep", "evaluate", "promote", "all"])
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--force-reeval", action="store_true")
    args = ap.parse_args()
    setup_logging()
    root = args.dataset_root.resolve()
    if args.stage in ("split", "all"):
        stage_split(root)
    if args.stage in ("train", "all"):
        stage_train(root)
    if args.stage in ("deep", "all"):
        stage_deep(root)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(root, force=args.force_reeval)
    if args.stage in ("promote", "all"):
        stage_promote()


if __name__ == "__main__":
    main()
