"""
FILE PATH: SmartSpray/scripts/retrain_classification.py

PURPOSE:
    Full, production-grade retraining experiment for Module 3 (Disease
    Classification). Pipeline stages:

        1. NEW stratified 70/15/15 train/validation/test split from the FULL
           models/classification/prepared_dataset/ (~5016 images),
           random_state=42, class proportions preserved in every split.
           Built from the COMPLETE dataset â€” NOT from the notebook's 80/20.
        2. Zero-overlap / full-coverage leakage verification.
        3. Backup: best.keras -> best_previous.keras (once, never overwritten).
        4. COMPLETELY NEW model initialised from ImageNet weights
           (EfficientNetB0(weights="imagenet")). The OLD best.keras is NEVER
           loaded as starting weights. Architecture identical to the notebook:
               EfficientNetB0 -> GlobalAveragePooling2D -> Dropout(0.3)
               -> Dense(256, relu) -> Dropout(0.2) -> Dense(4, softmax)
        5. Phase 1 (frozen backbone, Adam 1e-3, max 20 epochs) and Phase 2
           (top-30 layers unfrozen, Adam 1e-5, max 10 epochs), each with
           early stopping + checkpointing based ONLY on validation accuracy.
        6. Model selection using ONLY validation performance. The test set is
           NOT touched during selection.
        7. Independent TEST evaluation of the selected model.
        8. NEW-model validation metrics (computed AFTER selection).
        9. Promotion (best_retrained.keras -> best.keras) ONLY after all
           gates pass. Old model stays at best_previous.keras.
       10. evaluation_summary.txt separating OLD baseline from NEW numbers.

USAGE (repo root, inside .venv):
    python scripts/retrain_classification.py --prepare-only
    python scripts/retrain_classification.py [--no-promote]

OUTPUTS (models/classification/evaluation/retrained/):
    train_split.csv, validation_split.csv, test_split.csv,
    split_verification.json, training_history.csv,
    validation_metrics.json, test_metrics.json,
    test_classification_report.csv, test_confusion_matrix.png,
    test_misclassified.csv, evaluation_summary.txt

DEPENDENCIES:
    tensorflow, numpy, pandas, scikit-learn, matplotlib, seaborn
"""

import argparse
import json
import logging
import os
import random
import shutil
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

# Silence TensorFlow info/warning logs before importing it (same as notebook).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications import EfficientNetB0  # noqa: E402
from tensorflow.keras.applications.efficientnet import (  # noqa: E402
    preprocess_input,
)
from tensorflow.keras.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (  # noqa: E402
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "classification"
PREPARED_DIR = MODEL_DIR / "prepared_dataset"
CLASS_ORDER_PATH = MODEL_DIR / "class_order.json"
EVAL_DIR = MODEL_DIR / "evaluation"
RETRAINED_DIR = EVAL_DIR / "retrained"

PROD_MODEL_PATH = MODEL_DIR / "best.keras"                # production model
BACKUP_MODEL_PATH = MODEL_DIR / "best_previous.keras"     # backup of OLD model
RETRAINED_MODEL_PATH = MODEL_DIR / "best_retrained.keras"  # NEW model

CKPT_PHASE1_PATH = RETRAINED_DIR / "checkpoint_phase1.keras"
CKPT_PHASE2_PATH = RETRAINED_DIR / "checkpoint_phase2.keras"

# â”€â”€ Configuration (matches the Module 3 training notebook exactly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
IMAGE_SIZE = 224           # EfficientNetB0 native input size
BATCH_SIZE = 32
EPOCHS_HEAD = 20           # Phase 1 maximum epochs (backbone frozen)
EPOCHS_FINETUNE = 10       # Phase 2 maximum epochs (top 30 layers unfrozen)
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_LAYERS = 30      # top backbone layers unfrozen in Phase 2
DROPOUT_RATE = 0.3
TEST_FRACTION = 0.15       # of the full dataset
VAL_FRACTION = 0.15        # of the full dataset
EARLY_STOP_PATIENCE = 5    # epochs without val_accuracy improvement before stop
REDUCE_LR_PATIENCE = 3     # epochs without val_loss improvement before LR halving
VERBOSE = 1                # Keras fit verbosity (1 = progress bar per epoch)

CLASS_NAMES: list[str] = [
    "early_blight",
    "healthy",
    "late_blight",
    "septoria_leaf_spot",
]  # alphabetical => matches Keras index order and class_order.json

NUM_CLASSES = len(CLASS_NAMES)

# Canonical class order, loaded from class_order.json at runtime by main().
# A module-level global keeps every stage referencing the same ordering.
CLASS_ORDER: list[str] = []

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logger = logging.getLogger("retrain_classification")


def setup_logging() -> None:
    """Configure root logging to console with readable timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seeds() -> None:
    """Seed every random source used by the pipeline for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    logger.info(f"Seeds set to {SEED} (random / numpy / tensorflow)")


def load_class_order() -> list[str]:
    """Load and validate the canonical class order from class_order.json."""
    with open(CLASS_ORDER_PATH, "r") as f:
        data = json.load(f)
    class_order = data.get("class_order", [])
    if class_order != CLASS_NAMES:
        raise ValueError(
            f"class_order.json {class_order} does not match canonical "
            f"{CLASS_NAMES}. Refusing to continue â€” predictions would be "
            f"silently mislabelled."
        )
    return class_order


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 1 â€” Dataset manifest + NEW stratified 70/15/15 split
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_manifest() -> pd.DataFrame:
    """
    Scan the FULL prepared_dataset directory and build one row per image.

    Only the four canonical class folders are used. Filepaths are stored
    relative to PREPARED_DIR with forward slashes so they are portable and
    directly consumable by flow_from_dataframe(directory=PREPARED_DIR).

    Returns:
        DataFrame with columns: filepath, filename, label.
    """
    rows: list[dict] = []
    for label in CLASS_NAMES:
        class_dir = PREPARED_DIR / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "filepath": path.relative_to(PREPARED_DIR).as_posix(),
                        "filename": path.name,
                        "label": label,
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(f"No images found under {PREPARED_DIR}")
    logger.info(
        f"Manifest built: {len(manifest)} images across {len(CLASS_NAMES)} classes"
    )
    return manifest


def create_splits(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create the NEW stratified 70/15/15 split from the COMPLETE dataset.

    Method (both steps use random_state=42):
        1. train_test_split(manifest, test_size=0.30, stratify=label)
           -> 70% train + 30% holdout.
        2. train_test_split(holdout, test_size=0.50, stratify=label)
           -> 15% validation + 15% test.

    Stratification preserves class proportions in every split.
    """
    train_df, holdout_df = train_test_split(
        manifest,
        test_size=(TEST_FRACTION + VAL_FRACTION),  # 0.30
        stratify=manifest["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,  # half of the 30% holdout -> 15% / 15%
        stratify=holdout_df["label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        f"Split created: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )
    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict:
    """
    Verify the split is leak-free and complete.

    Checks: trainâˆ©val = 0, trainâˆ©test = 0, valâˆ©test = 0, and the union of the
    three splits equals the full manifest (every image in exactly one split).

    Returns:
        JSON-serialisable verification report dict.

    Raises:
        RuntimeError: if any overlap or coverage check fails.
    """
    train_set = set(train_df["filepath"])
    val_set = set(val_df["filepath"])
    test_set = set(test_df["filepath"])
    manifest_set = set(manifest["filepath"])

    ov_tv = train_set & val_set
    ov_tt = train_set & test_set
    ov_vt = val_set & test_set
    union = train_set | val_set | test_set
    missing = manifest_set - union
    extra = union - manifest_set

    report = {
        "seed": SEED,
        "split_method": (
            "sklearn train_test_split x2, stratified by class label, "
            "random_state=42: 70/30 then holdout 50/50"
        ),
        "n_dataset": len(manifest_set),
        "n_train": len(train_set),
        "n_validation": len(val_set),
        "n_test": len(test_set),
        "overlap_train_validation": len(ov_tv),
        "overlap_train_test": len(ov_tt),
        "overlap_validation_test": len(ov_vt),
        "missing_from_splits": len(missing),
        "extra_in_splits": len(extra),
        "leak_free": len(ov_tv) == 0 and len(ov_tt) == 0 and len(ov_vt) == 0,
        "full_coverage": len(missing) == 0 and len(extra) == 0,
        "class_distribution": {
            split: {label: int((df["label"] == label).sum()) for label in CLASS_NAMES}
            for split, df in
            (("train", train_df), ("validation", val_df), ("test", test_df))
        },
    }

    if not report["leak_free"]:
        raise RuntimeError(
            f"Leakage detected! trainâˆ©val={len(ov_tv)} trainâˆ©test={len(ov_tt)} "
            f"valâˆ©test={len(ov_vt)} â€” refusing to continue."
        )
    if not report["full_coverage"]:
        raise RuntimeError(
            f"Coverage failure: missing={len(missing)} extra={len(extra)} â€” "
            f"refusing to continue."
        )
    logger.info(
        "Leakage verification PASSED (zero overlap, full coverage, "
        f"{report['n_train']}/{report['n_validation']}/{report['n_test']})"
    )
    return report


def save_split_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verification: dict,
) -> None:
    """Persist the three split manifests + verification report as CSV/JSON."""
    train_df.to_csv(RETRAINED_DIR / "train_split.csv", index=False)
    val_df.to_csv(RETRAINED_DIR / "validation_split.csv", index=False)
    test_df.to_csv(RETRAINED_DIR / "test_split.csv", index=False)
    (RETRAINED_DIR / "split_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    logger.info(f"Split manifests + verification saved to {RETRAINED_DIR}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 2 â€” Production model backup
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def backup_production_model() -> None:
    """
    Copy best.keras -> best_previous.keras exactly once.

    If the backup already exists it is LEFT UNTOUCHED, so repeated runs can
    never overwrite the original old model with a promoted new one.
    """
    if not PROD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at '{PROD_MODEL_PATH}' â€” nothing to back up."
        )
    if BACKUP_MODEL_PATH.exists():
        logger.info(f"Backup already exists, leaving untouched: {BACKUP_MODEL_PATH}")
        return
    shutil.copy2(PROD_MODEL_PATH, BACKUP_MODEL_PATH)
    size_mb = BACKUP_MODEL_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Backed up OLD best.keras -> best_previous.keras ({size_mb:.1f} MB)")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• UNCHANGED â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 3 â€” Data generators
#
# Preprocessing and training augmentation are EXACTLY the same as the original
# notebook (m3_classification_trials.ipynb) and the previous evaluation script:
#   - target_size 224x224
#   - preprocessing_function = efficientnet.preprocess_input
#   - training augmentation: rotation 20, shifts 0.15, hflip, zoom 0.15,
#     brightness 0.8-1.2, fill_mode nearest
#   - validation/test: preprocessing ONLY (no augmentation)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def make_train_generator(df: pd.DataFrame) -> "ImageDataGenerator flow":
    """
    Training generator: EfficientNet preprocessing + the SAME augmentation as
    the original notebook, applied only to the 70% train split rows.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(PREPARED_DIR),
        x_col="filepath",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,  # fixes index order: early_blight=0, healthy=1, ...
        seed=SEED,
        shuffle=True,
    )


def make_eval_generator(df: pd.DataFrame, shuffle: bool = False):
    """
    Validation/test generator: EfficientNet preprocessing ONLY â€” no
    augmentation, so evaluation measures realistic input performance.
    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(PREPARED_DIR),
        x_col="filepath",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=shuffle,
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 4 â€” NEW model (ImageNet initialisation â€” NOT the old best.keras)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_new_model() -> tuple["tf.keras.Model", "tf.keras.Model"]:
    """
    Build a COMPLETELY NEW EfficientNetB0 classifier.

    CRITICAL GUARANTEE:
        The backbone is initialised from ImageNet weights
        (EfficientNetB0(weights="imagenet")). The old production model
        (models/classification/best.keras) is NEVER loaded here â€” not as
        starting weights, not for any other purpose.

    Architecture (identical to the original notebook / Module 3 design):
        EfficientNetB0(224x224x3, ImageNet, include_top=False)
        -> GlobalAveragePooling2D
        -> Dropout(0.3)
        -> Dense(256, relu)
        -> Dropout(0.2)
        -> Dense(4, softmax)

    Returns:
        (model, base_model) â€” base_model is returned so Phase 2 can unfreeze
        its top layers.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    base_model.trainable = False  # frozen backbone for Phase 1

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)  # keeps BatchNorm layers frozen
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(DROPOUT_RATE * 0.67)(x)  # 0.2 when dropout=0.3
    outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    n_total = model.count_params()
    n_train = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    logger.info(
        f"NEW model built from ImageNet weights: {n_total:,} params "
        f"({n_train:,} trainable in Phase 1)"
    )
    return model, base_model


def make_callbacks(checkpoint_path: Path) -> list:
    """
    Callbacks for both phases. Checkpoint selection and early stopping monitor
    VALIDATION accuracy ONLY â€” the test set is never touched during training.

    Args:
        checkpoint_path: Where ModelCheckpoint saves the best-so-far weights.
                         Phase 1 and Phase 2 use separate files so each phase's
                         best model can be compared afterwards.
    """
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_phase1(model, train_gen, val_gen) -> "tf.keras.callbacks.History":
    """
    Phase 1 â€” train the new Dense head only (backbone frozen).
    LR 1e-3, Adam, categorical crossentropy, max 20 epochs,
    early stopping + checkpointing on val_accuracy only.
    """
    logger.info("=" * 68)
    logger.info("PHASE 1 â€” training head (EfficientNetB0 backbone frozen), lr=1e-3")
    logger.info("=" * 68)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=make_callbacks(CKPT_PHASE1_PATH),
        verbose=VERBOSE,
    )
    best_val_acc = max(history.history["val_accuracy"])
    logger.info(f"Phase 1 complete. Best val_accuracy = {best_val_acc:.4f}")
    return history


def train_phase2(model, base_model, train_gen, val_gen) -> "tf.keras.callbacks.History":
    """
    Phase 2 â€” fine-tune the top 30 backbone layers at lr=1e-5.
    Same validation-only monitoring. Returns the Phase 2 history.
    """
    logger.info("=" * 68)
    logger.info("PHASE 2 â€” fine-tuning top 30 backbone layers, lr=1e-5")
    logger.info("=" * 68)
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=make_callbacks(CKPT_PHASE2_PATH),
        verbose=VERBOSE,
    )
    best_val_acc = max(history.history["val_accuracy"])
    logger.info(f"Phase 2 complete. Best val_accuracy = {best_val_acc:.4f}")
    return history


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 5 â€” Evaluation utilities (validation for selection, test ONLY at the end)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def collect_predictions(model, generator) -> dict:
    """
    Run the model over an entire generator (shuffle=False) and gather everything
    needed for metric computation.

    Returns:
        {
            "y_true":    np.ndarray[int]   â€” true class index per image,
            "y_pred":    np.ndarray[int]   â€” predicted class index,
            "y_prob":    np.ndarray[float] â€” (N, 4) softmax probability matrix,
            "filepaths": list[str]         â€” generator order of image paths,
        }
    """
    generator.reset()
    n = generator.samples
    steps = int(np.ceil(n / generator.batch_size))
    probs = model.predict(generator, steps=steps, verbose=VERBOSE)
    probs = probs[:n]
    # generator.classes can be a plain Python list for categorical
    # flow_from_dataframe generators -> coerce to an int numpy array so
    # every downstream numpy operation (==, .sum(), fancy indexing) is safe.
    y_true = np.asarray(generator.classes[:n], dtype=int)
    y_pred = np.argmax(probs, axis=1)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs,
        "filepaths": list(generator.filenames[:n]),
    }


def compute_metrics(preds: dict) -> dict:
    """
    Compute the full metric suite from collected predictions:
    accuracy, macro/weighted precision-recall-F1, per-class metrics,
    confusion matrix, correct/incorrect counts, and confidence statistics.

    Returns:
        JSON-serialisable metrics dict.
    """
    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]
    # Defensive coercion: y_true may arrive as a list from DataFrameIterator.
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidences = y_prob[np.arange(len(y_pred)), y_pred]
    correct_mask = y_pred == y_true

    per_class = {}
    for idx, label in enumerate(CLASS_NAMES):
        per_class[label] = {
            "precision": round(float(precision_score(
                y_true, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(
                y_true, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "f1": round(float(f1_score(
                y_true, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "support": int((y_true == idx).sum()),
        }

    return {
        "n_images": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "correct": int(correct_mask.sum()),
        "incorrect": int((~correct_mask).sum()),
        "confidence_mean_correct": round(
            float(confidences[correct_mask].mean()), 4) if correct_mask.any() else None,
        "confidence_mean_incorrect": round(
            float(confidences[~correct_mask].mean()), 4) if (~correct_mask).any() else None,
        "confidence_mean_overall": round(float(confidences.mean()), 4),
    }


def save_eval_artifacts(preds: dict, metrics: dict, split_name: str) -> None:
    """
    Persist one split's evaluation artifacts:
        <split>_metrics.json  â€” always
    For the test split additionally:
        test_classification_report.csv, test_confusion_matrix.png,
        test_misclassified.csv
    (matching the required NEW-artifact list; validation gets only the JSON).
    """
    (RETRAINED_DIR / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info(f"{split_name} metrics saved -> {split_name}_metrics.json")

    if split_name != "test":
        return

    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]

    # Per-image classification report CSV
    rows = [
        {
            "class": label,
            "precision": metrics["per_class"][label]["precision"],
            "recall": metrics["per_class"][label]["recall"],
            "f1": metrics["per_class"][label]["f1"],
            "support": metrics["per_class"][label]["support"],
        }
        for label in CLASS_NAMES
    ]
    pd.DataFrame(rows).to_csv(
        RETRAINED_DIR / "test_classification_report.csv", index=False
    )

    # Confusion matrix PNG (counts annotated, class names on both axes)
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"NEW model - Test confusion matrix (acc={metrics['accuracy']:.4f}, "
        f"n={metrics['n_images']})"
    )
    plt.tight_layout()
    fig.savefig(RETRAINED_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Misclassified images CSV
    mis_rows = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            mis_rows.append({
                "filepath": preds["filepaths"][i],
                "true_label": CLASS_NAMES[y_true[i]],
                "predicted_label": CLASS_NAMES[y_pred[i]],
                "confidence": round(float(y_prob[i, y_pred[i]]), 4),
                "prob_true": round(float(y_prob[i, y_true[i]]), 4),
            })
    pd.DataFrame(mis_rows).to_csv(
        RETRAINED_DIR / "test_misclassified.csv", index=False
    )
    logger.info(f"Test artifacts saved (misclassified: {len(mis_rows)} images)")


def save_training_history(history_p1, history_p2) -> None:
    """
    Concatenate both phases' Keras histories into one CSV with a 'phase'
    column so the whole run is auditable epoch by epoch.
    """
    rows = []
    for phase, hist in (("phase1_frozen", history_p1), ("phase2_finetune", history_p2)):
        if hist is None:
            continue
        n = len(hist.history["loss"])
        for i in range(n):
            row = {"phase": phase, "epoch": i + 1}
            for key, values in hist.history.items():
                row[key] = round(float(values[i]), 6)
            rows.append(row)
    pd.DataFrame(rows).to_csv(RETRAINED_DIR / "training_history.csv", index=False)
    logger.info(f"Training history saved ({len(rows)} epoch rows)")


def promote_model(promote: bool) -> None:
    """
    Copy best_retrained.keras -> best.keras when promote=True.
    The old production model is already safely stored at best_previous.keras
    (backup_production_model never overwrites an existing backup).
    """
    if not promote:
        logger.info("Promotion skipped (--no-promote). best.keras left unchanged.")
        return
    if not RETRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{RETRAINED_MODEL_PATH} not found - nothing to promote."
        )
    shutil.copy2(RETRAINED_MODEL_PATH, PROD_MODEL_PATH)
    logger.info(
        f"PROMOTED: {RETRAINED_MODEL_PATH.name} -> best.keras "
        f"(old model safe at {BACKUP_MODEL_PATH.name})"
    )


def write_summary(
    verification: dict,
    val_metrics: dict,
    test_metrics: dict,
    selected_phase: str,
    promote: bool,
) -> None:
    """
    Write evaluation_summary.txt clearly separating the OLD baseline numbers
    (from evaluation/validation_metrics.json, kept unchanged) from the NEW
    retrained-model numbers.
    """
    old_path = EVAL_DIR / "validation_metrics.json"
    old_block = (
        "OLD MODEL BASELINE (historical, notebook 80/20 split)\n"
        "-----------------------------------------------------\n"
        "  NOTE: the old number is a VALIDATION number. The notebook's 500-image\n"
        "  'test' subset overlapped its validation split (scripts/analyze_splits.py),\n"
        "  so it was NEVER an independent test accuracy.\n"
    )
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_block += (
            f"  old validation accuracy : {old.get('accuracy')}\n"
            f"  old f1 macro            : {old.get('f1_macro')}\n"
            f"  old n_images            : {old.get('n_images')}\n"
        )
    else:
        old_block += "  (old evaluation artifacts not found)\n"

    ft_helped = (
        "yes - Phase 2 checkpoint selected"
        if selected_phase == "phase2"
        else "no - Phase 1 checkpoint selected (fine-tuning did not improve val)"
    )
    cd = verification["class_distribution"]
    class_lines = "".join(
        f"  {c:22s} train={cd['train'][c]:5d} val={cd['validation'][c]:4d} "
        f"test={cd['test'][c]:4d}\n"
        for c in CLASS_NAMES
    )
    header = (
        f"SMARTSPRAY MODULE 3 - NEW 70/15/15 RETRAINING SUMMARY\n"
        f"=======================================================\n"
        f"Generated: {datetime.now().isoformat()}\n\n"
        f"SPLIT (NEW, stratified, random_state={SEED})\n"
        f"----------------------------------------\n"
        f"{verification['split_method']}\n"
        f"  dataset : {verification['n_dataset']} images\n"
        f"  train   : {verification['n_train']}\n"
        f"  val     : {verification['n_validation']}\n"
        f"  test    : {verification['n_test']}\n"
        f"  overlap train/val : {verification['overlap_train_validation']}\n"
        f"  overlap train/test: {verification['overlap_train_test']}\n"
        f"  overlap val/test  : {verification['overlap_validation_test']}\n"
        f"  coverage: missing={verification['missing_from_splits']} "
        f"extra={verification['extra_in_splits']}\n"
        f"  leak-free + full coverage: "
        f"{verification['leak_free'] and verification['full_coverage']}\n\n"
        f"Class distribution\n"
        f"------------------\n"
        f"{class_lines}\n"
        f"MODEL (NEW EXPERIMENT)\n"
        f"----------------------\n"
        f"  Init   : EfficientNetB0(weights='imagenet') - NOT the old best.keras\n"
        f"  Arch   : EfficientNetB0(224x224x3) -> GAP -> Dropout(0.3)\n"
        f"           -> Dense(256,relu) -> Dropout(0.2) -> Dense(4,softmax)\n"
        f"  Phase 1: backbone frozen, Adam 1e-3, max {EPOCHS_HEAD} epochs, ES on val_accuracy\n"
        f"  Phase 2: top {FINE_TUNE_LAYERS} layers unfrozen, Adam 1e-5, max {EPOCHS_FINETUNE} epochs\n"
        f"  Selection: best validation checkpoint -> {selected_phase}\n"
        f"  Did fine-tuning help?: {ft_helped}\n"
    )
    (RETRAINED_DIR / "evaluation_summary.txt").write_text(header, encoding="utf-8")

    per_class_block = "".join(
        f"  {label:22s} P={test_metrics['per_class'][label]['precision']:.4f} "
        f"R={test_metrics['per_class'][label]['recall']:.4f} "
        f"F1={test_metrics['per_class'][label]['f1']:.4f} "
        f"n={test_metrics['per_class'][label]['support']}\n"
        for label in CLASS_NAMES
    )
    cm_block = "".join(
        f"  {label:22s} {test_metrics['confusion_matrix'][i]}\n"
        for i, label in enumerate(CLASS_NAMES)
    )
    metrics_block = (
        f"\nNEW MODEL - VALIDATION (used for selection)\n"
        f"--------------------------------------------\n"
        f"  accuracy      : {val_metrics['accuracy']}\n"
        f"  f1 macro      : {val_metrics['f1_macro']}\n"
        f"  f1 weighted   : {val_metrics['f1_weighted']}\n"
        f"  correct/wrong : {val_metrics['correct']}/{val_metrics['incorrect']} "
        f"of {val_metrics['n_images']}\n"
        f"\nNEW MODEL - INDEPENDENT TEST (untouched until after selection)\n"
        f"---------------------------------------------------------------\n"
        f"  TEST ACCURACY : {test_metrics['accuracy']}\n"
        f"  f1 macro      : {test_metrics['f1_macro']}\n"
        f"  f1 weighted   : {test_metrics['f1_weighted']}\n"
        f"  correct/wrong : {test_metrics['correct']}/{test_metrics['incorrect']} "
        f"of {test_metrics['n_images']}\n"
        f"  conf mean correct  : {test_metrics['confidence_mean_correct']}\n"
        f"  conf mean incorrect: {test_metrics['confidence_mean_incorrect']}\n"
        f"\nPer-class test metrics\n"
        f"----------------------\n"
        f"{per_class_block}\n"
        f"Confusion matrix (rows=true, cols=pred; order: {CLASS_NAMES})\n"
        f"{cm_block}\n"
        f"PROMOTION\n"
        f"---------\n"
        f"  best_retrained.keras saved: always\n"
        f"  promoted to best.keras    : {'yes' if promote else 'no (--no-promote)'}\n"
        f"  old model preserved at    : best_previous.keras\n"
        f"\n{old_block}"
    )
    with open(RETRAINED_DIR / "evaluation_summary.txt", "a", encoding="utf-8") as f:
        f.write(metrics_block)
    logger.info("Summary written -> evaluation_summary.txt")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 6 â€” Pipeline orchestration
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_prepare_only() -> None:
    """
    Stage the split WITHOUT training: manifest -> stratified 70/15/15 ->
    leakage verification -> manifests saved -> production model backed up.
    """
    RETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    load_class_order()
    manifest = build_manifest()
    train_df, val_df, test_df = create_splits(manifest)
    verification = verify_no_leakage(train_df, val_df, test_df, manifest)
    save_split_artifacts(train_df, val_df, test_df, verification)
    backup_production_model()
    logger.info("PREPARE-ONLY complete. Ready to train.")


def run_full(promote: bool) -> None:
    """
    Full experiment: split + backup + NEW ImageNet-init model + Phase 1 +
    Phase 2 + validation-only selection + independent test evaluation +
    artifacts + (optional) promotion.
    """
    RETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    load_class_order()
    set_seeds()

    # â”€â”€ Split, verify, save, backup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    manifest = build_manifest()
    train_df, val_df, test_df = create_splits(manifest)
    verification = verify_no_leakage(train_df, val_df, test_df, manifest)
    save_split_artifacts(train_df, val_df, test_df, verification)
    backup_production_model()

    # â”€â”€ Generators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Building data generators...")
    train_gen = make_train_generator(train_df)
    val_gen_eval = make_eval_generator(val_df)   # no augmentation, shuffle=False

    # â”€â”€ NEW model from ImageNet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model, base_model = build_new_model()

    # â”€â”€ Phase 1 then Phase 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    history_p1 = train_phase1(model, train_gen, val_gen_eval)
    history_p2 = train_phase2(model, base_model, train_gen, val_gen_eval)
    save_training_history(history_p1, history_p2)

    # â”€â”€ Selection on VALIDATION only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Selecting the best checkpoint using VALIDATION performance only...")
    candidates = []
    for path, phase in ((CKPT_PHASE1_PATH, "phase1"), (CKPT_PHASE2_PATH, "phase2")):
        if path.exists():
            cand_model = tf.keras.models.load_model(str(path))
            cand_preds = collect_predictions(cand_model, val_gen_eval)
            cand_metrics = compute_metrics(cand_preds)
            candidates.append((phase, path, cand_metrics["accuracy"],
                               cand_metrics["f1_macro"], cand_metrics))
            logger.info(
                f"  {phase}: val_accuracy={cand_metrics['accuracy']:.4f} "
                f"val_f1_macro={cand_metrics['f1_macro']:.4f}"
            )
    if not candidates:
        raise RuntimeError("No phase checkpoints found - cannot select a model.")

    candidates.sort(key=lambda c: (c[2], c[3]), reverse=True)
    selected_phase, selected_path, _, _, _ = candidates[0]
    logger.info(f"SELECTED: {selected_phase} checkpoint ({selected_path.name})")

    # Freeze the final model: copy the winner to best_retrained.keras
    shutil.copy2(selected_path, RETRAINED_MODEL_PATH)
    logger.info(f"Final NEW model saved -> {RETRAINED_MODEL_PATH.name}")

    # â”€â”€ Final metrics of the SELECTED model (validation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    final_model = tf.keras.models.load_model(str(RETRAINED_MODEL_PATH))
    val_preds = collect_predictions(final_model, val_gen_eval)
    val_metrics = compute_metrics(val_preds)
    save_eval_artifacts(val_preds, val_metrics, "validation")

    # â”€â”€ Independent TEST evaluation (first and only test-set use) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("=" * 68)
    logger.info("TEST EVALUATION - test set is now touched for the first time")
    logger.info("=" * 68)
    test_gen = make_eval_generator(test_df)
    test_preds = collect_predictions(final_model, test_gen)
    test_metrics = compute_metrics(test_preds)
    save_eval_artifacts(test_preds, test_metrics, "test")

    # â”€â”€ Summary + promotion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    write_summary(verification, val_metrics, test_metrics,
                  selected_phase, promote)
    promote_model(promote)

    logger.info("=" * 68)
    logger.info("RETRAINING COMPLETE")
    logger.info(f"  val_accuracy={val_metrics['accuracy']:.4f} "
                f"TEST_ACCURACY={test_metrics['accuracy']:.4f} "
                f"test_f1_macro={test_metrics['f1_macro']:.4f}")
    logger.info(f"  Artifacts: {RETRAINED_DIR}")
    logger.info("=" * 68)


def run_resume_eval(promote: bool) -> None:
    """
    Resume AFTER training from existing checkpoints only.

    Intended for when Phase 1/2 training already completed but the script
    crashed or was interrupted before selection+testing. Skips the split,
    backup, generator-building and training stages entirely — it just:

        1. Loads the persisted split manifests + verification report.
        2. Loads both phase checkpoints and recomputes VALIDATION metrics.
        3. Selects the winner using validation performance ONLY.
        4. Copies the winner to best_retrained.keras.
        5. Evaluates the final model on the untouched TEST split.
        6. Writes all validation/test artifacts + evaluation_summary.txt.
        7. Optionally promotes best_retrained.keras -> best.keras.

    This never touches best.keras during training (already trained), and
    the test set is only used after selection.
    """
    RETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    load_class_order()
    set_seeds()

    logger.info("Resuming evaluation from existing checkpoints...")

    # ── Load the persisted (verified) split manifests ─────────────────────────
    required = ("train_split.csv", "validation_split.csv", "test_split.csv",
                "split_verification.json")
    missing = [name for name in required
               if not (RETRAINED_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot resume: missing {missing} in {RETRAINED_DIR}."
        )
    train_df = pd.read_csv(RETRAINED_DIR / "train_split.csv")
    val_df = pd.read_csv(RETRAINED_DIR / "validation_split.csv")
    test_df = pd.read_csv(RETRAINED_DIR / "test_split.csv")
    verification = json.loads(
        (RETRAINED_DIR / "split_verification.json").read_text(encoding="utf-8")
    )

    if not any((CKPT_PHASE1_PATH.exists(), CKPT_PHASE2_PATH.exists())):
        raise FileNotFoundError(
            f"No phase checkpoints found under {RETRAINED_DIR} - cannot resume."
        )

    # ── Eval generators (validation used for selection) ───────────────────────
    val_gen_eval = make_eval_generator(val_df)
    test_gen = make_eval_generator(test_df)

    logger.info("Selecting the best checkpoint using VALIDATION performance only...")
    candidates = []
    for path, phase in ((CKPT_PHASE1_PATH, "phase1"), (CKPT_PHASE2_PATH, "phase2")):
        if path.exists():
            cand_model = tf.keras.models.load_model(str(path))
            cand_preds = collect_predictions(cand_model, val_gen_eval)
            cand_metrics = compute_metrics(cand_preds)
            candidates.append((phase, path, cand_metrics["accuracy"],
                               cand_metrics["f1_macro"], cand_metrics))
            logger.info(
                f"  {phase}: val_accuracy={cand_metrics['accuracy']:.4f} "
                f"val_f1_macro={cand_metrics['f1_macro']:.4f}"
            )
    candidates.sort(key=lambda c: (c[2], c[3]), reverse=True)
    selected_phase, selected_path, _, _, _ = candidates[0]
    logger.info(f"SELECTED: {selected_phase} checkpoint ({selected_path.name})")

    # Freeze the final model: copy the winner to best_retrained.keras
    shutil.copy2(selected_path, RETRAINED_MODEL_PATH)
    logger.info(f"Final NEW model saved -> {RETRAINED_MODEL_PATH.name}")

    # ── Final metrics of the SELECTED model (validation) ──────────────────────
    final_model = tf.keras.models.load_model(str(RETRAINED_MODEL_PATH))
    val_preds = collect_predictions(final_model, val_gen_eval)
    val_metrics = compute_metrics(val_preds)
    save_eval_artifacts(val_preds, val_metrics, "validation")

    # ── Independent TEST evaluation (first and only test-set use) ─────────────
    logger.info("=" * 68)
    logger.info("TEST EVALUATION - test set is now touched for the first time")
    logger.info("=" * 68)
    test_preds = collect_predictions(final_model, test_gen)
    test_metrics = compute_metrics(test_preds)
    save_eval_artifacts(test_preds, test_metrics, "test")

    write_summary(verification, val_metrics, test_metrics,
                  selected_phase, promote)
    promote_model(promote)

    logger.info("=" * 68)
    logger.info("RESUME EVAL COMPLETE")
    logger.info(f"  val_accuracy   = {val_metrics['accuracy']:.4f} "
                f"val_f1_macro   = {val_metrics['f1_macro']:.4f}")
    logger.info(f"  TEST_ACCURACY  = {test_metrics['accuracy']:.4f} "
                f"test_f1_macro  = {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Selected phase = {selected_phase}")
    logger.info(f"  Artifacts      = {RETRAINED_DIR}")
    logger.info("=" * 68)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Module 3 retraining: NEW 70/15/15 split + NEW ImageNet-init model"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the split manifests and back up best.keras, then exit.",
    )
    parser.add_argument(
        "--resume-eval",
        action="store_true",
        help="Skip training; run selection + test evaluation from existing "
             "checkpoints in evaluation/retrained/.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not copy best_retrained.keras over best.keras at the end.",
    )
    args = parser.parse_args()

    if args.prepare_only:
        run_prepare_only()
    elif args.resume_eval:
        run_resume_eval(promote=not args.no_promote)
    else:
        run_full(promote=not args.no_promote)


if __name__ == "__main__":
    main()
