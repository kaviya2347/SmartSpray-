"""
FILE PATH: SmartSpray/app/classification/classifier.py

PURPOSE:
    Core inference engine for Module 3 — Disease Classification.
    Loads the trained EfficientNetB0 model exactly ONCE (lazy singleton) and
    classifies individual plant crops on demand.

    Single responsibility: receive a BGR crop array, return predictions.
    No database code, no HTTP code, no business logic.

INPUT:
    A BGR numpy array (uint8) of a single plant crop (any size; resized here).

OUTPUT (dict):
    label            predicted class (one of CLASS_NAMES)
    confidence       calibrated confidence of `label` (0..1)
    all_probs        calibrated probability for every class (sums to 1)
    raw_confidence   un-calibrated softmax confidence
    uncertain        True when confidence < uncertainty threshold (0.60)

    Calibration (temperature scaling) and horizontal-flip test-time
    augmentation are read from models/classification/inference_config.json,
    which is written by scripts/m3_train_classifier.py after they were chosen
    on the VALIDATION split. Without that file the classifier behaves exactly
    like a plain softmax model (T = 1, no TTA).

DEPENDENCIES:
    tensorflow/keras, opencv-python, numpy, app.classification.prediction
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.classification.prediction import CLASS_NAMES, MODEL_VERSION

logger = logging.getLogger("classification.classifier")

# ── Constants ──────────────────────────────────────────────────────────────────
# Paths are anchored at the repository root so the app works no matter which
# folder uvicorn / pytest is started from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "models" / "classification"
DEFAULT_WEIGHTS_PATH = MODEL_DIR / "best.keras"
CLASS_ORDER_PATH = MODEL_DIR / "class_order.json"
INFERENCE_CONFIG_NAME = "inference_config.json"

IMAGE_SIZE = 224                       # EfficientNetB0 native input size
DEFAULT_UNCERTAINTY_THRESHOLD = 0.60   # conf < 0.60 -> flagged uncertain


def _softmax_with_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale softmax outputs with a temperature (T=1 returns the input)."""
    if abs(temperature - 1.0) < 1e-9:
        return probs
    z = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class DiseaseClassifier:
    """
    Loads the EfficientNetB0 model once and exposes predict().
    Use get_classifier() to obtain the process-wide instance.
    """

    def __init__(self, weights_path: Optional[Path] = None):
        """
        Raises:
            FileNotFoundError: if best.keras or class_order.json do not exist.
            ValueError: if class_order.json does not match CLASS_NAMES.
            RuntimeError: if the model file cannot be loaded by Keras.
        """
        weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH

        # ── Class order ───────────────────────────────────────────────────────
        # Module-level lookup (not a default argument) so tests can monkeypatch it.
        class_order_path = Path(CLASS_ORDER_PATH)
        if not class_order_path.exists():
            raise FileNotFoundError(
                f"Class order file not found at '{class_order_path}'.\n"
                f"Run scripts/m3_train_classifier.py to generate it."
            )
        with open(class_order_path, "r", encoding="utf-8") as f:
            loaded_class_order = json.load(f).get("class_order", [])
        if loaded_class_order != CLASS_NAMES:
            raise ValueError(
                f"Class order mismatch!\n"
                f"  From class_order.json : {loaded_class_order}\n"
                f"  From prediction.py    : {CLASS_NAMES}\n"
                f"The model was trained with a different class ordering."
            )
        logger.info(f"Class order validated: {loaded_class_order}")

        # ── Model ─────────────────────────────────────────────────────────────
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'.\n"
                f"Run scripts/m3_train_classifier.py --stage all to create best.keras."
            )
        logger.info(f"Loading EfficientNetB0 model from: {weights_path}")
        try:
            self._model = tf.keras.models.load_model(str(weights_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{weights_path}': {e}") from e

        self._weights_path = weights_path
        self._class_order = loaded_class_order

        # ── Inference config (calibration / TTA), optional ────────────────────
        self._temperature = 1.0
        self._tta = False
        self._threshold = DEFAULT_UNCERTAINTY_THRESHOLD
        self._model_version = MODEL_VERSION
        cfg_path = weights_path.parent / INFERENCE_CONFIG_NAME
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("class_order", CLASS_NAMES) != CLASS_NAMES:
                    raise ValueError("inference_config class_order mismatch")
                self._temperature = float(cfg.get("temperature", 1.0))
                self._tta = bool(cfg.get("tta_hflip", False))
                self._threshold = float(cfg.get("uncertainty_threshold",
                                                DEFAULT_UNCERTAINTY_THRESHOLD))
                self._model_version = cfg.get("model_version", MODEL_VERSION)
                logger.info(f"Inference config: T={self._temperature:.4f} "
                            f"tta={self._tta} threshold={self._threshold}")
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Ignoring invalid {cfg_path.name}: {e}")
        logger.info("Model loaded successfully. Ready for inference.")

    # ── Inference ─────────────────────────────────────────────────────────────
    def _prepare(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Received empty crop image for classification.")
        if len(crop_bgr.shape) != 3 or crop_bgr.shape[2] != 3:
            raise ValueError(f"Expected a 3-channel BGR image, got shape {crop_bgr.shape}.")
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)          # BGR -> RGB
        if crop_rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            crop_rgb = cv2.resize(crop_rgb, (IMAGE_SIZE, IMAGE_SIZE))  # 224x224
        # Keras EfficientNet preprocess_input is a pass-through (the model has
        # its own Rescaling/Normalization layers); kept for API symmetry.
        return preprocess_input(crop_rgb.astype(np.float32))

    def predict_batch(self, crops_bgr: list) -> list:
        """Classify several crops in one forward pass. Returns a list of dicts."""
        if not crops_bgr:
            return []
        batch = np.stack([self._prepare(c) for c in crops_bgr], axis=0)
        raw = self._forward(batch)
        cal = _softmax_with_temperature(raw, self._temperature)
        results = []
        for i in range(len(crops_bgr)):
            row, raw_row = cal[min(i, len(cal) - 1)], raw[min(i, len(raw) - 1)]
            idx = int(np.argmax(row))
            confidence = round(float(row[idx]), 4)
            results.append({
                "label": self._class_order[idx],
                "confidence": confidence,
                "all_probs": {cls: round(float(row[j]), 4)
                              for j, cls in enumerate(self._class_order)},
                "raw_confidence": round(float(raw_row[idx]), 4),
                "uncertain": bool(confidence < self._threshold),
            })
        return results

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """
        Softmax probabilities for a prepared batch, averaged with the
        horizontally flipped batch when TTA is on.

        Real Keras models go through one compiled call (original + flipped in
        the same batch): same numbers as model.predict(), several times faster
        for the one-crop-at-a-time calls the simulator makes.
        """
        if isinstance(self._model, tf.keras.Model):
            if getattr(self, "_compiled", None) is None:
                model = self._model
                self._compiled = tf.function(lambda x: model(x, training=False),
                                             reduce_retracing=True)
            n = len(batch)
            x = np.concatenate([batch, batch[:, :, ::-1, :]]) if self._tta else batch
            out = np.asarray(self._compiled(tf.convert_to_tensor(x, tf.float32)),
                             dtype=np.float64)
            return (out[:n] + out[n:]) / 2.0 if self._tta else out
        probs = np.asarray(self._model.predict(batch, verbose=0), dtype=np.float64)
        if self._tta:
            flipped = np.asarray(self._model.predict(batch[:, :, ::-1, :], verbose=0),
                                 dtype=np.float64)
            probs = (probs + flipped) / 2.0
        return probs

    def predict(self, crop_bgr: np.ndarray) -> dict:
        """
        Classify a single plant crop (BGR uint8).

        Raises:
            ValueError: if the input array is empty or has an invalid shape.
        """
        logger.debug(f"Running classification on crop (shape: {getattr(crop_bgr, 'shape', None)})")
        result = self.predict_batch([crop_bgr])[0]
        logger.debug(f"  => {result['label']} (confidence: {result['confidence']:.4f})")
        return result

    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def class_order(self) -> list[str]:
        return list(self._class_order)

    @property
    def uncertainty_threshold(self) -> float:
        return self._threshold

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def tta(self) -> bool:
        return self._tta


# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: Optional[DiseaseClassifier] = None


def get_classifier() -> DiseaseClassifier:
    """Return the shared DiseaseClassifier instance, creating it if needed."""
    global _instance
    if _instance is None:
        _instance = DiseaseClassifier()
    return _instance


def reset_classifier() -> None:
    """Drop the cached instance (used after a model is re-promoted, and in tests)."""
    global _instance
    _instance = None
