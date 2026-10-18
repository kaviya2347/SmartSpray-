"""
FILE PATH: SmartSpray/app/classification/model_factory.py

PURPOSE:
    Single source of truth for the Module 3 network architecture, shared by
    the training script (scripts/m3_train_classifier.py) and the tests.

    Architecture (unchanged from the original Module 3 design):

        EfficientNetB0 (ImageNet backbone, include_top=False)
          -> GlobalAveragePooling2D
          -> Dropout(0.3)
          -> Dense(256, ReLU)
          -> Dropout(0.2)
          -> Dense(4, Softmax)

    Input: RGB uint8/float32 images of shape (224, 224, 3) in the 0..255
    range. Keras' EfficientNetB0 contains its own Rescaling + Normalization
    layers, so `efficientnet.preprocess_input` is a pass-through (it does NOT
    scale to [-1, 1]).

    No TensorFlow import happens at module import time of the app, because
    this file is only imported by training code and tests.
"""

from pathlib import Path
from typing import Optional

IMAGE_SIZE = 224
NUM_CLASSES = 4
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
BACKBONE_NAME = "efficientnetb0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRETRAINED_BACKBONE_WEIGHTS = (
    REPO_ROOT / "models" / "pretrained" / "efficientnetb0_imagenet_notop.weights.h5"
)
OFFICIAL_KERAS_NOTOP_H5 = REPO_ROOT / "models" / "pretrained" / "efficientnetb0_notop.h5"


def build_backbone(weights_path: Optional[Path] = None):
    """
    Build an EfficientNetB0 backbone initialised with ImageNet weights.

    Resolution order for the ImageNet weights:
      1. `weights_path` if given
      2. models/pretrained/efficientnetb0_imagenet_notop.weights.h5
         (ImageNet backbone recovered from the legacy model - see
         models/pretrained/README.md for the verification record)
      3. models/pretrained/efficientnetb0_notop.h5 (official Keras file)
      4. weights="imagenet" (Keras downloads it; needs internet)
    """
    import keras

    candidates = [p for p in (weights_path, PRETRAINED_BACKBONE_WEIGHTS) if p]
    for path in candidates:
        if Path(path).exists():
            base = keras.applications.EfficientNetB0(
                weights=None, include_top=False,
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            )
            base.load_weights(str(path))
            return base, str(path)

    if OFFICIAL_KERAS_NOTOP_H5.exists():
        base = keras.applications.EfficientNetB0(
            weights=str(OFFICIAL_KERAS_NOTOP_H5), include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
        return base, str(OFFICIAL_KERAS_NOTOP_H5)

    base = keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    return base, "keras-download:imagenet"


def build_model(num_classes: int = NUM_CLASSES,
                dropout: float = DROPOUT_RATE,
                backbone_weights: Optional[Path] = None,
                seed: int = 42):
    """
    Build the full classifier. Returns (model, base_model, weights_source).
    The backbone starts frozen (Phase 1 of training).
    """
    import keras
    from keras import layers

    base, source = build_backbone(backbone_weights)
    base.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    # training=False keeps every BatchNormalization layer in inference mode,
    # in both training phases.
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, seed=seed, name="dropout_head_1")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_256")(x)
    x = layers.Dropout(round(dropout * 0.67, 2), seed=seed, name="dropout_head_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="smartspray_m3_efficientnetb0")
    return model, base, source


def unfreeze_top_layers(base, n_layers: int = 30, keep_bn_frozen: bool = True) -> int:
    """
    Phase 2: unfreeze the last `n_layers` backbone layers. BatchNormalization
    layers stay frozen (standard Keras transfer-learning practice) so their
    ImageNet statistics are not disturbed by small batches.
    Returns the number of layers that became trainable.
    """
    import keras

    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    count = 0
    for layer in base.layers[-n_layers:]:
        if keep_bn_frozen and isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            if layer.weights:
                count += 1
    return count


def head_layers(model) -> list:
    """Return the layers that come after the backbone (used by Grad-CAM)."""
    names = ["gap", "dropout_head_1", "dense_256", "dropout_head_2", "predictions"]
    found = []
    for layer in model.layers:
        if layer.name in names:
            found.append(layer)
    if len(found) == len(names):
        return found
    # Legacy models (original notebook) use Keras auto-names; fall back to
    # "everything after the nested backbone".
    after = False
    found = []
    for layer in model.layers:
        if after:
            found.append(layer)
        if layer.name.startswith(BACKBONE_NAME):
            after = True
    return found
