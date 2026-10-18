"""
FILE PATH: SmartSpray/app/classification/explain.py

PURPOSE:
    Explainability for Module 3 — Grad-CAM (Selvaraju et al., 2017).

    Grad-CAM answers "which part of the leaf made the model say
    late_blight?". It weights the last convolutional feature map of
    EfficientNetB0 (top_activation, 7x7x1280) by the gradient of the class
    LOGIT (pre-softmax score, so confident predictions still give a useful
    signal), then upsamples the result to the crop size.

    Works with both the new model (named head layers) and the legacy
    notebook model (auto-named layers).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.classification.model_factory import BACKBONE_NAME, IMAGE_SIZE, head_layers


def _find_backbone(model):
    nested = [layer for layer in model.layers if hasattr(layer, "layers")]
    for layer in nested:
        if layer.name.startswith(BACKBONE_NAME):
            return layer
    if nested:
        return nested[0]
    raise ValueError("Could not find the nested EfficientNetB0 backbone in the model.")


_COMPILED: dict = {}


def _compiled_grad_fn(model):
    """
    One compiled (tf.function) Grad-CAM pass per model: about 4x faster than
    running the tape eagerly, which matters when the simulator explains
    every plant as the drone photographs it.
    """
    import tensorflow as tf

    key = id(model)
    if key in _COMPILED:
        return _COMPILED[key]
    backbone = _find_backbone(model)
    head = head_layers(model)
    if not head:
        raise ValueError("Could not locate the classification head layers.")
    *body, final = head

    @tf.function(input_signature=[tf.TensorSpec([1, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32),
                                  tf.TensorSpec([], tf.int32)])
    def fn(x, class_index):
        with tf.GradientTape() as tape:
            fmap = backbone(x, training=False)
            tape.watch(fmap)
            h = fmap
            for layer in body:
                h = layer(h, training=False)
            logits = tf.matmul(h, final.kernel) + final.bias
            idx = tf.where(class_index < 0, tf.cast(tf.argmax(logits[0]), tf.int32), class_index)
            score = tf.gather(logits[0], idx)
        grads = tape.gradient(score, fmap)
        weights = tf.reduce_mean(grads, axis=(1, 2))              # (1, C)
        cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1)[0])
        return cam, idx, tf.nn.softmax(logits)[0]

    _COMPILED[key] = fn
    return fn


def grad_cam(model, image_rgb: np.ndarray, class_index: int | None = None):
    """
    Compute a Grad-CAM heatmap for one RGB image (H, W, 3, 0-255).

    Returns:
        (heatmap float32 in [0, 1] with the image's HxW,
         class index explained,
         softmax probabilities (np.ndarray))
    """
    import tensorflow as tf

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {image_rgb.shape}")
    h0, w0 = image_rgb.shape[:2]
    img = image_rgb
    if (h0, w0) != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    x = tf.convert_to_tensor(img[None].astype("float32"))
    fn = _compiled_grad_fn(model)
    cam, idx, probs = fn(x, tf.constant(-1 if class_index is None else int(class_index), tf.int32))
    cam = cam.numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam.astype(np.float32), (w0, h0), interpolation=cv2.INTER_CUBIC)
    return np.clip(cam, 0.0, 1.0), int(idx.numpy()), probs.numpy()


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap (JET colours) over an RGB image. Returns RGB uint8."""
    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = image_rgb.astype(np.float32)
    out = (1 - alpha) * base + alpha * colored.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
