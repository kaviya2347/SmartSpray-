"""
FILE PATH: SmartSpray/scripts/make_demo_field.py

Builds a SYNTHETIC top-down "drone orthomosaic" of a small tomato plot so the
whole SmartSpray pipeline (M1 -> M9) can be demonstrated end-to-end without
field imagery:

    * 10 m x 8 m plot, 7 rows x 9 plants, 1 m spacing (63 plants)
    * every plant is a real PlantVillage leaf photo taken ONLY from the
      held-out TEST split of models/classification/split_manifest.json
      (never seen in training or model selection)
    * realistic disease layout: a late-blight hotspot, an early-blight patch,
      a septoria patch, the rest healthy
    * four ArUco markers (DICT_4X4_50, ids 0-3) at the plot corners so Module 5
      can calibrate pixels -> metres
    * mild perspective tilt, lighting gradient and sensor noise
    * ground_truth.json with each plant's true label, field position and pixel
      position — used by the report for a system-level evaluation

It is a demonstration aid, NOT field validation. Results on this image must
be described as "synthetic demo field" in any presentation.

USAGE:
    python scripts/make_demo_field.py [--survey-id demo_survey_001] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "models" / "classification" / "split_manifest.json"
DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "plantvillage_tomato"
IMAGES_ROOT = REPO_ROOT / "data" / "images"

FIELD_W_M, FIELD_H_M = 10.0, 8.0
PPM = 220                     # pixels per metre (ground sampling distance ~4.5 mm/px)
MARGIN_PX = 200
PLANT_PX = 188
MARKER_PX = 160
ROWS, COLS = 7, 9

# Disease layout (row, col) -> label; everything else healthy.
LATE_BLIGHT_HOTSPOT = [(0, 7), (0, 8), (1, 7), (1, 8), (1, 6), (2, 8)]
EARLY_BLIGHT_PATCH = [(5, 0), (5, 1), (6, 0), (6, 1), (4, 1), (6, 2), (4, 0)]
SEPTORIA_PATCH = [(3, 3), (3, 4), (3, 5), (2, 4), (4, 4), (3, 2)]


def field_to_px(x_m: float, y_m: float) -> tuple[float, float]:
    """Field metres (x east, y north, origin = marker 0) -> canvas pixels."""
    return MARGIN_PX + x_m * PPM, MARGIN_PX + (FIELD_H_M - y_m) * PPM


def soil_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (62, 92, 128)                                  # BGR brown soil
    noise = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    base += (noise * 10 + fine * 5)[..., None]
    # planting rows: slightly darker furrows between rows
    for r in range(ROWS + 1):
        y = int(field_to_px(0, 0.5 + r * 1.0)[1])
        cv2.line(base, (0, y), (w, y), (48, 72, 100), 18)
    return np.clip(base, 0, 255).astype(np.uint8)


def paste_plant(canvas: np.ndarray, tile_bgr: np.ndarray, cx: float, cy: float,
                size: int, rng: np.random.Generator) -> None:
    """
    Paste the leaf photo as a square "canopy tile" with softly rounded
    corners. The photo keeps its own neutral background: PlantVillage-trained
    classifiers read brown soil around a leaf as lesion tissue, so the demo
    keeps crops close to the training domain (see README, limitations).
    """
    tile = cv2.resize(tile_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        tile = cv2.flip(tile, 1)                                  # mirror only
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    edge = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy])
    alpha = np.clip(edge / 6.0, 0, 1)[..., None]                 # 6 px soft border
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    roi = canvas[y0:y0 + size, x0:x0 + size].astype(np.float32)
    canvas[y0:y0 + size, x0:x0 + size] = (alpha * tile + (1 - alpha) * roi).astype(np.uint8)


def draw_marker(canvas: np.ndarray, marker_id: int, cx: float, cy: float) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
    quiet = MARKER_PX // 5
    x0, y0 = int(round(cx - MARKER_PX / 2)), int(round(cy - MARKER_PX / 2))
    cv2.rectangle(canvas, (x0 - quiet, y0 - quiet),
                  (x0 + MARKER_PX + quiet, y0 + MARKER_PX + quiet), (245, 245, 245), -1)
    canvas[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey-id", default="demo_survey_001")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-tilt", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit("split_manifest.json not found — run scripts/m3_train_classifier.py --stage split")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    test_rows = [r for r in manifest["images"] if r["split"] == "test"]
    pools = {}
    rnd = random.Random(args.seed)
    for label in manifest["class_order"]:
        pool = [r for r in test_rows if r["label"] == label]
        rnd.shuffle(pool)
        pools[label] = pool

    layout = {}
    for rc in LATE_BLIGHT_HOTSPOT:
        layout[rc] = "late_blight"
    for rc in EARLY_BLIGHT_PATCH:
        layout[rc] = "early_blight"
    for rc in SEPTORIA_PATCH:
        layout[rc] = "septoria_leaf_spot"

    rng = np.random.default_rng(args.seed)
    h = int(FIELD_H_M * PPM + 2 * MARGIN_PX)
    w = int(FIELD_W_M * PPM + 2 * MARGIN_PX)
    canvas = soil_texture(h, w, rng)

    plants = []
    for r in range(ROWS):
        for c in range(COLS):
            label = layout.get((r, c), "healthy")
            src = pools[label].pop()
            x_m = 1.0 + c * 1.0 + float(rng.uniform(-0.08, 0.08))
            y_m = FIELD_H_M - 1.0 - r * 1.0 + float(rng.uniform(-0.08, 0.08))
            cx, cy = field_to_px(x_m, y_m)
            tile = cv2.imread(str(DATASET_ROOT / src["path"]))
            size = int(PLANT_PX * rng.uniform(0.95, 1.03))
            paste_plant(canvas, tile, cx, cy, size, rng)
            plants.append({"row": r, "col": c, "true_label": label,
                           "field_x_m": round(x_m, 4), "field_y_m": round(y_m, 4),
                           "pixel_x": cx, "pixel_y": cy, "size_px": size,
                           "source_image": src["path"], "source_sha256": src["sha256"],
                           "source_split": src["split"]})

    for marker_id, (fx, fy) in enumerate([(0, 0), (FIELD_W_M, 0),
                                          (FIELD_W_M, FIELD_H_M), (0, FIELD_H_M)]):
        draw_marker(canvas, marker_id, *field_to_px(fx, fy))

    # Lighting gradient + sensor noise.
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.9 + 0.2 * (gx / w) * (1 - 0.3 * gy / h)
    img = np.clip(canvas.astype(np.float32) * light[..., None]
                  + rng.normal(0, 2.0, canvas.shape), 0, 255).astype(np.uint8)

    # Mild perspective tilt (oblique drone view).
    if not args.no_tilt:
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        d = 0.025
        dst_pts = np.float32([[w * d, h * 0.01], [w * (1 - 0.005), 0],
                              [w * (1 - d * 0.6), h * (1 - 0.01)], [w * 0.004, h]])
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
        pts = np.float32([[[p["pixel_x"], p["pixel_y"]]] for p in plants])
        warped = cv2.perspectiveTransform(pts, H)
        for p, q in zip(plants, warped):
            p["pixel_x"], p["pixel_y"] = float(q[0][0]), float(q[0][1])

    out_dir = IMAGES_ROOT / args.survey_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_name = "field_orthomosaic.jpg"
    cv2.imwrite(str(out_dir / image_name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    for p in plants:
        p["image_filename"] = image_name
        p["pixel_x"], p["pixel_y"] = round(p["pixel_x"], 2), round(p["pixel_y"], 2)
    counts = {}
    for p in plants:
        counts[p["true_label"]] = counts.get(p["true_label"], 0) + 1
    gt = {
        "survey_id": args.survey_id,
        "note": ("SYNTHETIC demo field: real PlantVillage TEST-split leaf photos placed on a "
                 "generated soil background with ArUco corner markers. Demonstration only — "
                 "not a field validation."),
        "field_width_m": FIELD_W_M, "field_height_m": FIELD_H_M,
        "pixels_per_metre_before_tilt": PPM,
        "aruco_dictionary": "DICT_4X4_50", "marker_ids": [0, 1, 2, 3],
        "match_radius_px": 90,
        "label_counts": counts,
        "seed": args.seed,
        "plants": plants,
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / image_name} ({w}x{h}) with {len(plants)} plants: {counts}")


if __name__ == "__main__":
    main()
