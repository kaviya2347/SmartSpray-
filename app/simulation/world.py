"""
app/simulation/world.py

Synthetic tomato field for the drone mission simulator, and the "camera"
that photographs it.

The field is the same kind of scene as scripts/make_demo_field.py: every
plant is a real PlantVillage photo from the held-out TEST split (never used
for training or model selection), laid out in rows on a generated soil
background. Disease appears in patches, the way it spreads in a real crop.

Three kinds of image are produced, all from one high-resolution ground
texture (220 px per metre):

    texture(drops)          the ground as it really is, including the ArUco
                            markers the drone has dropped
    overview(drops)         what the drone camera sees from high above the
                            field (lower resolution, slight tilt, lighting,
                            sensor noise) -> used to find markers and plants
    closeup(drops, x, y)    a low-altitude photo centred on (x, y)
                            -> the photo of one plant that the model classifies

Coordinates: field metres, x east, y north, origin at the south-west corner
(where marker 0 goes). The simulator's 3D view uses the same frame.

SYNTHETIC: this is a demonstration environment, not field validation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = REPO_ROOT / "models" / "classification" / "split_manifest.json"
DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "plantvillage_tomato"

PRESETS = {
    # 63 plants — same size and layout as the scripted demo field.
    "small": {"rows": 7, "cols": 9, "width_m": 10.0, "height_m": 8.0},
    # 117 plants — a longer mission.
    "medium": {"rows": 9, "cols": 13, "width_m": 14.0, "height_m": 10.0},
}

TEXTURE_PPM = 220            # ground texture: pixels per metre (~4.5 mm / px)
TEXTURE_MARGIN_M = 1.0       # bare headland around the field in the texture
OVERVIEW_PPM = 110           # overview photo from ~15 m: ~9 mm / px
CLOSEUP_VIEW_M = 1.6         # close-up photo footprint from scan altitude
PLANT_SIZE_M = 0.855         # canopy tile size (188 px at 220 px/m, as in the demo)
MARKER_SIZE_M = 0.60         # black ArUco square
MARKER_QUIET_M = 0.12        # white border around it
ARUCO_DICT = "DICT_4X4_50"
HOME = (-2.0, -1.6)          # take-off / refill pad, outside the south-west corner

# The scripted demo layout (row 0 = northern row), used for preset "small"
# with seed 7 so the simulator matches scripts/make_demo_field.py.
_DEMO_LATE_BLIGHT = [(0, 7), (0, 8), (1, 7), (1, 8), (1, 6), (2, 8)]
_DEMO_EARLY_BLIGHT = [(5, 0), (5, 1), (6, 0), (6, 1), (4, 1), (6, 2), (4, 0)]
_DEMO_SEPTORIA = [(3, 3), (3, 4), (3, 5), (2, 4), (4, 4), (3, 2)]

DISEASES = ("late_blight", "early_blight", "septoria_leaf_spot")


# ── World generation ─────────────────────────────────────────────────────────

def _test_pools(seed: int) -> dict[str, list[dict]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "models/classification/split_manifest.json not found — run "
            "scripts/m3_train_classifier.py --stage split")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rnd = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for label in manifest["class_order"]:
        pool = [r for r in manifest["images"]
                if r["split"] == "test" and r["label"] == label]
        rnd.shuffle(pool)
        pools[label] = pool
    return pools


def _demo_layout(rows: int) -> dict[tuple[int, int], str]:
    """Demo (row-from-north, col) cells -> our (row-from-south, col)."""
    out = {}
    for cells, label in ((_DEMO_LATE_BLIGHT, "late_blight"),
                         (_DEMO_EARLY_BLIGHT, "early_blight"),
                         (_DEMO_SEPTORIA, "septoria_leaf_spot")):
        for r, c in cells:
            out[(rows - 1 - r, c)] = label
    return out


def _patch_layout(rows: int, cols: int, rng: np.random.Generator) -> dict:
    """One spreading patch per disease (~9 % of plants each)."""
    layout: dict[tuple[int, int], str] = {}
    target = max(4, round(rows * cols * 0.09))
    for label in DISEASES:
        free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in layout]
        start = free[int(rng.integers(len(free)))]
        patch, frontier = {start}, [start]
        while len(patch) < target and frontier:
            r, c = frontier.pop(int(rng.integers(len(frontier))))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                nb = (r + dr, c + dc)
                if (0 <= nb[0] < rows and 0 <= nb[1] < cols and nb not in layout
                        and nb not in patch and rng.random() < 0.55):
                    patch.add(nb)
                    frontier.append(nb)
                    if len(patch) >= target:
                        break
            if not frontier and len(patch) < target:
                frontier = list(patch)
        for cell in patch:
            layout[cell] = label
    return layout


def make_world(sim_id: str, seed: int = 7, preset: str = "small") -> dict:
    """Build a JSON-serialisable description of a synthetic field."""
    if preset not in PRESETS:
        raise ValueError(f"unknown field preset '{preset}' (use {sorted(PRESETS)})")
    p = PRESETS[preset]
    rows, cols, width, height = p["rows"], p["cols"], p["width_m"], p["height_m"]
    rng = np.random.default_rng(seed)
    layout = (_demo_layout(rows) if (preset == "small" and seed == 7)
              else _patch_layout(rows, cols, rng))
    pools = _test_pools(seed)

    plants = []
    gid = 0
    for r in range(rows):
        for c in range(cols):
            label = layout.get((r, c), "healthy")
            if not pools[label]:
                raise ValueError(f"not enough test-split photos for '{label}'")
            src = pools[label].pop()
            gid += 1
            plants.append({
                "gid": gid, "row": r, "col": c,
                "x_m": round(1.0 + c + float(rng.uniform(-0.08, 0.08)), 4),
                "y_m": round(1.0 + r + float(rng.uniform(-0.08, 0.08)), 4),
                "true_label": label,
                "size_m": round(PLANT_SIZE_M * float(rng.uniform(0.95, 1.03)), 4),
                "flip": bool(rng.random() < 0.5),
                "source_image": src["path"],
                "source_split": src["split"],
            })

    counts: dict[str, int] = {}
    for pl in plants:
        counts[pl["true_label"]] = counts.get(pl["true_label"], 0) + 1
    return {
        "sim_id": sim_id,
        "seed": seed,
        "preset": preset,
        "field_width_m": width,
        "field_height_m": height,
        "rows": rows,
        "cols": cols,
        "plants": plants,
        "label_counts": counts,
        "home": {"x_m": HOME[0], "y_m": HOME[1]},
        # The operator marks the field corners roughly on a map; the drone
        # flies there and drops a marker at each one.
        "planned_corners": [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        "marker": {"dictionary": ARUCO_DICT, "ids": [0, 1, 2, 3],
                   "size_m": MARKER_SIZE_M, "plate_m": MARKER_SIZE_M + 2 * MARKER_QUIET_M},
        "camera": {"overview_ppm": OVERVIEW_PPM, "closeup_view_m": CLOSEUP_VIEW_M,
                   "texture_ppm": TEXTURE_PPM},
        "note": ("SYNTHETIC field: real PlantVillage TEST-split photos on a generated soil "
                 "background. Demonstration only — not a field trial."),
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def _soil(h: int, w: int, rng: np.random.Generator, furrow_px: list[int]) -> np.ndarray:
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (62, 92, 128)                                     # BGR brown soil
    coarse = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    base += (coarse * 10 + fine * 5)[..., None]
    for y in furrow_px:
        cv2.line(base, (0, y), (w, y), (48, 72, 100), 18)
    return np.clip(base, 0, 255).astype(np.uint8)


def _paste(canvas: np.ndarray, tile: np.ndarray, cx: float, cy: float) -> None:
    size = tile.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    edge = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy])
    alpha = np.clip(edge / 6.0, 0, 1)[..., None]
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    roi = canvas[y0:y0 + size, x0:x0 + size].astype(np.float32)
    canvas[y0:y0 + size, x0:x0 + size] = (alpha * tile + (1 - alpha) * roi).astype(np.uint8)


def aruco_marker(marker_id: int, side_px: int, quiet_px: int) -> np.ndarray:
    """Grey ArUco marker (DICT_4X4_50) with a white quiet zone."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), int(side_px))
    return cv2.copyMakeBorder(marker, quiet_px, quiet_px, quiet_px, quiet_px,
                              cv2.BORDER_CONSTANT, value=245)


def aruco_png(marker_id: int, side_px: int = 192) -> bytes:
    img = aruco_marker(marker_id, side_px, int(side_px * MARKER_QUIET_M / MARKER_SIZE_M))
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("could not encode marker")
    return buf.tobytes()


class FieldRenderer:
    """Renders camera images of one world. Caches the ground texture."""

    def __init__(self, world: dict, dataset_root: Path = DATASET_ROOT):
        self.world = world
        self.dataset_root = Path(dataset_root)
        self.ppm = TEXTURE_PPM
        self.margin = TEXTURE_MARGIN_M
        self.w_m = float(world["field_width_m"])
        self.h_m = float(world["field_height_m"])
        self.w_px = int(round((self.w_m + 2 * self.margin) * self.ppm))
        self.h_px = int(round((self.h_m + 2 * self.margin) * self.ppm))
        self._base: Optional[np.ndarray] = None
        self._texture: Optional[np.ndarray] = None
        self._texture_key: Optional[str] = None

    # field metres <-> texture pixels
    def to_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return ((x_m + self.margin) * self.ppm, (self.h_m + self.margin - y_m) * self.ppm)

    def _ground(self) -> np.ndarray:
        """Soil + plants (no markers). Built once."""
        if self._base is not None:
            return self._base
        rng = np.random.default_rng(int(self.world["seed"]) + 101)
        rows = int(self.world["rows"])
        furrows = [int(self.to_px(0, 0.5 + r)[1]) for r in range(rows + 1)]
        canvas = _soil(self.h_px, self.w_px, rng, furrows)
        for p in self.world["plants"]:
            img = cv2.imread(str(self.dataset_root / p["source_image"]))
            if img is None:
                raise FileNotFoundError(
                    f"plant photo missing: data/datasets/plantvillage_tomato/{p['source_image']}")
            size = int(round(p["size_m"] * self.ppm))
            tile = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            if p.get("flip"):
                tile = cv2.flip(tile, 1)
            _paste(canvas, tile, *self.to_px(p["x_m"], p["y_m"]))
        self._base = canvas
        return canvas

    def texture(self, drops: list[dict]) -> np.ndarray:
        """Ground with every dropped marker lying where it landed."""
        key = json.dumps(sorted((int(d["marker_id"]), round(float(d["x_m"]), 3),
                                 round(float(d["y_m"]), 3)) for d in drops))
        if self._texture is not None and self._texture_key == key:
            return self._texture
        canvas = self._ground().copy()
        side = int(round(MARKER_SIZE_M * self.ppm))
        quiet = int(round(MARKER_QUIET_M * self.ppm))
        for d in drops:
            m = cv2.cvtColor(aruco_marker(int(d["marker_id"]), side, quiet), cv2.COLOR_GRAY2BGR)
            cx, cy = self.to_px(float(d["x_m"]), float(d["y_m"]))
            s = m.shape[0]
            x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
            x1, y1 = max(0, x0), max(0, y0)
            x2, y2 = min(self.w_px, x0 + s), min(self.h_px, y0 + s)
            if x2 > x1 and y2 > y1:
                canvas[y1:y2, x1:x2] = m[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        self._texture, self._texture_key = canvas, key
        return canvas

    def overview(self, drops: list[dict], seed: int = 0) -> np.ndarray:
        """High-altitude photo of the whole field (used for mapping)."""
        rng = np.random.default_rng(int(self.world["seed"]) * 31 + seed)
        tex = self.texture(drops)
        scale = OVERVIEW_PPM / float(self.ppm)
        img = cv2.resize(tex, (int(self.w_px * scale), int(self.h_px * scale)),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        h, w = img.shape[:2]
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        tilt = float(rng.uniform(0.15, 0.25))
        light = 0.9 + tilt * (gx / w) * (1 - 0.3 * gy / h)
        img = np.clip(img * light[..., None] + rng.normal(0, 2.0, img.shape), 0, 255)
        img = img.astype(np.uint8)
        # mild perspective: the camera is never perfectly nadir
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        d = float(rng.uniform(0.015, 0.03))
        dst = np.float32([[w * d, h * 0.01], [w * (1 - 0.005), 0],
                          [w * (1 - d * 0.6), h * (1 - 0.01)], [w * 0.004, h]])
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    def closeup(self, drops: list[dict], x_m: float, y_m: float, seed: int = 0) -> np.ndarray:
        """Low-altitude photo centred on the drone's position (gimbal north-up)."""
        rng = np.random.default_rng(int(self.world["seed"]) * 977 + seed)
        tex = self.texture(drops)
        size = int(round(CLOSEUP_VIEW_M * self.ppm))
        cx, cy = self.to_px(x_m, y_m)
        x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
        pad = size
        padded = cv2.copyMakeBorder(tex, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        crop = padded[y0 + pad:y0 + pad + size, x0 + pad:x0 + pad + size].astype(np.float32)
        gain = float(rng.uniform(0.95, 1.05))
        crop = np.clip(crop * gain + rng.normal(0, 2.0, crop.shape), 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(crop, (3, 3), 0.6)

    def ground_truth_pixel(self, overview_shape, x_m: float, y_m: float):
        """Approximate overview pixel of a field point (before tilt) — for debugging."""
        scale = OVERVIEW_PPM / float(self.ppm)
        px, py = self.to_px(x_m, y_m)
        return px * scale, py * scale
