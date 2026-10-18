"""
FILE PATH: SmartSpray/scripts/severity_diagnostics.py

PURPOSE:
    Stage 3A real-image diagnostic evaluation for app/severity/estimator.py.

    This is a REAL-IMAGE SEGMENTATION DIAGNOSTIC / SANITY EVALUATION.
    It does NOT measure real-world severity accuracy, because no
    severity-labeled ground truth exists in the repository.

    What it actually does:
        1. Fixed-seed reproducible sampling of N images per disease class
           from data/datasets/plantvillage_tomato/ (formerly models/classification/prepared_dataset/; Module 3 frozen,
           read-only access).
        2. Runs the estimator on every sample, recording leaf_px, lesion_px,
           severity_score, severity_category, is_valid, leaf fill ratio.
        3. Cross-profile probe: every disease lesion profile is applied to
           every class. This tests (a) whether disease-specific profiles are
           discriminative on real images and (b) the healthy-class
           false-positive lesion rate (how much each profile fires on leaf
           tissue that is genuinely healthy).
        4. Flags suspicious cases (diseased-with-zero-lesion,
           healthy-with-lesions, invalid, near-full-lesion, tiny-leaf-mask).
        5. Writes evaluation/severity/stage3a/{summary.json, per_image.csv,
           cross_profile.csv, panels/} for visual inspection.

USAGE (repo root, inside .venv):
    python scripts/severity_diagnostics.py --samples 60
"""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.severity.estimator import (  # noqa: E402
    SeverityEstimator,
    DEFAULT_PROFILES,
    KNOWN_LABELS,
    CATEGORY_LOW,
    CATEGORY_MEDIUM,
    CATEGORY_HIGH,
    MODEL_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# The PlantVillage images moved from models/classification/prepared_dataset/
# to data/datasets/plantvillage_tomato/ during the Module 3 rebuild.
DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "plantvillage_tomato"
OUT_ROOT = REPO_ROOT / "evaluation" / "severity" / "stage3a"
PANEL_DIR = OUT_ROOT / "panels"

# Fixed seed so the whole evaluation is reproducible.
SEED = 42

# Number of images sampled per class (override via --samples).
DEFAULT_SAMPLES = 60

LABELS = list(KNOWN_LABELS)

# Severity band boundaries mirror app/severity/estimator.SEVERITY_BANDS
# (the canonical source). Kept here so the diagnostic's summary.json
# "bands" field reports the bands actually in effect.
# Canonical: <20% low / 20-50% medium / >50% high (original SmartSpray architecture).
BANDS: list[tuple[float, float, str]] = [
    (0.00, 0.20, CATEGORY_LOW),
    (0.20, 0.50, CATEGORY_MEDIUM),
    (0.50, 1.01, CATEGORY_HIGH),
]


def band_of(score: float) -> str:
    """Mirror app/severity/estimator.SEVERITY_BANDS for diagnostic reporting."""
    for lo, hi, cat in BANDS:
        if lo <= score < hi:
            return cat
    return CATEGORY_LOW

def run_one(est: SeverityEstimator, path: Path, label: str, idx: int) -> dict:
    """
    Compute estimator output AND probe-masks for one real image.

    Returns a flat dict usable for both per-image CSV and aggregate stats.
    For healthy class the estimator short-circuits to 0.0; we ALSO compute
    the leaf/lesion masks here so we can inspect whether the healthy class
    would produce false-positive lesions under each profile probe.
    """
    rgb = load_rgb(path)
    result = est.estimate(rgb, label, idx, "stage3a_diag")
    total_px = max(rgb.shape[0] * rgb.shape[1], 1)

    leaf_mask, leaf_px = est._segment_leaf(rgb)
    leaf_mask_bool = leaf_mask if leaf_mask.size else np.zeros((0, 0), dtype=bool)

    # Probe: lesion pixels produced by each disease profile on this image.
    probe: dict[str, int] = {}
    for dname, prof in DEFAULT_PROFILES.items():
        if leaf_px == 0:
            probe[dname] = 0
        else:
            _, lpx = est._segment_lesions(rgb, leaf_mask_bool, prof)
            probe[dname] = int(lpx)

    return {
        "class": label,
        "image": path.name,
        "path": str(path),
        "idx": idx,
        "valid": result.is_valid,
        "leaf_px": result.leaf_px,
        "lesion_px": result.lesion_px,
        "score": result.severity_score,
        "category": result.severity_category,
        "fill_ratio": round(result.leaf_px / total_px, 4),
        "leaf_mask_px": int(leaf_px),
        "probe_early_blight": probe["early_blight"],
        "probe_late_blight": probe["late_blight"],
        "probe_septoria": probe["septoria_leaf_spot"],
    }


def aggregate(records: list[dict]) -> dict:
    """Compute per-class aggregate statistics."""
    out: dict[str, dict] = {}
    for label in LABELS:
        rows = [r for r in records if r["class"] == label]
        if not rows:
            continue
        scores = [r["score"] for r in rows]
        cats = {"low": 0, "medium": 0, "high": 0}
        for r in rows:
            cats[r["category"]] += 1
        out[label] = {
            "n": len(rows),
            "n_valid": sum(1 for r in rows if r["valid"]),
            "n_invalid": sum(1 for r in rows if not r["valid"]),
            "n_zero_lesion": sum(1 for r in rows if r["lesion_px"] == 0),
            "n_nonzero_lesion": sum(1 for r in rows if r["lesion_px"] > 0),
            "score_mean": round(float(mean(scores)), 4),
            "score_median": round(float(median(scores)), 4),
            "score_min": round(float(min(scores)), 4),
            "score_max": round(float(max(scores)), 4),
            "category_counts": cats,
            "fill_ratio_mean": round(
                float(mean(r["fill_ratio"] for r in rows)), 4
            ),
            "leaf_mask_px_mean": round(
                float(mean(r["leaf_mask_px"] for r in rows)), 1
            ),
        }
    return out


def suspicious_flags(rec: dict) -> list[str]:
    flags: list[str] = []
    if not rec["valid"]:
        flags.append("invalid")
    if rec["leaf_mask_px"] < 500:
        flags.append("tiny_leaf_mask")
    if rec["class"] != "healthy" and rec["lesion_px"] == 0:
        flags.append("diseased_zero_lesion")
    if rec["class"] == "healthy" and (
        rec["probe_early_blight"] > 0
        or rec["probe_late_blight"] > 0
        or rec["probe_septoria"] > 0
    ):
        flags.append("healthy_has_probed_lesions")
    if rec["score"] >= 0.95:
        flags.append("near_full_leaf")
    return flags


def write_summary_json(agg: dict, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "summary.json"
    payload = {
        "estimator_model_version": MODEL_VERSION,
        "seed": SEED,
        "samples_per_class": DEFAULT_SAMPLES,
        "bands": BANDS,
        "per_class": agg,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[stage3a] wrote {path}")


def write_per_image_csv(records: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "per_image.csv"
    keys = [
        "class",
        "image",
        "idx",
        "valid",
        "leaf_px",
        "lesion_px",
        "score",
        "category",
        "fill_ratio",
        "leaf_mask_px",
        "probe_early_blight",
        "probe_late_blight",
        "probe_septoria",
        "flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in keys if k in r}
            row["flags"] = " | ".join(suspicious_flags(r)) or "-"
            w.writerow(row)
    print(f"[stage3a] wrote {path}")


def write_cross_profile_csv(records: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "cross_profile.csv"
    profiles = list(DEFAULT_PROFILES.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class",
                "probed_disease",
                "mean_probe_lesion_px",
                "n_samples",
                "n_with_any_probe_lesion",
            ]
        )
        for label in LABELS:
            rows = [r for r in records if r["class"] == label]
            n = len(rows)
            for prof in profiles:
                key = f"probe_{prof}"
                if key not in rows[0]:
                    key = f"probe_{prof.replace('_leaf_spot','')}"
                vals = [r[key] for r in rows]
                mean_probe = round(float(mean(vals)), 1) if vals else 0.0
                n_with = sum(1 for v in vals if v > 0)
                w.writerow([label, prof, mean_probe, n, n_with])
    print(f"[stage3a] wrote {path}")
def _panel_for_record(r: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(
        f"{r['class']}  {r['image']}  score={r['score']}  "
        f"cat={r['category']}  valid={r['valid']}",
        fontsize=9,
    )

    rgb = cv2.cvtColor(
        cv2.imread(r["path"]), cv2.COLOR_BGR2RGB
    )
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    leaf_mask = None
    if r["leaf_mask_px"] > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array((25, 40, 40), dtype=np.uint8)
        upper = np.array((90, 255, 255), dtype=np.uint8)
        lm = cv2.inRange(hsv, lower, upper) > 0
        kernel = np.ones((7, 7), dtype=np.uint8)
        lm = cv2.morphologyEx(
            lm.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            lm.astype(np.uint8), connectivity=8
        )
        if n > 1:
            idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            lm = labels == idx
        leaf_mask = lm.astype(np.uint8) * 255
    if leaf_mask is None or leaf_mask.size == 0:
        leaf_mask = np.zeros(
            (rgb.shape[0], rgb.shape[1]), dtype=np.uint8
        )
    axes[0, 1].imshow(leaf_mask, cmap="gray")
    axes[0, 1].set_title(f"leaf mask ({r['leaf_mask_px']} px)")
    axes[0, 1].axis("off")

    lesion_mask = np.zeros_like(leaf_mask)
    prof = DEFAULT_PROFILES.get(r["class"])
    if prof is not None and r["leaf_mask_px"] > 0:
        lm_bool = leaf_mask > 0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array(prof.hsv_lesion_lower, dtype=np.uint8)
        upper = np.array(prof.hsv_lesion_upper, dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower, upper) > 0
        lesion = hsv_mask
        if prof.lab_use:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab_lower = np.array(
                prof.lab_lesion_lower, dtype=np.uint8
            )
            lab_upper = np.array(
                prof.lab_lesion_upper, dtype=np.uint8
            )
            lesion = lesion | (
                cv2.inRange(lab, lab_lower, lab_upper) > 0
            )
        open_k = np.ones(
            (prof.open_kernel_size, prof.open_kernel_size),
            dtype=np.uint8,
        )
        lesion = cv2.morphologyEx(
            lesion.astype(np.uint8), cv2.MORPH_OPEN, open_k
        ).astype(bool)
        close_k = np.ones(
            (prof.close_kernel_size, prof.close_kernel_size),
            dtype=np.uint8,
        )
        lesion = cv2.morphologyEx(
            lesion.astype(np.uint8), cv2.MORPH_CLOSE, close_k
        ).astype(bool)
        lesion = lesion & lm_bool
        lesion_mask = lesion.astype(np.uint8) * 255
    axes[1, 0].imshow(lesion_mask, cmap="gray")
    axes[1, 0].set_title(
        f"lesion mask ({r['class']}, {r['lesion_px']} px)"
    )
    axes[1, 0].axis("off")

    probes = {
        "early_blight": r["probe_early_blight"],
        "late_blight": r["probe_late_blight"],
        "septoria": r["probe_septoria"],
    }
    ax = axes[1, 1]
    bars = ax.bar(
        list(probes.keys()), list(probes.values()), color="#4c72b0"
    )
    ax.set_title("cross-profile probe (lesion px)")
    ax.set_ylabel("lesion px")
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    return fig


def write_panels(records: list[dict], panel_dir: Path) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for r in records:
        fig = _panel_for_record(r)
        safe = r["image"].replace(" ", "_").replace("/", "_")
        out = panel_dir / f"{r['class']}__{safe}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        n_written += 1
    print(f"[stage3a] wrote {n_written} panel PNGs -> {panel_dir}")


def load_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_images(label: str, n: int, rng: random.Random) -> list[Path]:
    """Return a reproducible sample of image paths for one class."""
    folder = DATASET_ROOT / label
    files = (
        sorted(folder.glob("*.jpg"))
        + sorted(folder.glob("*.JPG"))
        + sorted(folder.glob("*.png"))
    )
    rng.shuffle(files)
    return files[:n]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3A real-image segmentation diagnostic for the "
            "severity estimator."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"images sampled per class (default {DEFAULT_SAMPLES}).",
    )
    args = parser.parse_args(argv)

    est = SeverityEstimator()
    rng = random.Random(SEED)
    records: list[dict] = []

    for label in LABELS:
        paths = sample_images(label, args.samples, rng)
        for idx, path in enumerate(paths):
            try:
                rec = run_one(est, path, label, idx)
                rec["flags"] = " | ".join(suspicious_flags(rec)) or "-"
                records.append(rec)
            except Exception as exc:
                print(f"[stage3a] SKIP {path}: {exc}")

    print(
        f"[stage3a] evaluated {len(records)} images "
        f"({args.samples} per class x {len(LABELS)} classes)"
    )

    agg = aggregate(records)
    write_summary_json(agg, OUT_ROOT)
    write_per_image_csv(records, OUT_ROOT)
    write_cross_profile_csv(records, OUT_ROOT)
    write_panels(records, PANEL_DIR)

    print("\n[stage3a] per-class summary:")
    for label in LABELS:
        a = agg.get(label)
        if a is None:
            continue
        print(
            f"  {label}: n={a['n']} valid={a['n_valid']} "
            f"invalid={a['n_invalid']} "
            f"zero_lesion={a['n_zero_lesion']} "
            f"nonzero_lesion={a['n_nonzero_lesion']} "
            f"score_mean={a['score_mean']} "
            f"score_median={a['score_median']} "
            f"fill_ratio_mean={a['fill_ratio_mean']} "
            f"leaf_mask_px_mean={a['leaf_mask_px_mean']} "
            f"cats={a['category_counts']}"
        )
    print("\n[stage3a] done.")


if __name__ == "__main__":
    main()

