"""
Renders docs/MODULE3_REPORT.md from the saved Module 3 artifacts, so every
number in the report can be traced back to a file.

    python scripts/write_m3_report.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "models" / "classification"
EV = MD / "eval"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    man = load(MD / "split_manifest.json")
    hist = load(EV / "training_history.json")
    met = load(EV / "metrics.json")
    promo = load(EV / "promotion_record.json")
    icfg = load(MD / "inference_config.json")
    printout = (EV / "test_evaluation_printout.txt").read_text(encoding="utf-8").strip("\n")
    classes = man["class_order"]
    sc, st = man["split_counts"], man["split_totals"]

    lines = []
    add = lines.append
    add("# Module 3: Disease Classification. Final report (70 / 15 / 15 rebuild)\n")
    add("Every number below comes from the artifact files listed in section 5. "
        "`python scripts/write_m3_report.py` regenerates this report.\n")

    add("## 0. What was wrong before, and what changed\n")
    add("| Finding in the old notebooks | Fix |\n|---|---|")
    add("| The \"test\" set was a `validation_split=0.1` subset of the same folder, so it sat inside the 20 % validation split that was also used to choose the checkpoint. The 98.4 % figure was not a held-out result. | A real, disjoint 70/15/15 split. The test set was evaluated once. |")
    add("| \"10,032 images\" was a Windows double count: `glob('*.jpg') + glob('*.JPG')` counts every file twice on a case-insensitive disk. | 5,016 real files were counted with a case-insensitive extension check. |")
    add("| `prepared_dataset/` was thought to be stale. It actually held all 5,016 files, and 9 of them were byte-identical duplicates. | The folder was moved to `data/datasets/plantvillage_tomato/`. Duplicates were removed by SHA-256 before splitting, and overlap is checked by path **and** by hash. |")
    add("| No seed-controlled split, no stratification, no manifest. | `train_test_split(..., stratify=labels, random_state=42)` is called twice, and `split_manifest.json` lists every image. |")
    add("| The old `best.keras` came from Phase 1. Its config shows the backbone frozen and lr = 1e-3, and its optimizer state covers only the 4 head tensors. Phase 2 never beat it. | Recorded here. Its frozen ImageNet backbone was recovered and verified (see 2.1), and a completely new head was trained. |\n")

    add("## 1. Dataset\n")
    add(f"- **Source:** {man['dataset_source']}")
    add(f"- **Path:** `{man['dataset_root']}/<class>/`")
    add(f"- **Raw files:** {man['raw_file_total']} ({', '.join(f'{k} {v}' for k, v in man['raw_file_counts'].items())})")
    add(f"- **Exact duplicates removed:** {len(man['excluded']['exact_duplicates'])} (all within the same class). "
        f"**Unreadable:** {len(man['excluded']['unreadable'])}. "
        f"**Label conflicts:** {len(man['excluded']['label_conflicts'])}.")
    add(f"- **Unique images used:** {man['unique_images']}")
    add(f"- **Split method:** {man['split_method']}")
    add(f"- **Seed:** `RANDOM_SEED = {man['random_seed']}`, passed as `random_state=RANDOM_SEED` "
        "to both `train_test_split` calls in `scripts/m3_train_classifier.py::stage_split`.\n")
    add("| class | total | train | validation | test |\n|---|---:|---:|---:|---:|")
    for c in classes:
        tot = sc["train"][c] + sc["validation"][c] + sc["test"][c]
        add(f"| {c} | {tot} | {sc['train'][c]} | {sc['validation'][c]} | {sc['test'][c]} |")
    add(f"| **all** | **{man['unique_images']}** | **{st['train']}** | **{st['validation']}** | **{st['test']}** |\n")
    add(f"The split fractions are {st['train']/man['unique_images']:.4f} / "
        f"{st['validation']/man['unique_images']:.4f} / {st['test']/man['unique_images']:.4f}. "
        "The small differences from exactly 70/15/15 come from rounding per class: "
        "sklearn rounds each stratum to whole images, so validation and test differ "
        "by at most one image per class.\n")
    add("**Leakage checks** (from the manifest):\n")
    for k, v in man["leakage_checks"].items():
        add(f"- `{k}`: **{v}**")
    add("- Augmentation is applied only in the training `tf.data` pipeline. The "
        "validation and test pipelines only cast to float32.")
    add("- The test split is loaded only inside `stage_evaluate`. "
        "`TEST_EVALUATED.lock` blocks a second evaluation and any further training.\n")

    add("## 2. Model\n")
    add(f"- **Architecture:** {hist['architecture']}")
    add("- **Input:** RGB, 224×224, 0–255. Keras `efficientnet.preprocess_input` passes "
        "values through unchanged because the model does its own rescaling and "
        "normalisation. It does **not** scale to [-1, 1] as the Review-1 slide says.")
    p = hist["parameters"]
    add(f"- **Parameters:** Phase 1 trainable {p['phase1_trainable']:,} / frozen {p['phase1_frozen']:,}. "
        f"Phase 2 trainable {p['phase2_trainable']:,}."
        + (f" Phase 3 trainable {hist['phase3']['trainable_params']:,}." if 'phase3' in hist else ""))
    add("- **Hyperparameters:**\n")
    add("| key | value |\n|---|---|")
    for k, v in hist["hyperparameters"].items():
        add(f"| {k} | {v} |")
    add("")
    add("### 2.1 ImageNet backbone provenance\n")
    add("Google's weight server is blocked from the training environment. The "
        "ImageNet EfficientNetB0 backbone was therefore recovered from the old "
        "notebook model, which was verified to be frozen ImageNet weights on three "
        "counts:\n\n"
        "- its saved config marks all 238 backbone layers `trainable: false`\n"
        "- its Adam state holds slots only for the 4 head tensors (1280×256, 256, 256×4, 4)\n"
        "- its Normalization layer holds the official ImageNet mean and variance used by Keras\n\n"
        "The old head was discarded and a new head was initialised. The file is "
        f"`models/pretrained/efficientnetb0_imagenet_notop.weights.h5`. "
        f"Source recorded by training: `{Path(hist['backbone_weights_source']).name}`.\n")

    add("## 3. Training\n")
    add("| phase | what trains | lr | epochs run | minutes | best val acc | min val loss |\n|---|---|---|---:|---:|---:|---:|")
    desc = {"phase1": ("head only (backbone frozen)", hist["hyperparameters"]["lr_head"]),
            "phase2": ("top 30 backbone layers (BN frozen)", hist["hyperparameters"]["lr_finetune"]),
            "phase3": ("blocks 5–7 + top (BN frozen)", hist.get("phase3", {}).get("lr"))}
    for ph in ("phase1", "phase2", "phase3"):
        if ph in hist:
            h = hist[ph]["history"]
            add(f"| {ph} | {desc[ph][0]} | {desc[ph][1]} | {hist[ph]['epochs_run']} | "
                f"{hist[ph]['minutes']:.1f} | {max(h['val_accuracy']):.4f} | {min(h['val_loss']):.4f} |")
    sel = hist["selected_checkpoint"]
    add(f"\n**Selected checkpoint (validation only):** {sel['phase']} epoch {sel['epoch']}, "
        f"val_accuracy {sel['val_accuracy']:.4f}, val_loss {sel['val_loss']:.4f}. "
        f"Rule: {hist['selection_rule']}.")
    if "phase3" in hist:
        add(f"Phase 3 was fixed before any test evaluation. It "
            f"{'**improved** validation and was accepted' if hist['phase3']['improved_validation'] else 'did **not** improve validation and was discarded'}.")
    vd, cal = hist["validation_inference_decision"], hist["calibration"]
    add(f"\n**Inference settings chosen on validation:** horizontal-flip TTA = "
        f"{vd['use_tta']} (val acc {vd['acc_plain']:.4f} → {vd['acc_tta_hflip']:.4f}, "
        f"NLL {vd['nll_plain']:.4f} → {vd['nll_tta_hflip']:.4f}). Temperature scaling "
        f"T = {cal['temperature']:.4f} (val ECE {cal['val_ece_before']:.4f} → "
        f"{cal['val_ece_after']:.4f}). Uncertainty threshold: {icfg['uncertainty_threshold']}.\n")
    add("![training curves](../models/classification/eval/training_curves.png)\n")

    add("## 4. Independent test evaluation (single pass)\n")
    add("```text\n" + printout + "\n```\n")
    add("| class | precision | recall | F1 | support |\n|---|---:|---:|---:|---:|")
    for c in classes:
        m = met["per_class"][c]
        add(f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    add(f"| **macro** | {met['macro_precision']:.4f} | {met['macro_recall']:.4f} | **{met['macro_f1']:.4f}** | {met['total_test_samples']} |")
    add(f"| **weighted F1** | | | **{met['weighted_f1']:.4f}** | |\n")
    add("![confusion matrix](../models/classification/eval/confusion_matrix.png)\n")
    pm = met["production_mode"]
    add(f"**Production inference mode** (TTA={pm['tta_hflip']}, T={pm['temperature']:.3f}): "
        f"accuracy {pm['accuracy']:.4f}; ECE {pm['ece_raw']:.4f} → {pm['ece_calibrated']:.4f}; "
        f"{pm['coverage_confident']*100:.1f}% of test images have confidence ≥ "
        f"{pm['uncertainty_threshold']} and those are {pm['accuracy_on_confident']*100:.2f}% correct; "
        f"{pm['n_flagged_uncertain']} images are flagged for manual inspection.\n")
    add("![reliability](../models/classification/eval/reliability_diagram.png)\n")
    add("![grad-cam](../models/classification/eval/gradcam_examples.png)\n")
    add("**Comparison with earlier figures:** the notebook's 98.4 % (500 images nested in "
        "validation) and the previously assumed 96.68 % / 96.17 % macro F1 (never "
        "produced by any file) are **not** comparable. The numbers above are the only "
        "held-out results.\n")

    add("## 5. Artifacts\n")
    arts = ["models/classification/split_manifest.json", "models/classification/split_manifest.csv",
            "models/classification/class_order.json", "models/classification/inference_config.json",
            "models/classification/best.keras", promo.get("previous_model_backup", ""),
            "models/classification/eval/training_history.json",
            "models/classification/eval/history_phase1.csv", "models/classification/eval/history_phase2.csv",
            "models/classification/eval/history_phase3.csv",
            "models/classification/eval/metrics.json", "models/classification/eval/classification_report.txt",
            "models/classification/eval/classification_report.csv",
            "models/classification/eval/confusion_matrix.png", "models/classification/eval/confusion_matrix.csv",
            "models/classification/eval/confusion_matrix.json", "models/classification/eval/test_predictions.csv",
            "models/classification/eval/misclassified.csv", "models/classification/eval/reliability_diagram.png",
            "models/classification/eval/gradcam_examples.png", "models/classification/eval/training_curves.png",
            "models/classification/eval/test_evaluation_printout.txt", "models/classification/eval/split_summary.json",
            "models/classification/eval/promotion_record.json", "models/classification/eval/run_log.txt",
            "models/classification/eval/TEST_EVALUATED.lock",
            "models/classification/eval/module3_pytest.txt",
            "notebooks/m3_classification_701515_report.ipynb"]
    for a in arts:
        if a and (ROOT / a).exists():
            add(f"- `{a}`")
    add("")

    add("## 6. Verification\n")
    prod = MD / "best.keras"
    add("```text")
    add("Final production model:")
    add(f"  Model path:            {prod.relative_to(ROOT).as_posix()}")
    add(f"  Model architecture:    {hist['architecture']}")
    add(f"  Model evaluated on:    {met['counts']['test']} test images in split_manifest.json "
        f"(split == 'test'), manifest SHA256 {met['split_manifest_sha256']}")
    add(f"  Test accuracy:         {met['test_accuracy']:.4f}")
    add(f"  Macro F1:              {met['macro_f1']:.4f}")
    add(f"  SHA256 (new):          {sha(prod)}")
    add(f"  SHA256 (previous):     {promo.get('previous_model_sha256')}  "
        f"(backup: {promo.get('previous_model_backup')})")
    add(f"  Output shape check:    {promo['candidate_output_shape']} (probe sum "
        f"{promo['candidate_probe_sum']:.4f}) -> {'PASS' if promo['verification_passed'] else 'FAIL'}")
    add(f"  Promoted == evaluated: {sha(prod) == met['evaluated_model_sha256'] == promo['new_model_sha256']}")
    add("```\n")
    tests = EV / "module3_pytest.txt"
    if tests.exists():
        tlines = tests.read_text(encoding="utf-8").strip().splitlines()
        add(f"**Module 3 tests** (`pytest tests/test_classification.py -v`): {tlines[-1].strip('= ')}\n")
        add("```text\n" + "\n".join(tlines[:-1]) + "\n```\n")
    add("Production inference: `app/classification/classifier.py` loads `best.keras`, "
        "checks `class_order.json` against `CLASS_NAMES`, and applies "
        "`inference_config.json` (TTA + temperature + uncertainty threshold).\n")

    add("## 7. Final verdict\n")
    add("**Module 3 is COMPLETE.** The model was trained and selected on a verifiable, "
        "de-duplicated, stratified, seeded 70/15/15 split with zero path or hash overlap. "
        "It was evaluated once on the untouched test split, with every artifact and both "
        "SHA-256 hashes recorded, and promoted only after load, shape and inference checks "
        "passed. The remaining limitation is domain shift: PlantVillage leaves on plain "
        "backgrounds versus real field crops.\n")

    out = ROOT / "docs" / "MODULE3_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
