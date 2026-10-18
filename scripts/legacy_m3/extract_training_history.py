"""
FILE PATH: SmartSpray/scripts/extract_training_history.py

PURPOSE:
    Extracts the training/validation accuracy-loss progression that is embedded
    in notebooks/m3_classification_trials.ipynb (the notebook printed per-epoch
    val_accuracy / val_loss / accuracy / loss lines during training).
    The history was NOT saved to a .npy/.json file, so this is the only
    surviving record of the training curves that were shown in the notebook.

OUTPUT:
    models/classification/evaluation/training_history_from_notebook.csv
"""

import csv
import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/m3_classification_trials.ipynb")
OUT_PATH = Path("models/classification/evaluation/training_history_from_notebook.csv")

# Strip ANSI escape sequences left in notebook stdout.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Matches lines like:
#   Epoch 1/20
#   ... 2s/step - accuracy: 0.8124 - loss: 0.4667 - val_accuracy: 0.9301 ...
# DOTALL lets ".*?" span the newline between the epoch header and metrics.
EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+).*?"
    r"accuracy: ([0-9.]+) - loss: ([0-9.]+).*?"
    r"val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)",
    re.DOTALL,
)


def main() -> None:
    nb = json.loads(Path(NOTEBOOK_PATH).read_text(encoding="utf-8"))

    rows = []
    phase = 1
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = "".join(out.get("text", []))
            text = ANSI_RE.sub("", text)
            # Both training phases stream into ONE output block, so split the
            # text at the fine-tuning header ("Phase 2 - Fine-tuning ...") and
            # tag epochs before it as Phase 1, after it as Phase 2.
            segments = re.split(r"Phase 2[^\n]*Fine-tuning[^\n]*", text)
            current_phase = phase
            for seg_idx, segment in enumerate(segments):
                if seg_idx > 0:
                    current_phase = 2  # everything after the header is fine-tuning
                for m in EPOCH_RE.finditer(segment):
                    rows.append(
                        {
                            "phase": current_phase,
                            "epoch": int(m.group(1)),
                            "total_epochs": int(m.group(2)),
                            "train_accuracy": float(m.group(3)),
                            "train_loss": float(m.group(4)),
                            "val_accuracy": float(m.group(5)),
                            "val_loss": float(m.group(6)),
                        }
                    )
            phase = current_phase

    if not rows:
        print("No training history rows found in notebook.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} training epochs -> {OUT_PATH}")
    print("Best val_accuracy per phase:")
    for ph in (1, 2):
        phase_rows = [r for r in rows if r["phase"] == ph]
        if not phase_rows:
            continue
        best = max(phase_rows, key=lambda r: r["val_accuracy"])
        print(f"  Phase {ph}: epoch {best['epoch']} val_acc={best['val_accuracy']} "
              f"val_loss={best['val_loss']}")


if __name__ == "__main__":
    main()