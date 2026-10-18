"""
Builds notebooks/m3_classification_701515_report.ipynb — a read-only
presentation notebook for the Module 3 rebuild. It only LOADS the saved
artifacts (manifest, history, metrics, figures); it never trains and never
re-evaluates the test split.

    python scripts/build_m3_report_notebook.py            # build + execute
"""

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "notebooks" / "m3_classification_701515_report.ipynb"

cells = [
    nbf.v4.new_markdown_cell(
        "# SmartSpray — Module 3 Disease Classification (70 / 15 / 15 rebuild)\n\n"
        "This notebook **only reads** the artifacts written by "
        "`scripts/m3_train_classifier.py`. It does not train and it does not touch the "
        "test split again.\n\n"
        "| Item | Value |\n|---|---|\n"
        "| Dataset | PlantVillage tomato subset (early blight, healthy, late blight, septoria) |\n"
        "| Split | de-duplicated, stratified, `random_state=42`, 70 / 15 / 15 |\n"
        "| Model | EfficientNetB0 (ImageNet) → GAP → Dropout 0.3 → Dense 256 → Dropout 0.2 → Dense 4 |\n"
        "| Selection | validation split only |\n"
        "| Test | evaluated once (lock file `models/classification/eval/TEST_EVALUATED.lock`) |"),
    nbf.v4.new_code_cell(
        "import json, hashlib\nfrom pathlib import Path\nimport pandas as pd\n"
        "from IPython.display import Image, display, Markdown\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "MODEL_DIR = ROOT / 'models' / 'classification'\nEVAL = MODEL_DIR / 'eval'\n"
        "manifest = json.loads((MODEL_DIR / 'split_manifest.json').read_text())\n"
        "metrics = json.loads((EVAL / 'metrics.json').read_text())\n"
        "history = json.loads((EVAL / 'training_history.json').read_text())\n"
        "promotion = json.loads((EVAL / 'promotion_record.json').read_text())\n"
        "print('Artifacts loaded from', EVAL)"),
    nbf.v4.new_markdown_cell("## 1. Dataset audit and split"),
    nbf.v4.new_code_cell(
        "print('Raw image files :', manifest['raw_file_total'], manifest['raw_file_counts'])\n"
        "print('Exact duplicates removed :', len(manifest['excluded']['exact_duplicates']))\n"
        "print('Unique images   :', manifest['unique_images'])\n"
        "print('Seed / method   :', manifest['random_seed'], '|', manifest['split_method'])\n"
        "df = pd.DataFrame(manifest['split_counts']).T\ndf['total'] = df.sum(axis=1)\n"
        "df.loc['all'] = df.sum()\ndisplay(df)"),
    nbf.v4.new_code_cell(
        "checks = manifest['leakage_checks']\n"
        "display(pd.Series(checks, name='value').to_frame())\n"
        "assert all(v == 0 for k, v in checks.items() if 'overlap' in k)\n"
        "print('Zero overlap between train / validation / test (by path and by SHA-256).')"),
    nbf.v4.new_code_cell(
        "pd.DataFrame(manifest['excluded']['exact_duplicates'])[['path', 'duplicate_of']]"),
    nbf.v4.new_markdown_cell(
        "## 2. Training (validation-only model selection)\n\n"
        "Phase 1 trains the head with the backbone frozen. Phase 2 fine-tunes the top 30 "
        "backbone layers. Phase 3 (pre-registered) fine-tunes blocks 5–7 and is accepted "
        "only if validation improves. BatchNorm stays frozen throughout."),
    nbf.v4.new_code_cell(
        "print('Backbone weights:', history['backbone_weights_source'].split('/')[-1])\n"
        "display(pd.Series(history['hyperparameters']).to_frame('value'))\n"
        "rows = []\nfor ph in ('phase1', 'phase2', 'phase3'):\n"
        "    if ph in history:\n"
        "        h = history[ph]['history']\n"
        "        rows.append({'phase': ph, 'epochs': history[ph]['epochs_run'],\n"
        "                     'minutes': round(history[ph]['minutes'], 1),\n"
        "                     'best val_acc': max(h['val_accuracy']),\n"
        "                     'min val_loss': min(h['val_loss'])})\n"
        "display(pd.DataFrame(rows))\n"
        "print('Selected checkpoint:', history['selected_checkpoint'])\n"
        "display(Image(str(EVAL / 'training_curves.png')))"),
    nbf.v4.new_code_cell(
        "print('TTA decision (validation):', history['validation_inference_decision'])\n"
        "print('Calibration (validation):', history['calibration'])"),
    nbf.v4.new_markdown_cell("## 3. Independent test evaluation (single pass)"),
    nbf.v4.new_code_cell("print((EVAL / 'test_evaluation_printout.txt').read_text())"),
    nbf.v4.new_code_cell(
        "display(Image(str(EVAL / 'confusion_matrix.png')))\n"
        "display(Image(str(EVAL / 'reliability_diagram.png')))"),
    nbf.v4.new_code_cell("pd.read_csv(EVAL / 'misclassified.csv')"),
    nbf.v4.new_markdown_cell("## 4. Explainability (Grad-CAM on test images)"),
    nbf.v4.new_code_cell("display(Image(str(EVAL / 'gradcam_examples.png')))"),
    nbf.v4.new_markdown_cell("## 5. Production model provenance"),
    nbf.v4.new_code_cell(
        "prod = MODEL_DIR / 'best.keras'\n"
        "sha = hashlib.sha256(prod.read_bytes()).hexdigest()\n"
        "print('best.keras SHA256        :', sha)\n"
        "print('promotion record (new)   :', promotion['new_model_sha256'])\n"
        "print('evaluated model SHA256   :', metrics['evaluated_model_sha256'])\n"
        "print('previous model (backup)  :', promotion.get('previous_model_sha256'),\n"
        "      '->', promotion.get('previous_model_backup'))\n"
        "assert sha == promotion['new_model_sha256'] == metrics['evaluated_model_sha256']\n"
        "print('The production model is exactly the model that was evaluated.')"),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", OUT)

if __name__ == "__main__":
    from nbclient import NotebookClient

    nb = nbf.read(OUT, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print("executed", OUT)
