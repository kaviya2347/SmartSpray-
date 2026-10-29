# SmartSpray 🌿

**Vision-Guided Autonomous UAV for Plant-Level Disease Detection and
Variable-Rate Spraying** (tomato crop).
SmartSpray takes a drone survey image and produces a per-plant spray plan:

```
survey image ─► M1 survey check ─► M2 plant detection ─► M3 disease classification
            ─► M4 severity ─► M5 field coordinates ─► M6 treatment decision
            ─► M7 tank mixes ─► M8 optimised route ─► M9 drone commands (+ simulator)
```

Every stage writes to SQLite, so any stage can be re-run on its own. You can
use the system through a FastAPI backend (`/docs`), a **3D drone mission
simulator** (`/simulator/`), a Streamlit dashboard, a self-contained HTML
report, or a one-command demo.

> Team 134 · VIT Chennai · B.Tech Data Science · Guide: Rajesh M
> For reviewers: **`docs/SmartSpray_Project_Documentation.docx`** is the
> complete technical documentation (architecture, every module, code,
> results, terminal guide, review Q&A; Markdown copy in
> `docs/PROJECT_DOCUMENTATION.md`). `docs/WHY_SMARTSPRAY.md` explains what
> makes the project different and why it is software first.
> `docs/MODULE3_REPORT.md` is the full Module 3 evaluation.
> `python scripts/inspect_classifier.py` shows, step by step, how the backend
> classifies one plant.

![3D mission simulator: spraying pass](docs/figures/sim_08_spray.jpg)

## The problem

Farmers usually spray a whole field on a fixed schedule, even when only a few plants are diseased. That wastes pesticide and money, adds chemical load to soil and produce, and still misses plants that need treatment early. SmartSpray works at the level of **individual plants**: it finds every plant in a drone image, diagnoses it, maps it to field coordinates, and sends the drone only to the plants that need treatment, with the right dose.

## Tech stack

Python 3.11 · TensorFlow / Keras 3 (EfficientNetB0 transfer learning, Grad-CAM) · Ultralytics YOLOv8n · OpenCV (ArUco calibration, homography, lesion segmentation) · scikit-learn · FastAPI + SQLite · three.js (3D simulator) · Streamlit · QGroundControl waypoint export · pytest

<!-- RESULTS:START -->
## Results at a glance

**Module 3 disease classifier (EfficientNetB0).** These numbers are for the
held-out test set: 752 images from a de-duplicated, stratified 70/15/15
split with seed 42. The test set was evaluated once.

| Metric | Value |
| --- | --- |
| Test accuracy | **97.34 %** (732/752) |
| Macro F1 / weighted F1 | **0.9681** / 0.9733 |
| Per-class F1 | early blight 0.921 · healthy 0.995 · late blight 0.978 · septoria 0.979 |
| Production mode (flip TTA + temperature, both chosen on validation) | 97.87 % accuracy; ECE 0.015 → 0.011 |
| Confident predictions (≥ 0.60) | 98.7 % of plants, **98.65 % correct**. The other 10 go to manual inspection. |

**End-to-end demo.** This is a synthetic 63-plant field built from
test-split photos (`python scripts/run_demo.py`).

| Stage | Result |
| --- | --- |
| Detection (M2, OpenCV fallback) | recall 1.00, precision 1.00 |
| YOLOv8n detector on real LeafPest photos (29-image test split) | mAP50 0.925 · mAP50-95 0.805 · precision 0.997 · recall 0.912 |
| Classification (M3) on field crops | 98.4 % (62/63) |
| Mapping (M5, ArUco auto-calibration) | mean position error 5.5 cm |
| Decision (M6) | 14 of 63 plants sprayed; diseased plants acted on: 19/19; healthy plants sprayed: 0/44 |
| Chemical | 112 mL vs 630 mL for blanket spraying (**82 % less**); ₹8.82 vs ₹59.85 |
| Route (M7–M8) | 38.4 m in 2 sorties vs 95 m lawn-mower pass (**60 % shorter**) |
| Simulated flight (M9) | 2.3 min, battery 3.8 %, spray placement error 5 cm, no safety violations |

Full evaluation: `docs/MODULE3_REPORT.md`. Figures: `models/classification/eval/`.
<!-- RESULTS:END -->

---

### Reproducing these numbers

The trained weights are in the repo (`models/classification/best.keras`, `models/yolov8_plant_detector/best.pt`). With the dataset in place (see *Datasets* below):

```bash
python scripts/verify_results.py     # re-scores the locked 752-image test split with the shipped model
python scripts/run_demo.py --fresh   # rebuilds the 63-plant demo field and runs M1–M9 + simulator
```

`verify_results.py` checks the model's SHA-256 against the evaluation record and every test image's SHA-256 against the split manifest, then recomputes the metrics. On 18 Oct 2026 a clean Linux run (Python 3.13, TensorFlow 2.21) reproduced every Module 3 figure above exactly (732/752 raw, 97.87 % production mode, 98.65 % on confident predictions), and the demo matched the end-to-end table.

## Datasets

The datasets are not stored in this repository because of their size. Download them into `data/datasets/`:

| Folder | Source | Used for |
|---|---|---|
| `data/datasets/plantvillage_tomato/{early_blight,healthy,late_blight,septoria_leaf_spot}/` | Tomato classes of the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) (colour images). 5,016 images, 5,007 after de-duplication | M3 classifier, demo field, simulator |
| `data/datasets/plant_detection/` | [Plant Detection v1, LeafPest on Roboflow Universe](https://universe.roboflow.com/leafpest/plant-detection-odssh/dataset/1) (CC BY 4.0), YOLOv8 export, 546 images | M2 detector |

`models/classification/split_manifest.csv` lists every image with its split and SHA-256, so you can confirm you have exactly the same files. Without the datasets, the API and the unit tests still run: the tests that need images skip themselves.

## 1. Setup (macOS / Linux / Windows)

Use Python **3.11** (3.10–3.12 also work). On a Mac without it:
`brew install python@3.11`.

```bash
git clone https://github.com/Sharank874/SmartSpray.git
cd SmartSpray
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On an Apple-silicon Mac you can optionally run `pip install tensorflow-metal`
to use the Mac GPU.

## 2. Run the full demo (one command)

```bash
python scripts/run_demo.py --fresh
```

What this does:

- **Builds a synthetic field** at `data/images/demo_survey_001/`. It is a
  10 m × 8 m plot with 63 plants. Every plant is a real PlantVillage
  test-split photo that the model never saw in training or selection. Four
  ArUco markers sit at the corners for calibration.
- **Runs Modules 1–9**, then the flight simulator.
- **Prints a summary:** detection, classification, mapping, chemical saved,
  route, simulation, and a check against the ground truth.
- **Writes the outputs** to `data/outputs/demo_survey_001/`:

| File | Contents |
| --- | --- |
| `report.html` | Open in any browser |
| `gradcam/plant_###.jpg` | Explainability heatmaps |
| `mission/sortie_##_*.waypoints` | Opens in QGroundControl / Mission Planner |
| `mission/drone_commands.json` | Command queue for the executor |
| `mission/simulation_log.json` | Software-in-the-loop flight log |
| `pipeline_result.json` | Full result of every stage |

The demo uses the OpenCV fallback detector because the YOLO model is
trained on real field photos, not on the synthetic mosaic.

## 3. Drone mission simulator (3D, with pilot override)

```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/simulator/      (add ?quality=low on an older laptop)
```

Pick a field, press **Prepare mission**, then **Start mission**. The drone:

1. flies to each field corner and **drops an ArUco marker**;
2. climbs to 15 m, photographs the field, reads the markers (**boundary and
   size**) and **locates every plant** in metres;
3. flies low and **photographs each plant**; each photo goes through the real
   disease model, severity estimate and dosage rules;
4. skips plants that need no spray and computes the **shortest
   capacity-aware route** over the rest;
5. **sprays each target with its own dose**, refilling between tank loads;
6. lands and shows the savings report (chemical, route, battery, ground-truth check).

Press **M** at any time to fly it yourself (W A S D, Q E turn, R F height,
Space spray, H return home). Press **M** again and the autopilot re-plans the
shortest route from where the drone is. Details: `docs/SIMULATOR.md`.

## 4. Run the API

```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

Typical flow with curl, using a real survey:

```bash
# 1. put 5-10 .jpg/.png images in data/images/survey_001/
curl -X POST localhost:8000/survey/register -H "Content-Type: application/json" \
     -d '{"survey_id":"survey_001","field_width_m":10,"field_height_m":8}'

# 2. run everything (detector: yolo | auto | classical)
curl -X POST localhost:8000/pipeline/run -H "Content-Type: application/json" \
     -d '{"survey_id":"survey_001","detector":"auto","save_gradcam":true,"simulate":true}'

# 3. look at the results
open http://127.0.0.1:8000/reports/survey_001
```

Or call the modules one at a time:

| Module | Endpoint |
| --- | --- |
| M1 | `POST /survey/register`, `GET /survey/{id}` |
| M2 | `POST /detect` (`{"survey_id", "detector"}`), `GET /detect/health` |
| M3 | `POST /classify` (`{"survey_id", "save_gradcam"}`), `GET /classify/results/{id}`, `GET /classify/health` |
| M4 | `POST /severity` |
| M5 | `POST /spatial/calibrate`, `POST /spatial/map`, `GET /spatial/locations/{id}` |
| M6 | `POST /decisions/generate`, `GET /decisions/{id}`, `GET /decisions/rules/current` |
| M7 | `GET /resources/{id}`, `GET /resources/config` |
| M8 | `POST /mission/plan`, `GET /mission/{id}` |
| M9 | `POST /drone/commands/generate`, `GET /drone/commands/{id}`, `GET /drone/mission/{id}/{sortie}.waypoints`, `POST /drone/simulate` |

You can also run the pipeline from the command line:

```bash
python -m app.pipeline demo_survey_001 --detector classical --gradcam --simulate --report
```

## 5. Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard has five pages:

- **Field overview** — map, key numbers and per-plant decisions
- **Explainability** — Grad-CAM gallery
- **Mission** — sorties, tank mixes, waypoint downloads and the simulator
- **Model card** — Module 3 metrics and figures
- **New survey** — upload images and run the pipeline

## 6. Tests

```bash
pytest -q
```

Tests that need model weights or the dataset skip themselves when those are missing. Current status: **277 passed** with the datasets in place, and 271 passed + 6 skipped without them.

## 7. Re-train the models

### Module 3: disease classifier (EfficientNetB0, 70/15/15)

The dataset lives in `data/datasets/plantvillage_tomato/<class>/` and the
ImageNet backbone in `models/pretrained/`.

```bash
python scripts/m3_train_classifier.py --stage split      # de-duplicate + seeded stratified split + manifest
python scripts/m3_train_classifier.py --stage train      # phase 1 (frozen) + phase 2 (top-30 fine-tune)
python scripts/m3_train_classifier.py --stage deep       # phase 3 (blocks 5-7), accepted only if validation improves
python scripts/m3_train_classifier.py --stage evaluate   # the single test pass (locked afterwards)
python scripts/m3_train_classifier.py --stage promote    # backup + SHA256 + copy to best.keras
python scripts/build_m3_report_notebook.py               # notebooks/m3_classification_701515_report.ipynb
```

Training takes about 1.5 h on a 2-core CPU and much less on an M-series Mac.
The test split is locked by `models/classification/eval/TEST_EVALUATED.lock`.
Delete that file only if you are deliberately starting a new, documented
experiment.

### Module 2: plant detector (YOLOv8n)

Copy the Roboflow export "Plant Detection.v1i.yolov8" to
`data/datasets/plant_detection/`, then run:

```bash
python scripts/m2_train_detector.py --epochs 50 --imgsz 640 --device mps   # or --device cpu
```

This writes `models/yolov8_plant_detector/best.pt` plus validation and test
metrics. The included `best.pt` was trained on CPU at 512 px for 30 epochs.
Its test split scores mAP50 0.925 and mAP50-95 0.805; details are in
`models/yolov8_plant_detector/eval/metrics.json`.

## 8. Project layout

```
app/
  field_survey/        M1  registration + blur/readability check
  detection/           M2  YOLOv8 detector, OpenCV fallback, plant IDs
  classification/      M3  EfficientNetB0 classifier, calibration/TTA, Grad-CAM
  severity/            M4  OpenCV lesion-area severity
  spatial_mapping/     M5  ArUco homography (pixels -> metres)
  decision/            M6  rule engine + decision tables
  resource_optimization/ M7 tank mixes, loads, cost
  mission_planner/     M8  nearest-neighbour + 2-opt capacity-aware routing
  drone_integration/   M9  command queue + QGroundControl export
  simulation/          3D mission simulator backend (synthetic field, camera, /sim API)
  pipeline.py          runs M1-M9;  reporting.py  HTML report
dashboard/             Streamlit app
simulator/             software-in-the-loop drone executor
simulator/web/         3D mission simulator page (three.js, served at /simulator/)
rules/                 dosage_rules.json, mission_config.json (auditable settings)
scripts/               training, demo, evaluation (scripts/legacy_m3 = superseded)
models/classification/ best.keras, class_order.json, inference_config.json,
                       split_manifest.json, eval/ (metrics, figures, provenance)
models/pretrained/     EfficientNetB0 ImageNet backbone weights
data/datasets/         plantvillage_tomato/, plant_detection/
data/images/<survey>/  survey images;  data/outputs/<survey>/  results
docs/                  WHY_SMARTSPRAY.md, MODULE3_REPORT.md, SIMULATOR.md, module limitations
tests/                 pytest suite
```

## 9. Limitations

- The classifier was trained on PlantVillage leaf photos taken against plain
  backgrounds. On real greenhouse photos it over-predicts disease, and
  Grad-CAM shows it looking at the background. Real field crops need
  fine-tuning (for example PlantDoc or your own labelled crops).
- Severity (M4) is an OpenCV heuristic and has not been checked against
  labelled severity data.
- Dosages and prices in `rules/` are indicative placeholders.
- Drone output has only been validated in simulation.
- The demo field and the simulator's fields are synthetic (no wind, GPS
  error or obstacles in the simulator).
