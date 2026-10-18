# Executive summary

SmartSpray is a software system for **plant-level, disease-aware precision spraying** of tomato crops by drone. It takes a survey image of a field and produces, for every plant, a diagnosis (disease and confidence), a severity estimate, a position in metres, a justified treatment decision with its own dose, and finally a shortest-route spray mission that a drone can fly. Plants that do not need treatment are skipped, so chemical use and flight distance fall sharply compared with spraying the whole field.

The system is built as nine modules (M1–M9) that each read from and write to one SQLite database, so any stage can be re-run, inspected or replaced on its own. It can be used through a REST API, a command-line pipeline, a Streamlit dashboard, a self-contained HTML report and a **3D drone mission simulator** in which the drone marks the field, maps it, photographs every plant, decides and sprays autonomously, and can be taken over by a pilot from the keyboard at any moment.

**Key results**

| Item | Result |
|---|---|
| Disease classifier (M3), held-out test set of 752 images | **97.34 %** accuracy (732/752), macro F1 **0.968** |
| Production inference mode (flip-TTA + temperature scaling) | 97.87 % accuracy, calibration error (ECE) 0.015 → 0.011 |
| Uncertainty gate (confidence < 0.60 → manual inspection) | 98.7 % of plants confident, 98.65 % of those correct |
| Plant detector (M2, YOLOv8n, 29-image test split) | mAP50 0.925 · mAP50-95 0.805 · precision 0.997 · recall 0.912 |
| End-to-end demo field (63 plants, synthetic) | 82 % less chemical than blanket spraying; route 60 % shorter; 19/19 diseased plants acted on; 0/44 healthy plants sprayed |
| 3D mission simulator (63-plant field) | 63/63 plants located (4.3 cm error), 61/63 classified correctly, 80.6 % less chemical, spray route 47 m vs 99 m lawn-mower pass (52 % shorter) |
| Automated tests | 277 passing |

**What this document contains.** Chapter 2 explains the problem and objectives. Chapter 3 gives the architecture. Chapter 4 covers the data. Chapters 5–13 describe each module in depth: purpose, algorithm, code, database tables, API, terminal commands and results. Chapter 7 (Module 3) explains exactly how a plant is classified inside the backend, step by step, with the real code and a worked example. Chapter 14 covers pipeline orchestration and storage, Chapter 15 the interfaces, Chapter 16 the 3D mission simulator, Chapter 17 testing, Chapter 18 the consolidated results, and Chapter 19 a complete terminal guide for running and opening every module. Chapter 20 lists likely review questions with answers. Chapter 21 lists limitations and the remaining work, followed by references and appendices (file-by-file index, configuration files, glossary).

> How to use this for the consolidated report: Chapters 1–4 map to Introduction / Literature / System Design; Chapters 5–16 to Implementation; Chapters 17–18 to Results and Testing; Chapter 21 to Conclusion and Future Work. Every number in this document comes from files in the repository (paths are given), so it can be checked during the review.

# Introduction

## Problem statement

Tomato is one of the most widely grown vegetable crops in India, and fungal and oomycete leaf diseases are among the main causes of yield loss. Three of the most damaging are:

- **Early blight** (*Alternaria solani*): brown concentric-ring lesions, usually on older leaves.
- **Late blight** (*Phytophthora infestans*): dark, water-soaked patches that spread very quickly in cool, humid weather and can destroy a crop in days.
- **Septoria leaf spot** (*Septoria lycopersici*): many small circular spots with pale centres.

Farmers usually respond by spraying the whole field on a fixed schedule ("blanket spraying"). Disease, however, appears in patches. Spraying every plant wastes chemical and money, increases residue on produce, harms soil and water, and exposes the operator. Agricultural drones make spraying faster, but most of them still spray uniformly.

Existing research models stop at the diagnosis of a single leaf photo, and commercial precision sprayers mostly target weeds and come as expensive machines. What is missing is the link from **an image of a field** to **a justified, plant-specific spray action** that a drone can execute.

## Objectives

1. Detect individual tomato plants in a drone survey image and give each one a stable ID.
2. Classify each plant as healthy, early blight, late blight or septoria leaf spot, with a **calibrated** confidence and a visual explanation (Grad-CAM).
3. Estimate the severity of the disease from the lesion area.
4. Convert every plant's pixel position into field coordinates in metres.
5. Decide, with an auditable rule table, whether each plant should be sprayed, monitored, inspected or left alone, and how much chemical it needs. Never spray automatically when the model is unsure.
6. Plan the shortest spray route over only the plants that need treatment, respecting the drone's tank capacity.
7. Export the mission in a standard drone format (MAVLink / QGroundControl) and validate it in simulation.
8. Measure the savings (chemical, cost, distance, time) against blanket spraying.
9. Demonstrate the whole mission in a 3D simulator with autonomous operation and a manual pilot override.

## Scope of the current build (≈ 75 %)

| Area | Status |
|---|---|
| M1 Field survey (registration, image quality check) | Done |
| M2 Plant detection (YOLOv8n + OpenCV fallback) | Done |
| M3 Disease classification (70/15/15, calibration, TTA, Grad-CAM) | Done |
| M4 Severity estimation (OpenCV lesion ratio) | Done (heuristic; not validated against labelled severity) |
| M5 Spatial mapping (ArUco homography, auto-calibration) | Done |
| M6 Decision engine (rule table, uncertainty gate, re-treatment) | Done |
| M7 Resource optimisation (tank mixes, loads, cost) | Done |
| M8 Mission planning (nearest neighbour + 2-opt, capacity-aware) | Done |
| M9 Drone commands (JSON queue, QGC waypoints, SITL executor) | Done (simulation only) |
| API, HTML report, dashboard, 3D mission simulator, tests | Done |
| **Remaining ≈ 25 %** | Field imagery and field-trained models; severity validation with agronomists; live MAVLink link and real pump; RTK-GPS; weather-aware spray windows; multi-field accounts |

## Why software first, not hardware

1. **The new part is the decision, not the machine.** Pumps, nozzles and drones are mature products. What is missing is knowing which plant to spray, with what, how much, and why, and that intelligence is software.
2. **Hardware-agnostic output.** SmartSpray produces standard MAVLink missions (`QGC WPL 110`) and a plain JSON command queue. The same plan can drive an ArduPilot/PX4 drone, a ground robot, or a person with a knapsack sprayer and a phone.
3. **Cost and regulation.** A spray drone costs roughly ₹3–5 lakh for 10–16 L models. Flying one for agriculture in India needs a DGCA Remote Pilot Certificate, a type-certified drone registered on Digital Sky, and compliance with the agriculture ministry's guidelines for pesticide application by drone. A simulator gives the same validation signals (route, tank, battery, placement error) safely and repeatably.
4. **Reproducible and testable.** 277 automated tests run in about a minute; a hardware prototype cannot be re-run identically for a review panel.
5. **Scales by copying.** A software update improves every field it runs on.
6. **Hardware-ready by design.** Swapping ArUco markers for RTK-GPS changes only Module 5; Module 9's command queue is the integration point for a real flight controller.

## What makes the project different

| Question | Typical existing work | SmartSpray |
|---|---|---|
| Which plant is sick? | Leaf-level or whole-field | Every plant has an ID and a position in metres |
| How sick is it? | Rarely estimated | Lesion-area severity graded low / medium / high |
| What should be done, and why? | Diagnosis only | Auditable rule table (`rules/dosage_rules.json`) plus a written reason for every plant |
| What if the AI is unsure? | Prediction used anyway | Calibrated confidence; below 0.60 the plant goes to manual inspection and is never sprayed automatically |
| Can the accuracy claim be trusted? | Often measured on overlapping splits | De-duplicated, seeded, stratified 70/15/15 split with a manifest; test set used once; SHA-256 hashes recorded |
| Why did the model decide that? | Black box | Grad-CAM heatmap for every plant |
| How much is saved? | Rarely measured | Chemical (mL), cost (₹), distance and time vs blanket spraying |
| Can a drone use the output? | No | JSON command queue + QGroundControl `.waypoints` (MAVLink `DO_SPRAYER`), tested in two simulators |
| Can a pilot take over? | Rarely modelled | Keyboard override at any time; on hand-back the autopilot re-plans from the drone's position and remaining tank |
| Does it avoid double spraying? | No | Re-treatment interval check by field position |

SmartSpray deliberately does not claim to beat every published model on accuracy: PlantVillage accuracies above 95 % are common in the literature. Its contribution is the **complete, measurable, explainable system** from survey image to drone mission.

# System architecture

## End-to-end pipeline

![End-to-end SmartSpray pipeline (M1–M9) with storage, interfaces and simulation](docs/figures/fig_pipeline.png)

A survey (one or more images of a field) flows through nine modules. Each module reads its inputs from SQLite and writes its outputs back, so the stages are loosely coupled:

| Module | Purpose | Method | Main code | Writes table |
|---|---|---|---|---|
| M1 Field survey | Register a survey and check image quality | Image listing, variance-of-Laplacian blur check | `app/field_survey/service.py` | `surveys`, `survey_images` |
| M2 Plant detection | Find every plant and give it an ID | YOLOv8n (trained) or OpenCV vegetation segmentation; adaptive row ordering | `app/detection/` | `detections` |
| M3 Disease classification | Disease + calibrated confidence per plant | EfficientNetB0 transfer learning, flip-TTA, temperature scaling, Grad-CAM | `app/classification/` | `classifications` |
| M4 Severity | Fraction of leaf area that is diseased | OpenCV leaf and lesion segmentation | `app/severity/` | `plant_severity` |
| M5 Spatial mapping | Pixel → field metres | ArUco markers + homography | `app/spatial_mapping/` | `spatial_calibrations`, `plant_locations` |
| M6 Treatment decision | Action, dose and reason per plant | Ordered rule table from JSON | `app/decision/` | `treatment_decisions` |
| M7 Resource optimisation | Tank mixes, loads, cost | Group by treatment, capacity arithmetic | `app/resource_optimization/` | part of `mission_plans` |
| M8 Mission planning | Shortest capacity-aware route | Nearest neighbour + 2-opt, route-first/cluster-second | `app/mission_planner/` | `mission_plans` |
| M9 Drone integration | Executable mission | JSON command queue + QGroundControl waypoints; SITL executor | `app/drone_integration/`, `simulator/` | `drone_commands`, `execution_logs` |

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (backend, ML); JavaScript ES modules (simulator front end) |
| Deep learning | TensorFlow 2.21 / Keras 3.15 (EfficientNetB0); Ultralytics YOLOv8 8.4 |
| Computer vision | OpenCV (ArUco, homography, segmentation, morphology) |
| Data / ML utilities | NumPy, pandas, scikit-learn (stratified splitting, metrics), SciPy, Matplotlib |
| API | FastAPI + Uvicorn, Pydantic request validation |
| Storage | SQLite (`database/smartspray.db`) |
| Interfaces | Streamlit dashboard, self-contained HTML report, Jupyter report notebook |
| 3D simulator | three.js r186 (bundled locally, no internet needed), WebGL |
| Drone formats | MAVLink mission file `QGC WPL 110` (QGroundControl / Mission Planner) |
| Testing | pytest (277 tests), headless Chromium (Playwright) for simulator run-throughs |

## Repository layout

```text
SmartSpray-main/
├── app/                         backend package (FastAPI app + all modules)
│   ├── main.py                  FastAPI app: routers, /pipeline/run, /reports
│   ├── pipeline.py              runs M1 → M9 for a survey (also a CLI)
│   ├── db.py                    SQLite connection + table creation
│   ├── reporting.py             self-contained HTML report
│   ├── field_survey/            M1
│   ├── detection/               M2  detector.py, identity_resolver.py, router.py
│   ├── classification/          M3  classifier.py, model_factory.py, explain.py, service.py
│   ├── severity/                M4  estimator.py, service.py, router.py
│   ├── spatial_mapping/         M5  calibration.py, mapper.py, router.py
│   ├── decision/                M6  rules_engine.py, service.py, database.py
│   ├── resource_optimization/   M7  grouping.py, router.py
│   ├── mission_planner/         M8  route_optimizer.py, service.py, router.py
│   ├── drone_integration/       M9  command_queue.py, router.py
│   └── simulation/              3D simulator backend: world.py, service.py, router.py
├── simulator/
│   ├── executor_simulator.py    software-in-the-loop executor for M9 commands
│   └── web/                     3D mission simulator page (index.html, js/, vendor/)
├── dashboard/streamlit_app.py   Streamlit dashboard (5 pages)
├── rules/                       dosage_rules.json, mission_config.json
├── models/
│   ├── classification/          best.keras, class_order.json, inference_config.json,
│   │                            split_manifest.json/.csv, eval/ (metrics, figures, logs)
│   ├── yolov8_plant_detector/   best.pt, eval/metrics.json
│   └── pretrained/              EfficientNetB0 ImageNet weights
├── data/
│   ├── datasets/                plantvillage_tomato/, plant_detection/
│   ├── images/<survey_id>/      survey images
│   └── outputs/<survey_id>/     report.html, gradcam/, mission/, logs
├── scripts/                     training, demo, inspection and housekeeping scripts
├── notebooks/                   executed Module 3 report notebook
├── docs/                        this document, MODULE3_REPORT.md, SIMULATOR.md, ...
├── tests/                       pytest suite
├── database/                    smartspray.db, schema.sql
└── requirements.txt, README.md
```

## Data flow and storage

Every module is a small service with three layers: a **pure algorithm** file (no database, easy to unit-test), a **service** file that loads inputs from SQLite, runs the algorithm and saves results, and a **router** file that exposes the service over HTTP. The pipeline (`app/pipeline.py`) calls the services in order, but each one can also be called alone. Because results are stored per `survey_id`, re-running a stage simply replaces that stage's rows.

## Coordinate frames

| Frame | Units | Origin and axes | Used by |
|---|---|---|---|
| Image | pixels | top-left corner, x right, y down | M2 boxes, M3/M4 crops |
| Field | metres | marker 0 (south-west corner), x east, y north | M5 output, M6–M8, simulator |
| GPS | degrees | WGS-84 latitude/longitude; field origin at `origin_lat`, `origin_lon` in `rules/mission_config.json` | M9 waypoint files |

M5 converts image → field with a homography. M9 converts field → GPS with a local flat-earth approximation: latitude = lat₀ + y / 111 320, longitude = lon₀ + x / (111 320 · cos lat₀).

# Datasets and data preparation

## PlantVillage tomato subset (Module 3)

The disease classifier is trained on four tomato classes from the public PlantVillage dataset: single leaves photographed on a plain background (256 × 256 px). The data lives in `data/datasets/plantvillage_tomato/<class>/`.

| Class | Raw files |
|---|---|
| early_blight | 800 |
| healthy | 1,273 |
| late_blight | 1,526 |
| septoria_leaf_spot | 1,417 |
| **Total** | **5,016** |

## De-duplication

Every file is hashed with SHA-256. Nine files were exact byte-for-byte duplicates of another file (four healthy, five late blight). They were removed **before** splitting, because a duplicate pair split across train and test would leak test images into training. After de-duplication: **5,007 unique images**. Unreadable files, label conflicts and non-image files were also checked for (none found).

## The 70 / 15 / 15 split

The split is made by `scripts/m3_train_classifier.py --stage split` with scikit-learn's `train_test_split` called twice, both with `stratify=labels` and `random_state=42`:

1. 70 % train vs 30 % hold-out.
2. Hold-out split 50 / 50 into validation and test.

| Split | early_blight | healthy | late_blight | septoria | Total |
|---|---|---|---|---|---|
| Train (70 %) | 560 | 888 | 1,064 | 992 | **3,504** |
| Validation (15 %) | 120 | 191 | 228 | 212 | **751** |
| Test (15 %) | 120 | 190 | 229 | 213 | **752** |

**Leakage checks** (all recorded in `models/classification/split_manifest.json`): path overlap and SHA-256 overlap between every pair of splits = 0; every unique image appears in exactly one split; the dropped duplicates are absent from all splits. The manifest lists every file with its split, label and hash (also as CSV), and its own SHA-256 is stored with the evaluation results so the exact split can be verified later.

**How each split is used.** Train: fitting weights, with augmentation. Validation: early stopping, learning-rate schedule, checkpoint selection, and the choice of inference settings (TTA, temperature). Test: evaluated **exactly once** after everything was frozen; a lock file (`models/classification/eval/TEST_EVALUATED.lock`) then stops any further training on the same split.

> Why the old numbers were replaced: the earlier notebook used an 80/20 train/validation split with no test set, and the model was selected on that same validation set. Its 98.4 % / 96.68 % figures were therefore not held-out results. The 97.34 % reported here is a single-use, held-out test result.

## Augmentation (training only)

Applied on the fly to training batches only, seeded with 42: random horizontal flip, random rotation ±20°, random translation ±15 %, random zoom ±15 %, random brightness ±20 %. Validation and test images are never augmented.

## Plant-detection dataset (Module 2)

LeafPest "Plant Detection" v1 (Roboflow, CC BY 4.0), YOLOv8 export, one class `Tomato-Plant`: 460 train, 57 validation and 29 test images of real greenhouse/field tomato plants. Stored in `data/datasets/plant_detection/`.

## Synthetic fields (demo and simulator)

There is no labelled drone imagery of a diseased tomato field with markers, so the end-to-end demo and the simulator use **synthetic fields** built from real photos:

- Every plant is a real PlantVillage photo taken **only from the test split**, so the classifier has never seen any of them in training or model selection.
- Plants are laid out in rows on a generated soil texture (furrows, noise, lighting gradient, sensor noise, slight perspective tilt).
- Disease appears in patches: the scripted demo field (10 m × 8 m, 7 rows × 9 plants = 63) has a late-blight hotspot (6 plants), an early-blight patch (7) and a septoria patch (6); other simulator fields use randomly grown patches (~9 % of plants per disease).
- Four ArUco markers (ids 0–3) sit at the corners.
- A `ground_truth.json` (demo) or the session state (simulator) records each plant's true label and position, so the system's output can be scored.

![The synthetic demo field: 63 real test-split leaf photos, four ArUco corner markers](docs/figures/demo_field.jpg)

The synthetic field demonstrates and measures the pipeline; it is not a field trial (see Chapter 21).
