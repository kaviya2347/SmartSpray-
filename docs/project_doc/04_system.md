# Pipeline orchestration and storage

## The pipeline

`app/pipeline.py` runs all stages for a registered survey, in order, and returns one result dictionary with a status per stage:

| Stage | Call | Skips / stops when |
|---|---|---|
| 1 Survey | read `surveys` row | survey not registered → error |
| 2 Detection | `get_detector(mode).detect()` per image → `resolve_survey_detections()` → save | no images |
| 3 Classification | `classify_survey(survey_id, save_gradcam)` | no detections |
| 4 Severity | `estimate_survey()` → save each result | no classifications |
| 5 Mapping | auto-calibrate from the survey's own images if needed → `map_survey()` | — (uncalibrated plants get NULL positions) |
| 6 Decisions | `generate_decisions()` | no classifications |
| 7 + 8 Resources + route | `plan_mission()` | no decisions |
| 9 Commands | `generate_commands()`; optionally the SITL executor | plan not flyable |
| Report | `write_survey_report()` → `data/outputs/<id>/report.html` | optional |

A failing stage records its error and the remaining stages still run where they can, so one problem never hides the others. The same function backs `POST /pipeline/run` and the command line:

```bash
python -m app.pipeline demo_survey_001 --detector classical --gradcam --simulate --report
python -m app.pipeline demo_survey_001 --json          # print the full JSON result
```

## Database

One SQLite file, `database/smartspray.db` (path in `app/db.py`; 30-second lock timeout; the folder is created on first use; `init_db()` creates every table at API start-up). All tables are keyed by `survey_id` and, where relevant, `plant_id`.

| Table | Written by | Columns |
|---|---|---|
| `surveys` | M1 | survey_id, capture_date, crop, field_size, num_images, created_at |
| `survey_images` | M1 | survey_id, image_filename, width_px, height_px, blur_score, is_readable, is_blurry, checked_at |
| `detections` | M2 | detection_id, survey_id, image_filename, plant_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, model_version, created_at |
| `classifications` | M3 | classification_id, survey_id, image_filename, plant_id, bbox_x1..y2, label, confidence, all_probs_json, model_version, is_uncertain, created_at |
| `plant_severity` | M4 | severity_id, survey_id, plant_id, image_filename, disease_label, severity_score, severity_category, leaf_px, lesion_px, is_valid, model_version, estimated_at |
| `spatial_calibrations` | M5 | calibration_id, survey_id, image_filename, field_width_m, field_height_m, aruco_dict, marker_count, homography_json, reprojection_error, is_valid, calibrated_at |
| `plant_locations` | M5 | location_id, survey_id, plant_id, image_filename, bbox_cx_px, bbox_cy_px, field_x_m, field_y_m, is_mapped, calibration_id, mapping_mode, model_version, mapped_at |
| `treatment_decisions` | M6 | decision_id, survey_id, plant_id, image_filename, disease_label, confidence, is_uncertain, severity_score, severity_category, severity_valid, field_x_m, field_y_m, is_mapped, action, dosage_ml, treatment, reason, rules_version, decided_at |
| `mission_plans` | M7 + M8 | survey_id, status, plan_json, created_at |
| `drone_commands` | M9 | command_id, survey_id, seq, command_json, status, created_at, executed_at |
| `execution_logs` | SITL executor | log_id, survey_id, seq, event, detail_json, logged_at |

Module 6 gathers its evidence with one LEFT JOIN of detections, classifications, severity and locations, so a missing downstream result shows up as NULL instead of silently dropping the plant.

## Output files per survey

```text
data/outputs/<survey_id>/
├── report.html                      self-contained HTML report (images embedded)
├── pipeline_result.json             full result of every stage
├── demo_ground_truth_evaluation.json   (synthetic demo only) scores vs ground truth
├── gradcam/plant_###.jpg            Grad-CAM overlays
└── mission/
    ├── drone_commands.json          command queue
    ├── sortie_##_<treatment>.waypoints   QGroundControl / Mission Planner files
    └── simulation_log.json          SITL executor log
```

# Interfaces

## REST API (FastAPI)

Start with `uvicorn app.main:app --reload` and open `http://127.0.0.1:8000/docs` for interactive documentation where every endpoint can be tried in the browser.

| Method | Path | Module | Purpose |
|---|---|---|---|
| GET | `/`, `/health` | — | service info / health check |
| POST | `/survey/register` | M1 | register a survey (`survey_id`, `crop`, `field_width_m`, `field_height_m`) |
| GET | `/survey/{survey_id}` | M1 | survey record |
| POST | `/detect` | M2 | detect plants (`survey_id`, `detector`: yolo / auto / classical) |
| GET | `/detect/health` | M2 | detector status |
| POST | `/classify` | M3 | classify all detected plants (`survey_id`, `save_gradcam`) |
| GET | `/classify/results/{survey_id}` | M3 | stored classifications |
| GET | `/classify/health` | M3 | model version, temperature, TTA, threshold |
| POST | `/severity` | M4 | estimate severity for every classified plant |
| GET | `/severity/health` | M4 | estimator status |
| POST | `/spatial/calibrate` | M5 | calibrate from one image with the four markers |
| POST | `/spatial/map` | M5 | map every plant to metres |
| GET | `/spatial/locations/{survey_id}` | M5 | stored positions |
| GET | `/spatial/health` | M5 | mapping status |
| POST | `/decisions/generate` | M6 | decide every plant |
| GET | `/decisions/{survey_id}` · `/decisions/rules/current` | M6 | decisions · active rule table |
| GET | `/resources/{survey_id}` · `/resources/config` | M7 | tank mixes · mission configuration |
| POST | `/mission/plan` | M8 | plan the route (optional tank / speed / flow overrides) |
| GET | `/mission/{survey_id}` | M8 | stored plan |
| POST | `/drone/commands/generate` | M9 | command queue + waypoint files |
| GET | `/drone/commands/{survey_id}` | M9 | command queue |
| GET | `/drone/mission/{survey_id}/{sortie}.waypoints` | M9 | download a QGroundControl file |
| POST | `/drone/simulate` | M9 | run the SITL executor (`gnss_noise_m`, `seed`) |
| POST | `/pipeline/run` | all | run M1–M9 (`survey_id`, `detector`, `save_gradcam`, `simulate`, `write_report`) |
| GET | `/reports/{survey_id}` | — | the HTML report |
| GET | `/simulator/` | sim | 3D mission simulator page |
| POST / GET | `/sim/...` | sim | simulator steps (Chapter 16) |

Every request body is validated by Pydantic (wrong types return HTTP 422), unknown surveys return 404, missing prerequisites return 400 with a message naming the module to run first, and file-serving endpoints reject path traversal.

![Interactive API documentation at /docs](docs/figures/api_docs.png)

## HTML report

`app/reporting.py` writes one self-contained `report.html` per survey (images embedded, works offline, light and dark mode): key-number tiles (plants, sprays, chemical and cost saved, route), a **field map** drawn to scale in metres with every plant coloured by decision, spray-target rings, the depot and the spray route; a **per-plant decision table** (disease, confidence, severity, position, action, dose, reason, Grad-CAM thumbnail); the check against ground truth (synthetic demo); the software-in-the-loop flight result; the status of every pipeline stage; and links to the mission files.

![Top of the generated HTML report](docs/figures/report_top.jpg)

## Streamlit dashboard

`streamlit run dashboard/streamlit_app.py` opens a dashboard with five pages: **Field overview** (field map, key numbers, per-plant decisions), **Explainability** (Grad-CAM gallery), **Mission** (sorties, tank mixes, waypoint downloads, simulator), **Model card** (Module 3 metrics and figures) and **New survey** (upload images and run the pipeline). Simulated missions are stored like real surveys, so they appear in the dashboard too.

# 3D drone mission simulator

## Purpose

The simulator shows the complete mission as the drone would fly it, with the real SmartSpray modules doing the perception, decisions and planning:

1. The drone flies to the field corners the operator marked and **places ArUco markers**; it then climbs, photographs the field, learns the **boundary** and locates **every plant** (the data points).
2. It flies low and **photographs each plant**; each photo goes to the disease model, which decides **how much to spray**.
3. Plants that need no spray are skipped; the drone computes the **shortest path and time** to spray the rest and does it **autonomously**.
4. A pilot can take **manual control** from the keyboard at any time and hand control back.

```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/simulator/          (add ?quality=low on an older laptop)
```

![Mission phases](docs/figures/fig_sim_phases.png)

## Architecture

![Browser (flight, autopilot, pilot, 3D) and backend (camera, onboard AI, planning) exchange one REST call per mission step](docs/figures/fig_sim_arch.png)

The work is split the way a real system would split it:

- **The browser is the drone and the ground station** (`simulator/web/`): flight dynamics, the mission executor and autopilot, the pilot's keyboard, and the 3D view. three.js r186 is bundled locally (`vendor/three.bundle.min.js`), so the page works offline.
- **The backend is the drone's camera and onboard AI** (`app/simulation/`): `world.py` renders the synthetic field and every photo the drone takes; `service.py` runs the real modules on those photos and stores everything in the same database tables as a real survey.

## Synthetic world (`app/simulation/world.py`)

| Item | Value |
|---|---|
| Field presets | Field A/B: 7 rows × 9 plants on 10 m × 8 m (63 plants); Field C: 9 × 13 on 14 m × 10 m (117 plants); 1 m spacing with ±8 cm jitter |
| Disease layout | Field A = scripted demo layout (6 late blight, 7 early blight, 6 septoria, 44 healthy); B/C = one randomly grown patch per disease (~9 % of plants each) |
| Plant photos | Real PlantVillage **test-split** photos, never reused within a field; canopy 0.855 m |
| Ground texture | 220 px/m: soil, furrows between rows, plants, and every dropped marker where it landed |
| Markers | ArUco `DICT_4X4_50` ids 0–3, 0.60 m black square + 0.12 m white border |
| Overview photo (from 15 m) | texture down-sampled to 110 px/m + lighting gradient + sensor noise + 1.5–3 % perspective tilt |
| Close-up photo (from 2.5 m) | 1.6 m × 1.6 m footprint at 220 px/m (352 × 352 px) centred on the drone's actual position; gain ±5 %, noise, slight blur |
| Home pad / refill station | (−2.0, −1.6) m, outside the south-west corner |

## Mission phases in detail

**Phase 1 — Place markers.** Take-off to 3 m; for each planned corner: fly there, descend to 1.2 m, release a marker (it falls under gravity and lands within ±3 cm of the release point), report the landing point (`POST /sim/{id}/marker`), climb back.

**Phase 2 — Map field.** Climb to 15 m over the field centre and take the overview photo (`POST /sim/{id}/overview`). The backend:

1. saves the photo as a survey image and registers the survey (M1, including the blur check), with the field size measured between the dropped markers;
2. detects the four markers and solves the homography (M5) — the **boundary and true field size**;
3. detects every plant (M2 vegetation detector, kernel 0.004) and maps each box centre to metres (M5) — the **data points**;
4. runs a **row check**: plants are in rows at regular spacing, so a hole in a row (gap > 1.5 × the median spacing) or a short row probably hides a plant the detector missed; those spots are added to the low pass as "gap" targets;
5. orders all targets **row by row** (serpentine, starting from the south row nearest home);
6. scores the mapping against ground truth (synthetic field only): plants matched within 0.45 m, mean position error.

The boundary is drawn, its measured size is labelled, and a pin appears at every located plant.

**Phase 3 — Scan plants.** At 2.5 m and 2 m/s the drone stops over each target (7 cm tolerance), hovers 0.2 s and takes a close-up (`POST /sim/{id}/capture` with its actual position). The request runs while the drone flies on to the next plant (at most six photos are processed at once). For each photo the backend:

1. **locates the plant** with a plant-versus-soil colour model (soil = hue 3–24, saturation ≥ 60, value 40–190; everything else is plant), keeping the blob nearest the photo centre within 0.35 m; this crops the whole plant, including brown and pale diseased tissue (fallback: the M2 vegetation detector);
2. computes the plant's position from the photo (drone position + pixel offset / 220 px/m); a plant found at a "gap" target is added as a new plant (`low-pass-recovered`);
3. classifies the crop with the **Module 3 classifier** (TTA + temperature) and, if enabled, computes **Grad-CAM**;
4. estimates **severity** (M4) and applies the **dosage rules** (M6) to get the action, dose, treatment and reason;
5. stores the classification and severity rows and the photos (`data/outputs/<sim_id>/closeups/`).

The capture card shows the photo (or its Grad-CAM), the disease, the confidence bar, all four probabilities, the severity, the decision and its reason. The plant's ring in 3D turns green (healthy), yellow (monitor), orange or red (spray target, by treatment) or violet (inspect), with the dose as a floating label.

**Phase 4 — Plan route** (`POST /sim/{id}/plan`): re-map (to include recovered plants), decide every plant (M6; the re-treatment check is skipped because each synthetic field is a different field), build tank mixes (M7), plan the shortest capacity-aware route from the home pad (M8) and write the command queue and QGroundControl files (M9). The route is drawn in 3D and on the minimap, with the comparison to the blanket pass.

**Phase 5 — Spray.** For each sortie: fly home, land, load the mix (sortie load + 5 %), wait the 60 s refill (shown with an 8× time-warp; the battery is swapped if below 50 %), take off, and fly the route at 1.6 m. Over each target the drone hovers and sprays its own dose at 5 mL/s; spray particles, the tank gauge and a "wet" disc show it. If a plant was already partly sprayed (for example by the pilot), only the missing amount is topped up; if the tank cannot cover the next target, the route is re-planned.

**Phase 6 — Return and land**, then `POST /sim/{id}/finish` saves the flight log and returns the mission report: chemical used vs blanket, spray-route distance vs the lawn-mower pass, flight time, battery, what the pilot did, and the check against ground truth.

## Flight model and autopilot (`simulator/web/js/drone.js`, `mission.js`)

| Item | Model |
|---|---|
| Control | Velocity commands (x, y, z, yaw rate), like a multirotor flight controller in velocity mode |
| Limits | horizontal acceleration 3 m/s²; vertical speed 1.8 m/s, vertical acceleration 2 m/s²; yaw rate 1.8 rad/s; ground at 0 m |
| Power / battery | P = hover + (cruise − hover) · min(1.6, (v / 3)²) + 25 W per m/s of climb; 90 Wh battery; energy integrated every step |
| Physics step | fixed 1/60 s simulated; speed ×1, ×2, ×4, ×8; simulated time pauses (is not lost) while the drone waits for the model or the planner |
| Auto "go to" | speed = min(cruise, 0.92 · √(2 · 2.4 · d), 4d + 0.05) toward the target, so it slows smoothly on arrival; climb rate = clamp(1.6 · Δz, ±1.4); faces the direction of travel on legs over 0.7 m; arrived when within tolerance, height within 6 cm and nearly stopped |
| Visuals | body tilts with velocity and acceleration; rotors spin; tank liquid level and colour follow the loaded mix |

## Pilot override

| Key | Action |
|---|---|
| `M` | take manual control / hand back to the autopilot |
| `W` `A` `S` `D` or arrows | forward, left, back, right relative to the drone's heading (2.2 m/s; `Shift` 4.5 m/s) |
| `Q` `E` | turn left / right |
| `R` `F` | climb / descend (1.3 m/s); `F` on the ground stops the motors |
| `Space` | spray (hold): the dose goes to the plant under the drone (within 0.5 m) |
| `H` | return to home and land |
| `P` | pause |
| `1` `2` `3` `4` / `0` | camera: follow, top, orbit (mouse), FPV / automatic |

Safety and bookkeeping while the pilot flies: the autopilot is suspended and the drone holds position when no key is pressed; a **geofence** blocks movement more than 3 m outside the marked field (and home pad); the altitude is capped at 25 m; every hand-sprayed millilitre is credited to the plant under the drone and logged as on-target, off-target or on bare soil. When the pilot presses `M` during the spray phase, the browser sends the drone's position, the mix left in the tank and the targets not yet fully sprayed to `POST /sim/{id}/replan`; the backend returns the shortest route from that position (Module 8 re-planning) and the drone continues. A battery below 20 % triggers an automatic return home.

## User interface

Phase stepper and a one-line explanation of the current step; mode badge (AUTO / MANUAL / RETURN HOME / PAUSED); telemetry (phase, flight time, altitude, speed, position, distance, battery, tank and mix, markers, plants found, photographed, sprayed); plant-status legend; live **gimbal camera** feed (picture-in-picture), which shows the overview photo with its detections after mapping; capture card with a film strip of recent photos; event log; 2D minimap with the boundary, plants, route and drone; 3D labels (markers, field size, doses); end-of-mission report with download links (waypoints, flight log, summary).

![Phase 1: the drone drops marker M0 at the first corner (live camera shows the marker)](docs/figures/sim_03_marker.jpg)

![Phase 2: overview from 15 m — boundary 9.99 m × 8.02 m from the four markers, 63 plants located (mean error 4.3 cm); the camera panel shows the overview photo with its detections](docs/figures/sim_04_map.jpg)

![Phase 3: low-altitude scan — each photo is classified; the capture card shows the Grad-CAM, probabilities, severity and decision](docs/figures/sim_06_gradcam.jpg)

![Phase 4: every plant decided; spray targets labelled with their doses and the shortest route planned](docs/figures/sim_07_plan.jpg)

![Phase 5: spraying each target with its own dose (tank gauge, sprayed counter, route)](docs/figures/sim_08_spray.jpg)

![Pilot override: manual control banner and keys; on hand-back the route is re-planned from the drone's position](docs/figures/sim_09_manual.jpg)

![End-of-mission report](docs/figures/sim_11_summary.jpg)

## Simulator API and files

| Endpoint | Step |
|---|---|
| `POST /sim/session` `{seed, preset}` | new synthetic field (`small` or `medium`) |
| `POST /sim/{id}/marker` `{marker_id, x_m, y_m}` | a marker landed |
| `POST /sim/{id}/overview` | overview photo → boundary, size, plants, scan order |
| `POST /sim/{id}/capture` `{target_id, drone_x_m, drone_y_m, gradcam}` | close-up → disease, confidence, severity, decision, Grad-CAM |
| `POST /sim/{id}/plan` | decisions + route + waypoint files |
| `POST /sim/{id}/replan` `{position, tank_ml, tank_treatment, remaining}` | route from the current position |
| `POST /sim/{id}/finish` `{telemetry}` | flight log + mission report |
| `GET /sim/{id}/state` · `/sim/{id}/files/{name}` · `/sim/aruco/{id}.png` | state · overview / close-ups / logs · marker images |

Per mission: `data/images/sim_<id>/overview.jpg`; `data/outputs/sim_<id>/closeups/`, `mission/` (commands + `.waypoints`), `flight_log.json` (events, track, mL per plant, re-plans), `sim_summary.json`, `sim_state.json`. Old missions are removed with `python scripts/clean_simulations.py` (keeps the 5 most recent; `--keep 0` removes all).

**Performance.** One close-up (render + locate + classify with TTA + severity + Grad-CAM + JPEG encoding) takes about 0.35 s on a 2-core cloud CPU and less on an Apple-silicon Mac; the model is warmed up in the background when a mission is prepared.

# Testing and verification

## Automated tests

```bash
pytest -q                                  # all 277 tests, about 1 minute
pytest tests/test_classification.py -v     # one module, verbose
```

| Test file | Tests | Covers |
|---|---|---|
| `test_field_survey.py` | 5 | registration, duplicate ids, image quality check |
| `test_detection.py` | 28 | YOLO wrapper, OpenCV fallback, plant-ID ordering (incl. tilted rows), DB, API |
| `test_classification.py` | 36 | data contracts, class-order guard, inference config, TTA, temperature, uncertainty flag, DB, API, real-model checks |
| `test_severity.py` | 39 | severity bands, disease profiles, invalid crops, DB, API |
| `test_spatial_mapping.py` | 104 (incl. parametrised) | homography maths, marker layouts, reprojection checks, uncalibrated mode, DB, API |
| `test_decision.py` | 19 | every branch of the decision order, rule-file validation, savings summary |
| `test_resource_optimization.py` | 4 | grouping, tank loads, mix plan |
| `test_mission_planner.py` | 12 | NN, 2-opt, capacity splits, baseline never beaten, blocked/partial plans |
| `test_drone_integration.py` | 5 | command queue, waypoint format, GPS conversion, executor |
| `test_simulation.py` | 16 | synthetic world, row-gap check, serpentine order, re-planning, rendering → calibration/detection, full mission through the API, security checks, simulated sprays kept out of real re-treatment checks |
| **Total collected** | **277** | all passing |

Tests that need the trained weights or the dataset skip themselves automatically when those files are missing.

## End-to-end verification

| Check | Result |
|---|---|
| `python scripts/run_demo.py --fresh` (all 9 stages + SITL) | all stages `success`; numbers in Chapter 18 |
| API smoke test (health, classify/health, detect/health, decisions, report, waypoints, path traversal, unknown survey) | all as expected (200 / 404) |
| Headless-browser simulator missions (Chromium + Playwright, WebGL) | full autonomous missions on Field A; a mission with **return-home during the scan + manual flight and hand-spraying during the spray phase + re-plan**: all spray targets fully sprayed, every pilot action logged |
| Model promotion | old/new model SHA-256 recorded, candidate verified before copy |
| Sync to the user's machine | content digest of every synced file verified on both sides |
