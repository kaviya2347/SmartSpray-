# Module 4 — Severity estimation

**Purpose.** Estimate how much of a diseased plant's leaf area shows symptoms, so the dose can follow the severity.

**Method.** A deterministic OpenCV estimator (`app/severity/estimator.py`), not a trained model. For each plant crop and its Module 3 label:

1. **Leaf segmentation.** Convert to HSV and keep green / yellow-green tissue (H 25–90, S ≥ 40, V ≥ 40); close small holes (7 × 7 kernel); keep the largest connected component as the leaf; expand it with a large elliptical closing (21 × 21) so that brown diseased tissue surrounded by green is counted as leaf. The pixel count is `leaf_px`.
2. **Lesion segmentation** with a disease-specific profile (table below), restricted to the leaf mask; small components are discarded. The pixel count is `lesion_px`.
3. **Score** = lesion_px / leaf_px (0–1).
4. **Category** from fixed bands: **< 0.20 low, 0.20–0.50 medium, > 0.50 high** (design parameters from the original specification, not agronomic thresholds).
5. **Validity.** Healthy plants get score 0 and category low. The estimate is marked `is_valid = False` (score 0) for an empty crop or one without three colour channels, a crop smaller than 16 × 16 px, no leaf tissue found, or a leaf fill ratio below 0.20 of the box. Module 6 then applies its invalid-severity policy (late blight is still treated; other diseases go to inspection).

Lesion profiles used in step 2:

| Disease | Lesion colour rule (OpenCV 8-bit HSV / Lab units) | Clean-up | Component filter |
|---|---|---|---|
| Early blight | HSV H 10–40, S ≥ 40, V ≥ 40 **and** Lab L* ≤ 110, a* ≥ 115 (dark reddish-brown rings) | open 3, close 5 | ≥ 9 px |
| Late blight | HSV V ≤ 90 (dark, water-soaked tissue) | close 11 | ≥ 20 px |
| Septoria | HSV H 10–40, V ≥ 90 **and** Lab L* ≥ 25, a* ≥ 90 (pale spots) | open 2, close 5 | 12 px up to 20 % of the leaf |

| Item | Value |
|---|---|
| Code | `app/severity/estimator.py` (algorithm), `service.py` (crops from survey + labels from M3), `database.py`, `router.py` |
| Table | `plant_severity` (survey_id, plant_id, image_filename, disease_label, severity_score, severity_category, leaf_px, lesion_px, is_valid, model_version) |
| API | `POST /severity` `{"survey_id"}` · `GET /severity/health` |
| Tests | `tests/test_severity.py` (39 tests: bands, profiles, invalid cases, DB) |
| Limitation | No labelled severity ground truth exists; the score is a heuristic and is reported as such. |

# Module 5 — Spatial mapping

**Purpose.** Convert every plant's pixel position into field coordinates in metres, so the drone knows where to fly.

**Marker layout.** Four ArUco markers (dictionary `DICT_4X4_50`) lie flat at the field corners: id 0 at (0, 0) — the origin, id 1 at (W, 0), id 2 at (W, H), id 3 at (0, H), where W × H is the field size in metres. ArUco markers are square binary patterns with error-correcting codes that OpenCV can detect and identify robustly even at low resolution.

**Calibration** (`app/spatial_mapping/calibration.py`):

1. Detect markers with `cv2.aruco.ArucoDetector` (OpenCV ≥ 5 API). The correspondence point for each marker is its **centre**, the mean of its four detected corners.
2. Match each detected id to its known field corner. At least 4 markers are required.
3. Solve the **homography** H, a 3 × 3 projective transform from image to field: with exactly 4 points the solution is exact (`cv2.findHomography(src, dst, 0)`); with more points RANSAC is used (5 px inlier threshold).
4. Compute the mean **reprojection error** in pixels; the calibration is valid only if it is ≤ 25 px.
5. Store the calibration (homography as JSON) in `spatial_calibrations`.

**Mapping** (`app/spatial_mapping/mapper.py`): for each detection take the bounding-box centre (u, v) and apply

  [X, Y, W]ᵀ = H · [u, v, 1]ᵀ,  x = X / W,  y = Y / W

which gives the plant position (x, y) in metres. A homography also corrects the mild perspective tilt of an oblique drone photo.

**No guessing.** If a survey has no valid calibration, plants are stored with `mapping_mode = "uncalibrated"` and NULL coordinates. Module 8 then refuses to plan a mission (`blocked_uncalibrated`) instead of inventing positions. The pipeline first tries **auto-calibration**: it looks for the four markers in the survey's own images.

| Item | Value |
|---|---|
| Tables | `spatial_calibrations` (calibration_id, survey_id, image_filename, field_width_m, field_height_m, aruco_dict, marker_count, homography_json, reprojection_error, is_valid); `plant_locations` (survey_id, plant_id, bbox_cx_px, bbox_cy_px, field_x_m, field_y_m, is_mapped, calibration_id, mapping_mode, model_version) |
| API | `POST /spatial/calibrate` `{"survey_id", "image_filename", "field_width_m", "field_height_m"}` · `POST /spatial/map` `{"survey_id"}` · `GET /spatial/locations/{survey_id}` · `GET /spatial/health` |
| Result | Demo field: mean position error **5.5 cm**, maximum 25 cm, against ground truth |
| Tests | `tests/test_spatial_mapping.py` (104 tests: homography maths, marker layouts, invalid inputs, DB, API) |
| Upgrade path | Replace ArUco with RTK-GPS geotags; only this module changes |

# Module 6 — Treatment decision engine

**Purpose.** Turn the evidence for each plant (disease and confidence from M3, severity from M4, position from M5) into an explicit, auditable decision: **action, dose, treatment and a written reason.**

**Actions:** `spray` · `no_spray` · `monitor` (mild; re-survey before treating) · `inspect` (a person must look) · `hold` (would spray, but it was sprayed recently).

**Decision order** (`app/decision/rules_engine.py`, first matching rule wins):

1. No classification → **inspect**.
2. Unknown label → **inspect**.
3. Healthy → **no_spray**.
4. Confidence < 0.60 (or flagged uncertain by M3) → **inspect**. *The system never sprays on a guess.*
5. Severity invalid or missing → invalid-severity policy: late blight → spray with the medium dose (it spreads too fast to wait); other diseases → **inspect**.
6. Disease × severity rule from the table below.
7. Re-treatment check: if the action is spray and a mapped spray decision from **another** survey within the last **7 days** lies within **0.35 m** of this plant → **hold**. (Simulated missions, ids starting with `sim_`, are separate synthetic fields and are ignored.)

**Rule table** (`rules/dosage_rules.json`, version `dosage-rules-v1`; doses in mL of prepared mix per plant):

| Disease | Low (< 20 %) | Medium (20–50 %) | High (> 50 %) | Treatment |
|---|---|---|---|---|
| Healthy | no spray | no spray | no spray | — |
| Early blight | monitor | spray 5 mL | spray 9 mL | protectant fungicide |
| Late blight | spray 10 mL | spray 10 mL | spray 10 mL | late-blight fungicide |
| Septoria leaf spot | monitor | spray 6 mL | spray 8 mL | protectant fungicide |

Blanket baseline: 10 mL on every plant. Prices (indicative): protectant mix ₹60/L, late-blight mix ₹95/L. Because the rules live in a JSON file, an agronomist can audit and change them without touching code; the file is validated at load time (every disease needs all three severity levels, valid actions, non-negative doses, and a treatment for every spray).

Every decision carries its reason, for example: *"Early blight 20–50% leaf area — standard protectant dose. (measured severity 41%)"* or *"Model confidence 0.44 < 0.60 for 'late_blight' — flagged uncertain, no automatic spray."*

**Savings summary** (`summarise_decisions`): chemical used vs blanket baseline (mL and %), cost vs blanket (blanket uses the most expensive programme on every plant, because a farmer who cannot see which disease is where sprays a broad programme), counts per action and per label.

| Item | Value |
|---|---|
| Table | `treatment_decisions` (plant_id, disease_label, confidence, is_uncertain, severity_score, severity_category, severity_valid, field_x_m, field_y_m, is_mapped, action, dosage_ml, treatment, reason, rules_version, decided_at) |
| API | `POST /decisions/generate` `{"survey_id"}` · `GET /decisions/{survey_id}` · `GET /decisions/rules/current` |
| Demo result | 14 of 63 plants sprayed, 4 monitored, 1 inspected, 44 left alone; 112 mL vs 630 mL (**82.2 % less**); ₹8.82 vs ₹59.85 |
| Tests | `tests/test_decision.py` (19 tests: every branch of the decision order, rule validation, summary) |

# Module 7 — Resource optimisation

**Purpose.** Work out what has to be mixed and loaded.

1. Group spray targets by treatment (largest programme first). Different mixes cannot share a tank.
2. For each programme: mix needed = Σ doses; mix to prepare = needed × (1 + 5 % reserve), rounded up; minimum tank loads = ⌈needed / usable capacity⌉; estimated cost.
3. Usable tank capacity = 250 mL × (1 − 5 % reserve) = 237.5 mL.

**Mission configuration** (`rules/mission_config.json`, placeholders for a small spot-spraying drone):

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Tank capacity | 250 mL | Cruise speed | 3.0 m/s |
| Tank reserve | 5 % | Spray flow | 5.0 mL/s |
| Refill time | 60 s | Take-off + landing time | 20 s |
| Cruise altitude | 2.5 m | Spray swath (blanket baseline) | 1.0 m |
| Battery | 90 Wh | Cruise / hover power | 180 W / 160 W |
| Depot (refill point) | field (0, 0) by default | GPS origin | 12.8406 N, 80.1534 E |

| Item | Value |
|---|---|
| Code | `app/resource_optimization/grouping.py`, `router.py` |
| API | `GET /resources/{survey_id}` · `GET /resources/config` |
| Tests | `tests/test_resource_optimization.py` |

# Module 8 — Mission planning

**Purpose.** Find the shortest flyable route over the plants that need spraying, starting and ending at the depot, without exceeding the tank.

**Algorithms** (`app/mission_planner/route_optimizer.py`, pure functions):

1. **Nearest neighbour** construction: from the depot, always fly to the closest unvisited target.
2. **2-opt** improvement (Croes, 1958): repeatedly reverse a segment of the tour whenever that shortens it (replace edges (a, b) and (c, d) by (a, c) and (b, d) if dist(a, c) + dist(b, d) < dist(a, b) + dist(c, d)), until no improvement remains.
3. **Capacity-aware routing** (a small capacitated vehicle-routing heuristic, "route first, cluster second"): build one optimised giant tour; try every rotation of it in both directions (plus the plant-id order as a baseline candidate); cut each candidate into consecutive tank-sized sorties; 2-opt every sortie; keep the candidate with the smallest total distance. Because the baseline is a candidate, the result is never worse than visiting plants in id order.
4. **Lawn-mower baseline**: the length of a back-and-forth pass covering the whole field with the spray swath (1 m), from and back to the depot. A 5 % tolerance avoids an extra pass when the measured field is a few centimetres taller than a whole number of swaths.
5. **Re-planning from any position** (used by the simulator after a manual override): serve as many targets as the remaining tank allows on an open path from the drone's position (nearest neighbour, then 2-opt with a fixed start and end at the depot), then plan the rest as normal sorties from the depot.

For 10–100 targets these run in milliseconds and are typically within a few percent of optimal, with no external solver.

**Metrics** for the plan and the blanket baseline:

- Flight time = distance / speed + Σ(dose / flow) + sorties × take-off/landing time + (sorties − 1) × refill time.
- Energy = (distance / speed) × cruise power + spray time × hover power; battery % = energy / 90 Wh.
- Blanket flight time uses the lawn-mower length and continuous spraying (no extra hover).

**Status:** `ready`, `ready_partial` (some targets unmapped, left out), `no_targets`, or `blocked_uncalibrated` (targets exist but none has a position — no savings are claimed).

| Item | Value |
|---|---|
| Code | `route_optimizer.py` (algorithms), `service.py` (build + store plan), `router.py` |
| Table | `mission_plans` (survey_id, status, plan_json, created_at) |
| API | `POST /mission/plan` `{"survey_id", "tank_capacity_ml"?, "cruise_speed_mps"?, "spray_flow_ml_per_s"?}` · `GET /mission/{survey_id}` |
| Demo result | **38.4 m** in 2 sorties vs **95.0 m** lawn-mower (59.6 % shorter); 16.8 % shorter than plant-id order |
| Tests | `tests/test_mission_planner.py`, plus re-planning tests in `tests/test_simulation.py` |

# Module 9 — Drone integration

**Purpose.** Turn the plan into commands a drone or executor can run, in a standard format.

**Command queue** (`app/drone_integration/command_queue.py`): for every sortie, in order: `load_tank` (treatment, volume) → `takeoff` (cruise altitude) → one `spray` per plant (location, dose, treatment, spray seconds, disease, severity) → `return_to_depot` → `land`. Stored in `drone_commands` with a status per command, and written to `data/outputs/<survey_id>/mission/drone_commands.json`.

**QGroundControl / Mission Planner export.** Each sortie is written as a `QGC WPL 110` text file (`sortie_##_<treatment>.waypoints`), one MAVLink mission item per line: index, current, frame (3 = global, altitude relative to home), command, four parameters, latitude, longitude, altitude, autocontinue.

| MAVLink command | Id | Used for |
|---|---|---|
| `MAV_CMD_NAV_WAYPOINT` | 16 | home, fly to plant, and hover (param 1 = spray seconds) |
| `MAV_CMD_NAV_TAKEOFF` | 22 | climb to cruise altitude |
| `MAV_CMD_DO_SPRAYER` | 216 | sprayer on (param 1 = 1) / off (param 1 = 0) |
| `MAV_CMD_NAV_RETURN_TO_LAUNCH` | 20 | return home at the end of the sortie |

```text
QGC WPL 110
0  1  0  16   0.00  0  0  0  12.84060000  80.15340000  0.00  1    home
1  0  3  22   0.00  0  0  0  12.84060000  80.15340000  2.50  1    take off to 2.5 m
2  0  3  16   0.00  0  0  0  12.84060888  80.15340998  2.50  1    fly to plant
3  0  3  216  1.00  0  0  0  0.00000000   0.00000000   0.00  1    sprayer ON
4  0  3  16   1.00  0  0  0  12.84060888  80.15340998  2.50  1    hover 1.0 s
5  0  3  216  0.00  0  0  0  0.00000000   0.00000000   0.00  1    sprayer OFF
```

Field metres are converted to GPS with lat = lat₀ + y / 111 320 and lon = lon₀ + x / (111 320 · cos lat₀).

**Software-in-the-loop executor** (`simulator/executor_simulator.py`) "flies" the queue: point-to-point flight at cruise speed with seeded GNSS noise (σ = 0.05 m), hover for dose / flow seconds while spraying, tank level and energy tracking, and safety checks (tank loaded only at the depot, tank never negative, spray point inside the field, battery reserve of 20 % respected). Every step is written to `execution_logs`; the summary goes to `mission/simulation_log.json`. Demo result: 135 s flight, 38.4 m, 3.8 % battery, mean spray placement error 5 cm, no violations.

| Item | Value |
|---|---|
| API | `POST /drone/commands/generate` `{"survey_id"}` · `GET /drone/commands/{survey_id}` · `GET /drone/mission/{survey_id}/{sortie}.waypoints` · `POST /drone/simulate` `{"survey_id", "gnss_noise_m", "seed"}` |
| CLI | `python -m simulator.executor_simulator <survey_id> --noise 0.05 --seed 7` |
| Tests | `tests/test_drone_integration.py` |
