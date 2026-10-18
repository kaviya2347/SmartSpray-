# Consolidated results

## Module-level results

| Module | Metric | Result | Source |
|---|---|---|---|
| M2 YOLOv8n | test mAP50 / mAP50-95 | 0.925 / 0.805 | `models/yolov8_plant_detector/eval/metrics.json` |
| M2 YOLOv8n | test precision / recall | 0.997 / 0.912 | same |
| M2 OpenCV fallback | demo field recall / precision | 1.00 / 1.00 (63/63) | `data/outputs/demo_survey_001/demo_ground_truth_evaluation.json` |
| M3 classifier | test accuracy (752 images) | **97.34 %** | `models/classification/eval/metrics.json` |
| M3 classifier | macro F1 / weighted F1 | 0.968 / 0.973 | same |
| M3 production mode | accuracy, ECE | 97.87 %, 0.0107 | same |
| M3 uncertainty gate | coverage / accuracy when confident | 98.7 % / 98.65 % | same |
| M3 on synthetic field crops | accuracy | 98.4 % (62/63) | demo evaluation |
| M5 mapping | mean / max position error | 5.5 cm / 25 cm | demo evaluation |
| M6 decisions | diseased plants acted on / healthy sprayed | 19/19 / 0/44 | demo evaluation |
| M6 chemical | used vs blanket | 112 mL vs 630 mL (**82.2 % less**) | pipeline result |
| M6 cost | used vs blanket | ₹8.82 vs ₹59.85 | pipeline result |
| M8 route | optimised vs lawn-mower | 38.4 m vs 95.0 m (**59.6 % shorter**); 16.8 % shorter than id order | pipeline result |
| M9 SITL | flight / battery / placement error | 135 s · 3.8 % · 5 cm, no safety violations | `mission/simulation_log.json` |

## 3D simulator (Field A, 63 plants)

Results of the documented autonomous run whose end-of-mission report is shown in Chapter 16 (values vary slightly between runs because marker drops and photo noise are random):

| Item | Result |
|---|---|
| Markers placed / read | 4 / 4; boundary measured as 9.99 m × 8.03 m (true 10 m × 8 m) |
| Plants located | 63 / 63, mean position error 4.3 cm |
| Plants photographed and classified | 63; 61 / 63 correct (96.8 %) |
| Decisions | 15 spray, 43 no spray, 5 monitor, 0 inspect; 19 / 19 diseased plants acted on; 1 healthy plant sprayed (misclassified as diseased) |
| Chemical | 122 mL vs 630 mL blanket (**80.6 % less**); ₹9.77 vs ₹59.85 |
| Spray route | 47.2 m vs 99.2 m lawn-mower pass from the home pad (**52.4 % shorter**), 2 tank loads |
| Mission | flight time 3 min 55 s, 217 m flown in total (markers, mapping, scan, spraying), 12 % battery (10.8 Wh) |
| With pilot override (separate run) | return-home during the scan, manual flight and hand-spraying during the spray phase, then hand-back: the route was re-planned from the drone's position and every spray target was completed; each pilot action is in the flight log |

## What the results show

1. **Plant-level treatment works end to end**: on the demo field only 14 of 63 plants were sprayed, all 19 diseased plants were acted on (sprayed, monitored or inspected), and no healthy plant was sprayed.
2. **The savings come from two places**: dose (only targets, each at its own dose) and distance (only targets, on the shortest capacity-aware route).
3. **The safety mechanisms trigger as designed**: uncertain predictions go to manual inspection; invalid severity follows a documented policy; uncalibrated surveys are blocked rather than guessed.
4. **The classifier is reliable in its training domain** (single-leaf photos), and the system is honest about the domain gap on real field photos (Chapter 21).

# Operating guide: terminal commands for every module

All commands are run from the project folder with the virtual environment active.

## Setup (once)

```bash
cd ~/Downloads/SmartSpray-main
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Every new terminal window needs `cd ~/Downloads/SmartSpray-main` and `source .venv/bin/activate` again.

## One-command demo

```bash
python scripts/run_demo.py --fresh                  # builds the synthetic field, runs M1-M9 + SITL
open data/outputs/demo_survey_001/report.html       # macOS; on Windows: start ...
```

## The API server

```bash
uvicorn app.main:app --reload                       # http://127.0.0.1:8000
# docs:       http://127.0.0.1:8000/docs
# simulator:  http://127.0.0.1:8000/simulator/
# report:     http://127.0.0.1:8000/reports/demo_survey_001
```

Stop it with `Ctrl+C`. Use `--port 8001` if port 8000 is busy. Leave this terminal running and open a second one for the commands below.

## Running each module through the API (curl)

The commands below re-run every module, one at a time, on the demo survey (run the demo once first). Each returns JSON; pipe into `python -m json.tool` to pretty-print.

```bash
H='Content-Type: application/json'
API=http://127.0.0.1:8000
S='"survey_id":"demo_survey_001"'

# M1  survey record
curl -s $API/survey/demo_survey_001 | python -m json.tool

# M2  detect plants (classical detector for the synthetic field)
curl -s -X POST $API/detect -H "$H" -d "{$S,\"detector\":\"classical\"}" | python -m json.tool | head -30

# M3  classify every plant (+ Grad-CAM files), then read the results
curl -s -X POST $API/classify -H "$H" -d "{$S,\"save_gradcam\":true}" | python -m json.tool | head -40
curl -s $API/classify/results/demo_survey_001 | python -m json.tool | head -40
curl -s $API/classify/health | python -m json.tool

# M4  severity
curl -s -X POST $API/severity -H "$H" -d "{$S}" | python -m json.tool | head -30

# M5  calibrate from the ArUco markers, then map plants to metres
curl -s -X POST $API/spatial/calibrate -H "$H" \
     -d "{$S,\"image_filename\":\"field_orthomosaic.jpg\",\"field_width_m\":10,\"field_height_m\":8}" | python -m json.tool
curl -s -X POST $API/spatial/map -H "$H" -d "{$S}" | python -m json.tool | head -30

# M6  decisions (+ the active rule table)
curl -s -X POST $API/decisions/generate -H "$H" -d "{$S}" | python -m json.tool | head -60
curl -s $API/decisions/rules/current | python -m json.tool | head -40

# M7 + M8  tank mixes and route
curl -s -X POST $API/mission/plan -H "$H" -d "{$S}" | python -m json.tool | head -60
curl -s $API/resources/demo_survey_001 | python -m json.tool

# M9  commands, waypoint file, software-in-the-loop flight
curl -s -X POST $API/drone/commands/generate -H "$H" -d "{$S}" | python -m json.tool | head -30
curl -s $API/drone/mission/demo_survey_001/1.waypoints
curl -s -X POST $API/drone/simulate -H "$H" -d "{$S}" | python -m json.tool | head -30

# everything at once
curl -s -X POST $API/pipeline/run -H "$H" \
     -d "{$S,\"detector\":\"classical\",\"save_gradcam\":true,\"simulate\":true}" | python -m json.tool | head -40
```

To run on your own images: put 5–10 photos in `data/images/survey_002/`, register with `curl -s -X POST $API/survey/register -H "$H" -d '{"survey_id":"survey_002","field_width_m":10,"field_height_m":8}'`, then call `/pipeline/run` with `"detector":"auto"`.

## Running each module directly in Python (no server needed)

```bash
# M1
python -c "import app; from app.field_survey.service import get_survey; print(get_survey('demo_survey_001'))"
# M2
python -c "import app; from app.detection.detector import get_detector; d = get_detector('classical'); print(len(d.detect('data/images/demo_survey_001/field_orthomosaic.jpg')), 'plants')"
# M3
python -c "import app; from app.classification.service import classify_survey; r = classify_survey('demo_survey_001'); print(r['classified'], 'classified,', r['uncertain'], 'uncertain')"
# M4
python -c "import app; from app.severity.service import estimate_survey; r = estimate_survey('demo_survey_001'); print(sum(x.is_valid for x in r), 'valid of', len(r))"
# M5
python -c "import app; from app.spatial_mapping.mapper import map_survey, summarise; print(summarise(map_survey('demo_survey_001')))"
# M6
python -c "import app; from app.decision.service import generate_decisions; print(generate_decisions('demo_survey_001')['summary'])"
# M7 + M8
python -c "import app; from app.mission_planner.service import plan_mission; print(plan_mission('demo_survey_001')['metrics'])"
# M9
python -c "import app; from app.drone_integration.command_queue import generate_commands; print(generate_commands('demo_survey_001')['files'])"
python -m simulator.executor_simulator demo_survey_001 --noise 0.05 --seed 7
# all stages
python -m app.pipeline demo_survey_001 --detector classical --gradcam --simulate --report
```

## Showing how one plant is classified (review demonstration)

```bash
python scripts/inspect_classifier.py                          # random test photo
python scripts/inspect_classifier.py --label early_blight     # random photo of one class
python scripts/inspect_classifier.py --survey demo_survey_001 --plant 12
python scripts/inspect_classifier.py "data/datasets/plantvillage_tomato/healthy/<file>.JPG"
open data/outputs/inspect/                                    # the Grad-CAM images it saved
```

It prints the nine steps of Section 7.7: model file and hash, layers, inference settings, pre-processing, both forward passes, the TTA average, temperature scaling, the final prediction, severity, the treatment decision, and where the Grad-CAM was saved.

## Looking inside the database

```bash
sqlite3 database/smartspray.db ".tables"
sqlite3 -header -column database/smartspray.db \
  "SELECT plant_id, label, ROUND(confidence,3) AS conf, is_uncertain FROM classifications WHERE survey_id='demo_survey_001' LIMIT 10;"
sqlite3 -header -column database/smartspray.db \
  "SELECT action, COUNT(*) AS plants, SUM(dosage_ml) AS ml FROM treatment_decisions WHERE survey_id='demo_survey_001' GROUP BY action;"
sqlite3 -header -column database/smartspray.db \
  "SELECT plant_id, ROUND(field_x_m,2) x, ROUND(field_y_m,2) y, mapping_mode FROM plant_locations WHERE survey_id='demo_survey_001' LIMIT 10;"
```

`sqlite3` is built into macOS. Without it (e.g. on Windows), use Python instead:

```bash
python -c "import sqlite3; c = sqlite3.connect('database/smartspray.db'); print(c.execute(\"SELECT action, COUNT(*) FROM treatment_decisions WHERE survey_id='demo_survey_001' GROUP BY action\").fetchall())"
```

## Opening the code

```bash
code .                                   # VS Code (if installed): whole project
open -a TextEdit app/classification/classifier.py
less app/decision/rules_engine.py        # q to quit
```

| To show… | Open |
|---|---|
| How a plant is classified | `app/classification/classifier.py` (`_prepare`, `_forward`, `predict_batch`), `app/classification/service.py` (`classify_survey`) |
| The model architecture | `app/classification/model_factory.py` |
| How training and the split work | `scripts/m3_train_classifier.py` |
| Grad-CAM | `app/classification/explain.py` |
| Severity | `app/severity/estimator.py` |
| Markers → metres | `app/spatial_mapping/calibration.py`, `mapper.py` |
| The dose rules | `rules/dosage_rules.json`, `app/decision/rules_engine.py` |
| Shortest route | `app/mission_planner/route_optimizer.py` |
| Drone file format | `app/drone_integration/command_queue.py` |
| The whole pipeline | `app/pipeline.py` |
| The simulator | `app/simulation/service.py` (backend), `simulator/web/js/mission.js` (autopilot + pilot), `simulator/web/js/drone.js` (flight model) |

## Re-training the models

```bash
# Module 3 (EfficientNetB0, 70/15/15)
python scripts/m3_train_classifier.py --stage split      # de-duplicate + split + manifest
python scripts/m3_train_classifier.py --stage train      # phases 1 + 2
python scripts/m3_train_classifier.py --stage deep       # phase 3 (kept only if validation improves)
python scripts/m3_train_classifier.py --stage evaluate   # the single test pass (then locked)
python scripts/m3_train_classifier.py --stage promote    # backup + SHA-256 + copy to best.keras
python scripts/build_m3_report_notebook.py               # executed report notebook
python scripts/write_m3_report.py                        # docs/MODULE3_REPORT.md

# Module 2 (YOLOv8n)
python scripts/m2_train_detector.py --epochs 50 --imgsz 640 --device mps    # Apple GPU; or --device cpu
```

The test split is locked by `models/classification/eval/TEST_EVALUATED.lock`; delete it only when deliberately starting a new, documented experiment.

## Dashboard, notebook, simulator, tests

```bash
streamlit run dashboard/streamlit_app.py                 # dashboard (second terminal)
pip install notebook && jupyter notebook notebooks/m3_classification_701515_report.ipynb
uvicorn app.main:app --reload   → open http://127.0.0.1:8000/simulator/
pytest -q                                                # all tests
python scripts/clean_simulations.py --keep 5             # remove old simulated missions
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `zsh: command not found: uvicorn` / `No module named fastapi` | activate the environment: `source .venv/bin/activate` |
| `No module named app` | run commands from the project folder (`cd ~/Downloads/SmartSpray-main`) |
| `Address already in use` | another server is running: stop it with `Ctrl+C`, or use `--port 8001` |
| Simulator page says "Backend not reachable" | start `uvicorn app.main:app --reload` and open the page from `http://127.0.0.1:8000/simulator/` |
| Simulator slow on an old laptop | open `http://127.0.0.1:8000/simulator/?quality=low` |
| `Model weights not found` | `models/classification/best.keras` is missing: copy it back or retrain |
| `No plant detections found` | run Module 2 first (or the pipeline) |
| Mission plan `blocked_uncalibrated` | the images have no readable ArUco markers: calibrate with `/spatial/calibrate` |

# Review preparation: likely questions and answers

**Q1. How exactly is a plant classified in the backend?**
The plant's bounding box from Module 2 is cropped from the survey image; the crop is converted from BGR to RGB and resized to 224 × 224; it and its mirror image go through EfficientNetB0 and the classification head in one batch; the two softmax outputs are averaged; the average is re-scaled with temperature T = 0.901; the class with the highest probability is the label and that probability is the confidence; if the confidence is below 0.60 the plant is flagged uncertain. The result is stored in the `classifications` table. Demonstrate with `python scripts/inspect_classifier.py` (Section 7.7).

**Q2. Why EfficientNetB0?**
It gives ImageNet-quality features with about 4 million parameters (ResNet-50 has about 25 million, VGG-16 about 138 million), so it trains in about an hour on a CPU and runs in tens of milliseconds per plant on a laptop or edge device, while being more accurate than MobileNet-class models of similar size.

**Q3. Why transfer learning instead of training from scratch?**
About 3,500 training images are far too few to learn general visual features from scratch. Starting from ImageNet features and fine-tuning only the upper layers converges fast and generalises better.

**Q4. Why 70/15/15, and what was wrong with 80/20?**
With only train and validation, the model is chosen on the validation set, so the validation score is optimistic. A separate test set, never used for any decision and evaluated once, gives an unbiased estimate. 70/15/15 keeps enough training data while giving 751 validation and 752 test images.

**Q5. How did you prevent data leakage?**
Exact duplicates were removed by SHA-256 before splitting; the split is stratified and seeded; path and hash overlaps between all splits were checked and are zero; augmentation is applied to training only; the test split is not even loaded during training; and a lock file prevents re-using it.

**Q6. Why is production accuracy (97.87 %) higher than the reported 97.34 %?**
97.34 % is the plain model. Production adds flip-TTA (chosen on validation), which averages two views and corrects a few borderline cases. Both are reported; neither was tuned on the test set.

**Q7. What is temperature scaling and why is T below 1?**
It divides the log-probabilities by a single number T fitted on validation data to make confidences match observed accuracy. T = 0.901 < 1 sharpens the probabilities, meaning the raw model was slightly under-confident. It never changes the predicted class.

**Q8. What happens when the model is not sure?**
Confidence below 0.60 sets `is_uncertain`; Module 6 then returns **inspect** and the plant is never sprayed automatically. On the test set, 10 plants were flagged and 6 of them were wrong predictions.

**Q9. What is Grad-CAM?**
It takes the gradient of the predicted class score with respect to the last convolutional feature maps, averages it per map to get importance weights, and forms a weighted sum of the maps; after a ReLU and upsampling this is a heatmap of the regions that supported the prediction. It is used to check that the model looks at lesions, and it revealed the domain gap on real greenhouse photos.

**Q10. How is severity measured, and is it accurate?**
Leaf pixels are segmented by colour, lesion pixels by disease-specific colour rules inside the leaf, and severity = lesion / leaf area, graded at 20 % and 50 %. It is a transparent heuristic; no labelled severity data was available, so no accuracy is claimed and it is listed as future work.

**Q11. How do you get positions in metres?**
Four ArUco markers at known field corners are detected in the image; their centres give four image–field point pairs, from which OpenCV computes a homography (a 3 × 3 projective transform). Every plant's box centre is multiplied by it to get (x, y) in metres. The demo error is 5.5 cm on average.

**Q12. Why ArUco and not GPS?**
Consumer GPS is accurate to 2–5 m, which is larger than the 1 m plant spacing. ArUco markers give centimetre-level accuracy with a normal camera and no extra hardware. RTK-GPS (2 cm) is the planned upgrade and would replace only Module 5.

**Q13. How is the dose decided, and who sets the rules?**
By an ordered rule table in `rules/dosage_rules.json`: healthy → no spray; unsure → inspect; disease × severity → monitor or spray with a dose; late blight always treated. Agronomists can edit the JSON without touching code, and every decision records the rule version and a reason.

**Q14. How is the route optimised? Why not an exact solver?**
Nearest-neighbour construction plus 2-opt improvement, cut into tank-sized sorties and optimised again. The exact travelling-salesman problem is NP-hard; for tens of plants 2-opt runs in milliseconds and is typically within a few percent of optimal without any external solver. The plant-id order is always a candidate, so the result is never worse than the naive route.

**Q15. How does tank capacity affect the plan?**
Usable capacity is 237.5 mL (250 mL minus 5 % reserve). Targets of the same treatment are split into consecutive sorties whose total dose fits; different treatments use different sorties because mixes cannot share a tank.

**Q16. How would this connect to a real drone?**
Module 9 writes standard MAVLink missions (`QGC WPL 110`): `NAV_TAKEOFF` (22), `NAV_WAYPOINT` (16) to each plant, `DO_SPRAYER` (216) on, hover for the spray time, `DO_SPRAYER` off, and `RETURN_TO_LAUNCH` (20). These load into QGroundControl or Mission Planner and fly on ArduPilot/PX4 drones. A live link would send the same items over MAVLink telemetry.

**Q17. Why software and not hardware?**
See Section 2.4: the innovation is the decision, the output is hardware-agnostic, hardware is costly and regulated, software is reproducible and testable, and the design keeps a slot for hardware integration.

**Q18. What in the 3D simulator is real and what is simulated?**
Real: ArUco calibration, plant detection, the disease classifier and Grad-CAM, severity, dosage rules, tank mixes, route optimiser, re-planning and waypoint export. Simulated: the field (real test photos on generated soil), the camera images, the flight dynamics, battery and nozzle.

**Q19. How do the manual override and re-planning work?**
`M` suspends the autopilot; the keys send velocity commands; a geofence keeps the drone near the field; sprayed millilitres are credited to the plant under the drone. Pressing `M` again sends the drone's position, remaining mix and unfinished targets to `/sim/{id}/replan`, which returns the shortest route from that position (Module 8, open-path 2-opt), and the autopilot continues.

**Q20. How do you know chemical is saved?**
The baseline is defined in the rules file: blanket spraying gives every plant 10 mL. SmartSpray's usage is the sum of the doses it actually decided (demo: 112 mL vs 630 mL). The route is compared with a lawn-mower pass covering the whole field with a 1 m swath.

**Q21. What is mAP?**
Mean average precision: the area under the precision–recall curve, averaged over classes; mAP50 counts a detection as correct at ≥ 50 % box overlap, mAP50-95 averages overlaps from 50 % to 95 %.

**Q22. What are the main limitations?**
The classifier is trained on single-leaf photos and does not transfer to real field photos without fine-tuning (Grad-CAM showed it attending to background); severity is not validated; fields are synthetic; doses and prices are placeholders; drone output is validated only in simulation.

**Q23. How would you deploy this on a real farm?**
Collect and label drone images of real plants (or use PlantDoc), fine-tune M2 and M3 on them, validate severity with an agronomist, place printed ArUco markers (or use RTK-GPS), get the dosage rules approved, fly survey missions, review the plan in the dashboard, and fly the exported mission with a licensed pilot.

**Q24. How is the system tested?**
277 automated pytest tests across all modules, an end-to-end demo with a ground-truth check, an API smoke test, and headless-browser runs of complete simulator missions including manual override and re-planning.

**Q25. How fast is it?**
About 75 ms per plant for classification with TTA and about 50 ms for Grad-CAM on a 2-core cloud CPU; the full 63-plant demo pipeline (all nine modules plus the flight simulation) runs in about 20–30 seconds; route planning takes milliseconds.

**Q26. Why SQLite and FastAPI?**
SQLite needs no server and keeps every stage's results in one file that can be inspected with standard tools; FastAPI gives typed, validated endpoints with automatic interactive documentation.

**Q27. What if a marker is missing or moved?**
Fewer than four markers, or a reprojection error above 25 px, makes the calibration invalid; plants stay unmapped with NULL positions and the mission is blocked. The system never invents coordinates.

**Q28. What was the hardest problem?**
Making the evaluation trustworthy: removing duplicates, building a real held-out test set, and choosing every setting on validation only. In the simulator, cropping the whole plant (not just its green part) at close range was key: it raised accuracy on the simulated photos from about 95 % to about 99 %.

# Limitations and future work

## Limitations

1. **Domain gap.** PlantVillage photos are single leaves on plain backgrounds. On six real greenhouse photos from the LeafPest test set, YOLOv8 found the plants correctly, but the classifier labelled most of them as diseased and Grad-CAM showed attention on irrigation pipes and background. The uncertainty gate and the inspect action limit the damage, but real deployment needs fine-tuning on labelled field crops.
2. **Severity** is a colour heuristic without ground-truth validation.
3. **Synthetic fields.** The demo and simulator fields are made from real test-split photos on generated soil; they demonstrate and measure the pipeline but are not a field trial.
4. **Placeholder parameters.** Doses, prices, tank, battery and power values in `rules/` are engineering defaults; they must be checked against product labels, an agronomist and the real airframe.
5. **Simulation only.** Drone output has been validated in software (SITL executor and 3D simulator), with no wind, GPS error or obstacles modelled.
6. **Marker dependence.** Positions need four visible markers (or a future RTK-GPS module).

## Remaining work (≈ 25 %)

| Item | Plan |
|---|---|
| Field imagery and field-trained models | collect and label drone crops or use PlantDoc; fine-tune M2 and M3; re-evaluate on held-out field images |
| Severity validation | agronomist-labelled severity scores; compare and calibrate the heuristic or train a segmentation model |
| Live drone link | send the MAVLink mission over telemetry; real pump and nozzle control |
| RTK-GPS positioning | replace marker calibration with geotagged images |
| Weather-aware spraying | block spraying in wind or rain; time windows |
| Simulator realism | wind, GPS noise, obstacles, camera perspective and motion blur |
| Multi-field accounts | field identities for the re-treatment check across surveys, user roles |

# References

1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of ICML 2019*.
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of ICML 2017*.
5. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*.
6. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (software). https://github.com/ultralytics/ultralytics
7. Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. *Pattern Recognition*, 47(6), 2280–2292.
8. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
9. Croes, G. A. (1958). A method for solving traveling-salesman problems. *Operations Research*, 6(6), 791–812.
10. Woebbecke, D. M., Meyer, G. E., Von Bargen, K., & Mortensen, D. A. (1995). Color indices for weed identification under various soil, residue, and lighting conditions. *Transactions of the ASAE*, 38(1), 259–269.
11. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proceedings of CoDS-COMAD 2020*.
12. LeafPest. *Plant Detection* dataset v1, Roboflow Universe, CC BY 4.0.
13. MAVLink developer guide — mission protocol and `MAV_CMD` reference; QGroundControl plan file formats. https://mavlink.io
14. Ministry of Civil Aviation, Government of India (2021). *The Drone Rules, 2021*.
15. John Deere (2024). See & Spray 59 percent herbicide savings. https://www.deere.com/en-us/john-deere-news/see-spray-59-percent-herbicide-savings
16. XBOOM. Agriculture drone price in India 2026. https://www.xboom.in/agriculture-drone-price-india/
17. Leher. Agri drone regulations and requirements in India. https://www.leher.ag/blog/agri-drone-regulations-requirements-india

# Appendix A — File-by-file reference

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI application: registers all routers, `/health`, `/pipeline/run`, `/reports/{id}`, serves `/simulator/` |
| `app/pipeline.py` | runs M1–M9 for a survey; CLI `python -m app.pipeline` |
| `app/db.py` | SQLite connection (`database/smartspray.db`) and `init_db()` |
| `app/__init__.py` | import-order guard that loads `triton` before TensorFlow (avoids a crash when both are installed) |
| `app/reporting.py` | self-contained HTML report and ground-truth evaluation |
| `app/field_survey/service.py`, `router.py` | M1 registration and image quality check |
| `app/detection/detector.py` | M2 YOLOv8 wrapper, OpenCV fallback detector, `get_detector()` |
| `app/detection/identity_resolver.py` | M2 plant-ID ordering (adaptive rows) |
| `app/detection/router.py` | M2 API and `detections` table |
| `app/classification/classifier.py` | M3 inference: model loading, class-order check, inference config, TTA, temperature, uncertainty |
| `app/classification/model_factory.py` | M3 architecture and fine-tuning helpers |
| `app/classification/explain.py` | M3 Grad-CAM (compiled) and heatmap overlay |
| `app/classification/prediction.py` | M3 data contracts, class names, model version, threshold |
| `app/classification/service.py`, `database.py`, `router.py` | M3 survey workflow, storage, API |
| `app/severity/estimator.py`, `service.py`, `database.py`, `router.py` | M4 |
| `app/spatial_mapping/calibration.py`, `mapper.py`, `database.py`, `router.py` | M5 |
| `app/decision/rules_engine.py`, `service.py`, `database.py`, `router.py` | M6 (database.py also stores M7–M9 tables) |
| `app/resource_optimization/grouping.py`, `router.py` | M7 and mission configuration loader |
| `app/mission_planner/route_optimizer.py`, `service.py`, `router.py` | M8 |
| `app/drone_integration/command_queue.py`, `router.py` | M9 |
| `app/simulation/world.py`, `service.py`, `router.py` | 3D simulator backend |
| `simulator/executor_simulator.py` | software-in-the-loop executor |
| `simulator/web/index.html`, `style.css`, `js/*.js`, `vendor/three.bundle.min.js` | 3D simulator front end (main, mission, drone, world3d, ui, api) |
| `dashboard/streamlit_app.py` | Streamlit dashboard |
| `rules/dosage_rules.json`, `rules/mission_config.json` | auditable decision rules and drone parameters |
| `scripts/m3_train_classifier.py` | M3 split, training, evaluation, promotion |
| `scripts/m2_train_detector.py` | M2 YOLO training and evaluation |
| `scripts/make_demo_field.py`, `scripts/run_demo.py` | synthetic demo field and one-command demo |
| `scripts/inspect_classifier.py` | step-by-step classification of one image |
| `scripts/build_m3_report_notebook.py`, `scripts/write_m3_report.py` | M3 report notebook and Markdown report |
| `scripts/severity_diagnostics.py` | M4 diagnostics on real images |
| `scripts/clean_simulations.py` | remove old simulated missions |
| `scripts/legacy_m3/` | superseded scripts from the old 80/20 workflow (kept for the record) |
| `tests/test_*.py` | pytest suite (277 tests) |
| `docs/` | this document (`PROJECT_DOCUMENTATION.md` / `.docx`), `MODULE3_REPORT.md`, `SIMULATOR.md`, `WHY_SMARTSPRAY.md` |

# Appendix B — Configuration files

`models/classification/inference_config.json`

```json
{
  "model_version": "efficientnetb0-tomato-disease-v2-701515",
  "class_order": ["early_blight", "healthy", "late_blight", "septoria_leaf_spot"],
  "image_size": 224,
  "input": "RGB uint8 0-255, resized with cv2.resize (bilinear); EfficientNet preprocess_input is a pass-through",
  "temperature": 0.9010197026101795,
  "tta_hflip": true,
  "uncertainty_threshold": 0.6
}
```

`rules/mission_config.json`

```json
{
  "tank_capacity_ml": 250, "tank_reserve_pct": 5, "cruise_speed_mps": 3.0,
  "spray_flow_ml_per_s": 5.0, "refill_time_s": 60, "takeoff_land_time_s": 20,
  "cruise_altitude_m": 2.5, "spray_swath_m": 1.0, "depot_x_m": 0.0, "depot_y_m": 0.0,
  "origin_lat": 12.8406, "origin_lon": 80.1534,
  "battery_wh": 90, "cruise_power_w": 180, "hover_power_w": 160
}
```

`rules/dosage_rules.json` (summary; full file in the repository): version `dosage-rules-v1`; unit mL of prepared mix per plant; blanket dose 10 mL; uncertainty threshold 0.6; re-treatment interval 7 days within 0.35 m; treatments `protectant_fungicide` (₹60 per litre of mix) and `late_blight_fungicide` (₹95); invalid-severity policy late blight → spray default, others → inspect; rule table as in Chapter 10.

# Appendix C — Glossary

| Term | Meaning |
|---|---|
| ArUco marker | square black-and-white fiducial pattern that a camera can detect and identify |
| Augmentation | random changes (flip, rotation, zoom…) applied to training images to improve generalisation |
| Blanket spraying | spraying every plant with the same dose |
| Calibration (model) | making predicted confidence match observed accuracy |
| ECE | expected calibration error: average gap between confidence and accuracy over confidence bins |
| EfficientNetB0 | compact convolutional neural network used as the feature extractor |
| ExG | excess-green index 2G − R − B, used to separate vegetation from soil |
| F1 score | harmonic mean of precision and recall |
| Grad-CAM | gradient-weighted class activation map, a visual explanation of a CNN decision |
| Homography | 3 × 3 projective transform mapping one plane (image) to another (field) |
| mAP | mean average precision for object detection |
| MAVLink | messaging protocol used by ArduPilot/PX4 drones and ground stations |
| NLL | negative log-likelihood, the loss used to fit the temperature |
| Reprojection error | distance between where calibration points should be and where the homography puts them |
| SITL | software-in-the-loop: running the mission against simulated hardware |
| Sortie | one flight between two refills |
| Stratified split | split that keeps each class's proportion the same in every subset |
| Temperature scaling | dividing log-probabilities by a fitted constant T to calibrate confidences |
| TTA | test-time augmentation: predicting on several versions of an image and averaging |
| 2-opt | local-search improvement for routes that reverses segments to remove crossings |
| YOLOv8 | single-stage real-time object detector |
