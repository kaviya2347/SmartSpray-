# SmartSpray: what makes it different, and why it is software first

*Vision-Guided Autonomous UAV for Plant-Level Disease Detection and Variable-Rate Spraying*

This note is for the review panel. It covers three questions:

1. What makes SmartSpray different from existing work?
2. Why did we build software instead of hardware?
3. How far along is the project?

---

## 1. What makes it different

### The gap it fills

Existing tools each do one part of the job:

- **Research models** classify a leaf photo. They stop at the diagnosis.
  Examples: PlantVillage CNNs, YOLO disease detectors (papers 8, 14 and 15 in our review).
- **Commercial precision sprayers** mostly target weeds, not disease, and
  come as expensive machines. John Deere's See & Spray, for example,
  reported about 59 % average herbicide savings in 2024. It is a weed
  system built into large, costly sprayers.
- **Agricultural drones** spray the whole field evenly. A 10–16 L spray drone
  costs roughly ₹3–5 lakh in India, and larger models cost ₹11 lakh or more.

No single existing tool takes one survey image all the way to a plant-level,
severity-based, justified spray command. SmartSpray connects all the steps:

```
image ─► plants (M2) ─► disease + confidence (M3) ─► severity (M4)
      ─► metres in the field (M5) ─► auditable decision + dose (M6)
      ─► tank mixes (M7) ─► optimised route (M8) ─► drone commands (M9)
```

### Point-by-point comparison

| Question a farmer or reviewer asks | Typical existing work | SmartSpray |
| --- | --- | --- |
| Which plant is sick? | Leaf-level or whole-field | Each plant gets an ID and a position in metres |
| How sick is it? | Rarely estimated | Lesion-area severity, graded Low / Medium / High |
| What should I do, and why? | Diagnosis only | A rule table anyone can check (`rules/dosage_rules.json`), plus a plain-language reason for every plant |
| What if the AI is unsure? | The prediction is used anyway | Calibrated confidence. Below 0.60 the plant is sent for **manual inspection** and never sprayed automatically |
| Can I trust the model's claim? | Accuracy often measured on overlapping splits | De-duplicated, seeded, stratified 70/15/15 split with a manifest. The test set is used exactly once and SHA-256 hashes are recorded |
| Why did the model decide that? | Black box | **Grad-CAM** heatmap for every plant |
| How much do we save? | Rarely measured | Chemical (mL), cost (₹), flight distance and flight time, each compared with blanket spraying |
| Can a drone use the output? | No | JSON command queue plus **QGroundControl `.waypoints`** files (MAVLink `DO_SPRAYER`), tested in a software flight simulator and a 3D mission simulator |
| Can a pilot take over? | Rarely modelled | Keyboard override at any moment; on hand-back the autopilot re-plans the shortest route from the drone's position and remaining tank |
| Does it avoid double spraying? | No | Re-treatment interval check by field position |

### Measured results

- **Module 3 test set:** the numbers are in
  `models/classification/eval/metrics.json` and `docs/MODULE3_REPORT.md`.
  They come from 752 held-out images that were never used for training or
  for choosing the model.
- **End-to-end demo:** run `python scripts/run_demo.py`. It builds a 63-plant
  synthetic field, then reports detection recall, classification accuracy,
  localisation error in metres, the chemical and cost saved, and the flight
  simulation.

We deliberately do **not** claim "better accuracy than every existing model".
PlantVillage accuracies above 95 % are common in the literature, and an
honest 70/15/15 number is more defensible in a viva than a copied 99 %.
SmartSpray is better as a **system**: it goes from diagnosis to a justified
action, and it is explainable, measurable and reproducible.

---

## 2. Why software first, not hardware

1. **The hard, new part is the decision, not the machine.** Pumps, nozzles
   and drones are mature, commercially available technology. What is
   missing is knowing which plant to spray, with what, how much, and why.
   That intelligence is software.
2. **Hardware-agnostic output means it can be adopted.** SmartSpray produces
   standard MAVLink missions (`QGC WPL 110`) and a plain JSON command queue.
   The same plan can drive a DJI/ArduPilot drone, a ground robot or a
   knapsack sprayer operator with a phone. Building our own drone would tie
   the project to one machine.
3. **Cost and risk for a student team.** A spray drone costs ₹3 lakh or
   more. Flying one for agriculture in India needs a DGCA Remote Pilot
   Certificate, a type-certified drone registered on Digital Sky (UIN), and
   the agriculture ministry's guidelines for pesticide spraying by drones,
   plus pesticide-handling safety measures. A software-in-the-loop simulator gives the same validation
   signals (route, tank, battery, placement error) safely and repeatably.
4. **Reproducible and testable.** 277 automated tests run in
   seconds. A hardware prototype cannot be re-run identically for a review
   panel.
5. **It scales by copying, not by manufacturing.** A software update
   improves every field it runs on. Hardware has to be built, shipped and
   maintained unit by unit.
6. **Hardware-ready by design.** Module 5 is "GPS-ready": swapping ArUco for
   RTK-GPS changes only that module. Module 9's command queue is the
   integration point for a real flight controller (MAVLink over
   telemetry/MQTT).

Hardware integration is the planned final phase, and the architecture
already has a slot for it.

---

## 3. Implementation status (about 75 %)

| Area | Status |
| --- | --- |
| M1 Field survey (registration, blur/readability check) | ✅ |
| M2 Plant detection (YOLOv8n on LeafPest; test mAP50 0.925, mAP50-95 0.805; plus OpenCV fallback) | ✅ |
| M3 Disease classification (70/15/15, calibration, TTA, Grad-CAM) | ✅ |
| M4 Severity estimation (OpenCV lesion ratio) | ✅ (heuristic; not checked against labelled severity) |
| M5 Spatial mapping (ArUco homography, auto-calibration) | ✅ |
| M6 Decision engine (rule table, uncertainty, re-treatment) | ✅ |
| M7 Resource optimisation (tank mixes, loads, cost) | ✅ |
| M8 Mission planning (nearest neighbour + 2-opt, capacity-aware sorties) | ✅ |
| M9 Drone commands (JSON queue, QGC waypoints, simulator) | ✅ (simulation only) |
| 3D mission simulator: marker placement, mapping, per-plant photos through the real model, shortest-route spraying, keyboard pilot override with re-planning (`docs/SIMULATOR.md`) | ✅ |
| API, HTML report, Streamlit dashboard, tests | ✅ |
| **Remaining 25 %** | Field imagery and field-trained models (e.g. PlantDoc, own crops); validating severity against agronomist labels; live MAVLink link and a real pump; RTK-GPS; weather-aware spray windows; multi-field accounts |

## Known limitations

- **Domain gap.** PlantVillage photos are single leaves on plain
  backgrounds. A model trained only on them can read brown soil around a
  leaf as disease. We checked this on 6 real greenhouse photos from the
  LeafPest detection test set: YOLOv8 found the plants correctly, but the
  classifier labelled most of them as diseased, and Grad-CAM shows it
  looking at yellow irrigation pipes and background. Grad-CAM is doing its
  job here by exposing the problem. Real deployment needs fine-tuning on
  labelled field crops (for example PlantDoc or our own). The uncertainty
  gate and the manual-inspection action limit the damage until then.
- **Synthetic demo field.** The demo field is made from real test-split leaf
  photos placed on a generated soil background. It demonstrates the
  pipeline; it is not a field trial.
- **Placeholder values.** Dosage values and prices in
  `rules/dosage_rules.json` are indicative engineering defaults. Check them
  against product labels and an agronomist before any field use.

## Sources for the market and regulation figures

- John Deere, "See & Spray 59 Percent Herbicide Savings" (2024 season):
  https://www.deere.com/en-us/john-deere-news/see-spray-59-percent-herbicide-savings
- XBOOM, "Agriculture Drone Price in India 2026":
  https://www.xboom.in/agriculture-drone-price-india/
- Leher, "Agri Drone Regulations and Requirements in India":
  https://www.leher.ag/blog/agri-drone-regulations-requirements-india
