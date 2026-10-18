# Drone mission simulator

A 3D, browser-based simulation of the full mission: the drone marks the
field, maps it, photographs every plant, decides which plants need spray and
how much, flies the shortest route to spray them, and lands. A pilot can take
over from the keyboard at any moment and hand control back; the autopilot
then re-plans from wherever the drone is.

```
uvicorn app.main:app --reload
open http://127.0.0.1:8000/simulator/
```

On an older laptop, add `?quality=low` to the address (turns off shadows and
anti-aliasing).

---

## What happens in a mission

| # | Phase | What the drone does | SmartSpray module behind it |
|---|---|---|---|
| 1 | **Place markers** | Takes off, flies to each corner the operator marked on the map, drops to 1.2 m and releases an ArUco marker (ids 0–3). | — |
| 2 | **Map field** | Climbs to 15 m over the centre and takes one overview photo. The four markers give the boundary and the true field size; every plant becomes a data point in metres. | M1 survey + photo quality check, M5 ArUco homography, M2 plant detection |
| 2b | Row check | Plants are in rows at regular spacing, so a hole in a row is probably a plant the detector missed. Those spots are added to the low pass. | `find_row_gaps` |
| 3 | **Scan plants** | Flies row by row at 2.5 m and takes a close-up photo over each data point. Each photo is classified while the drone flies on. | M2 locate in photo, M3 EfficientNetB0 (+ calibrated confidence, Grad-CAM), M4 severity, M6 rules → dose |
| 4 | **Plan route** | Healthy and mild plants are skipped. Unsure plants (confidence < 0.60) go to manual inspection and are never sprayed automatically. The spray targets are routed nearest-neighbour + 2-opt, split by treatment and tank size. | M6 decisions, M7 tank mixes, M8 capacity-aware route, M9 QGroundControl export |
| 5 | **Spray** | Lands, loads the first mix, flies the route at 1.6 m and sprays each target its own dose (5 mL/s nozzle flow). Returns to refill between tank loads; swaps the battery if it is below 50 %. | Executor (browser) |
| 6 | **Return & land** | Lands at home and shows the mission report: chemical vs blanket spraying, route vs a lawn-mower pass, battery, and a check against the synthetic field's ground truth. | `/sim/{id}/finish` |

## Pilot override (keyboard)

| Key | Action |
|---|---|
| `M` | Take manual control / hand back to the autopilot |
| `W` `A` `S` `D` (or arrow keys) | Forward, left, back, right (relative to the drone's heading) |
| `Q` `E` | Turn left / right |
| `R` `F` | Climb / descend (hold `F` on the ground to stop the motors) |
| `Shift` | Fly faster (4.5 m/s instead of 2.2 m/s) |
| `Space` | Spray (hold). Dose goes to the plant under the drone. |
| `H` | Return to home and land |
| `P` | Pause |
| `1` `2` `3` `4` / `0` | Camera: follow, top, orbit (mouse), FPV / automatic |

While the pilot is flying:

- the autopilot is paused and the drone holds position when no key is pressed;
- a **geofence** keeps the drone within 3 m of the marked field;
- hand-sprayed millilitres are credited to the plant under the drone (and
  flagged if it was not a spray target);
- pressing `M` again during the spray phase asks the backend for a **new
  shortest route** from the drone's current position, using what is left in
  the tank (`POST /sim/{id}/replan`). Targets the pilot already sprayed are
  dropped from the route.

## What is real and what is simulated

| Real SmartSpray code | Simulated |
|---|---|
| ArUco calibration (M5), plant detection (M2), disease model and calibration (M3), Grad-CAM, severity (M4), dosage rules (M6), tank mixes (M7), route optimiser (M8), waypoint export (M9) | The field (real PlantVillage **test-split** photos on generated soil), the camera images, the drone's flight dynamics, battery model and spray nozzle |

Every simulated mission is stored like a real survey (`survey_id = sim_…`),
so it also appears in the dashboard and the API (`/decisions/{id}`,
`/mission/{id}`, `/drone/mission/{id}/{sortie}.waypoints`).

## API used by the page

| Endpoint | Step |
|---|---|
| `POST /sim/session` `{seed, preset}` | New synthetic field (`small` = 63 plants, `medium` = 117) |
| `POST /sim/{id}/marker` | A marker landed at (x, y) |
| `POST /sim/{id}/overview` | Overview photo → boundary, field size, plant positions, scan order |
| `POST /sim/{id}/capture` | Close-up of one plant → disease, confidence, severity, decision, Grad-CAM |
| `POST /sim/{id}/plan` | Decisions + shortest capacity-aware route + waypoint files |
| `POST /sim/{id}/replan` | Shortest route from the current position and tank |
| `POST /sim/{id}/finish` | Flight log + mission summary |
| `GET /sim/{id}/files/{name}` | `overview.jpg`, `closeups/plant_###.jpg`, `flight_log.json`, `sim_summary.json` |

## Files written per mission

```
data/images/sim_<id>/overview.jpg            the overview photo (input to M1/M2/M5)
data/outputs/sim_<id>/closeups/              every close-up photo and Grad-CAM overlay
data/outputs/sim_<id>/mission/               drone_commands.json + QGroundControl .waypoints
data/outputs/sim_<id>/flight_log.json        events, flight track, ml sprayed per plant
data/outputs/sim_<id>/sim_summary.json       mission report incl. ground-truth check
```

Old missions can be removed with `python scripts/clean_simulations.py`
(keeps the 5 most recent; `--keep 0` removes all).

## Limitations

- The field is synthetic and flat; there is no wind, GPS error or obstacle
  avoidance. Marker drops land within a few centimetres of the release point.
- Close-up photos are crops of a high-resolution ground texture, not a
  rendered camera with perspective and motion blur.
- The PlantVillage-trained classifier is only reliable on photos like its
  training data (see the domain-gap note in `docs/WHY_SMARTSPRAY.md`).
- Dosages, prices, battery and tank figures in `rules/` are placeholders.
