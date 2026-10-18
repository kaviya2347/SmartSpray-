# Module 5 - Spatial Mapping: Known Limitations

## What Module 5 Actually Produces

Per plant, Module 5 stores one plant_locations row containing:

- bbox_cx_px, bbox_cy_px - the pixel centre of the Module 2 bounding box
- field_x_m, field_y_m - metres from the field origin, or NULL
- is_mapped - 1 when real coordinates were computed, 0 otherwise
- mapping_mode - calibrated | uncalibrated | simulation
- calibration_id - the homography used, or NULL

Two modes exist and they are never mixed:

    calibrated    - a valid ArUco calibration exists: coordinates computed via H
    uncalibrated  - no valid calibration exists: coordinates are NULL

simulation is reserved for a future offline coordinate source. Nothing in
Module 5 produces it today.

**Uncalibrated rows are deliberate. Module 5 never interpolates, estimates,
or fabricates a field position.**

## The Measured Limitation: No ArUco Markers Exist

ArUco detection was verified working on synthetic marker images (4-marker
solves reproduce field corners to under 1e-6 m; the full detect-to-homography
path is covered by tests/test_spatial_mapping.py).

However, the survey imagery currently in data/images/ contains **no ArUco
markers**, and no calibration image has been captured. Consequently:

- **No real calibration can be produced from the existing dataset.**
- Every survey maps in uncalibrated mode with NULL coordinates.
- Module 5 output today is bbox_cx_px / bbox_cy_px only.

This was identified as the single largest Module 5 risk during the Stage 5
inspection (rules/severity/stage3_correction_fraction_report.md) and it
remains unresolved. Resolving it requires the operational step below, not a
code change.

### Required operational step

To obtain real field coordinates, capture at least one image per survey with
four ArUco markers placed flat on the ground at the four field corners, using:

- dictionary DICT_4X4_50
- marker id 0 at (0, 0)
- marker id 1 at (field_width_m, 0)
- marker id 2 at (field_width_m, field_height_m)
- marker id 3 at (0, field_height_m)

The correspondence point is each marker CENTRE, so marker centres must sit on
the physical corners. Then call POST /spatial/calibrate followed by
POST /spatial/map.

## Assumptions Behind the Homography

1. **Planarity.** The field is treated as a plane. Homography is exact for a
   planar scene and increasingly wrong as ground relief grows.
2. **One pose per survey.** A single H is applied to every image in the
   survey. If drone altitude, yaw, or gimbal angle differ between images, the
   per-image mapping is wrong. There is no per-image calibration.
3. **Marker centres equal field corners.** Ground-truth corner positions are
   the operator-declared field_width_m / field_height_m, not surveyed
   coordinates.
4. **Lens distortion is not modelled.** ArUco homography absorbs some radial
   error but no camera intrinsics exist in the repository, so distortion is
   uncorrected.
5. **Field origin is the id-0 marker.** All coordinates are relative to that
   marker, not to any global or geographic frame. There is no GPS anywhere in
   the project.
## The bbox Centre Is a Proxy for Plant Position

`field_x_m` / `field_y_m` locate the CENTRE OF THE DETECTED BOUNDING BOX,
not the plant stem. Module 2 detects plant/leaf regions, so the centre is the
centre of the visible canopy in that image. It is stable and independent of
whether Module 4 produced a valid severity mask, which is why it was chosen.

Where a plant is partially occluded or clipped by the image edge, its centre
is biased toward the visible portion.

## No Cross-Image Plant Identity

`plant_locations` is keyed `UNIQUE (survey_id, plant_id)`. Module 2 assigns
`plant_id` per image, so the same physical plant appearing in several images
gets several ids and therefore several rows - possibly at different mapped
positions.

**There is no plant re-identification across images and no cross-image
deduplication.** Treat `plant_locations` as per-(survey, detection) records,
not as a de-duplicated field inventory. This also means a single plant cannot
be given one authoritative coordinate by averaging its appearances.

## No Ground-Truth Validation Is Possible

No surveyed field coordinates, GPS traces, or measured plant positions exist
in the repository. The homography is verified for geometric correctness on
synthetic markers, but **no real-world positional accuracy claim (in cm or m)
can be made.**

## Guidance for Module 6

- **Do not treat NULL coordinates as `(0, 0)`.** A NULL `field_x_m` means
  "unknown", not "at the origin". Always branch on `is_mapped`.
- If `is_mapped` is 0 for a survey, Module 6 cannot build a travel route from
  field coordinates. Either fall back to per-image pixel ordering, or block
  treatment planning for that survey until a calibration exists.
- `calibration_id` on each row records which H produced the coordinate. If a
  survey is re-mapped with a new calibration, coordinates change while
  `plant_id` stays the same - version any downstream directive on
  `calibration_id`.
- Points outside `[0, field_width_m] x [0, field_height_m]` are stored as
  measured and logged as warnings, not clamped. A spray route planner should
  decide explicitly whether to skip out-of-bounds targets.

## End-to-End Pipeline Testing

Pipeline Stage 5 (spatial mapping) is implemented and covered by
`tests/test_spatial_mapping.py`, including a real mapping pass through a
temporary SQLite database. Stage 3 (classification) and Stages 6-9 remain
`not_implemented` in `app/pipeline.py`, so full M1 -> M9 end-to-end automated
testing is not yet possible.
