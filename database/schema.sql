-- database/schema.sql
-- Core SmartSpray tables. Tables owned by a module are also created
-- idempotently by that module (see app/db.py:init_db).

CREATE TABLE IF NOT EXISTS surveys (
    survey_id     TEXT PRIMARY KEY,
    capture_date  TEXT NOT NULL,
    crop          TEXT NOT NULL,
    field_size    TEXT NOT NULL,
    num_images    INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Module 4 severity results (owned by Module 4; added 2026-07-13).
-- Authoritative severity bands (app/severity/estimator.py SEVERITY_BANDS,
-- matching the project design / Review-1 slides):
--   [0.00, 0.20) -> "low",  [0.20, 0.50) -> "medium",  [0.50, 1.00] -> "high"
-- (An earlier draft comment here said 0.15/0.50; the code has always used
--  0.20/0.50, so the comment was corrected on 2026-09-16.)
CREATE TABLE IF NOT EXISTS plant_severity (
    severity_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id         TEXT    NOT NULL,
    plant_id          INTEGER NOT NULL,
    image_filename    TEXT    NOT NULL,
    disease_label     TEXT    NOT NULL,
    severity_score    REAL    NOT NULL,
    severity_category TEXT    NOT NULL,
    leaf_px           INTEGER NOT NULL,
    lesion_px         INTEGER NOT NULL,
    is_valid          INTEGER NOT NULL DEFAULT 1,
    model_version     TEXT    NOT NULL,
    estimated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (survey_id, plant_id)
);

-- Module 5 spatial mapping tables (owned by Module 5; added 2026-07-14).

CREATE TABLE IF NOT EXISTS plant_locations (
    location_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id     TEXT    NOT NULL,
    plant_id      INTEGER NOT NULL,
    image_filename TEXT   NOT NULL,
    bbox_cx_px    REAL    NOT NULL,  -- bounding box center x, pixels
    bbox_cy_px    REAL    NOT NULL,  -- bounding box center y, pixels
    field_x_m     REAL,              -- real-world x coordinate (meters from field origin)
    field_y_m     REAL,              -- real-world y coordinate (meters from field origin)
    is_mapped     INTEGER NOT NULL DEFAULT 0,  -- 1 if real coordinates computed, 0 if not
    calibration_id TEXT,             -- which calibration was used (NULL = no calibration)
    mapping_mode  TEXT    NOT NULL DEFAULT 'uncalibrated',
                                     -- 'calibrated' | 'uncalibrated' | 'simulation'
    model_version TEXT    NOT NULL,
    mapped_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (survey_id, plant_id)
);

CREATE TABLE IF NOT EXISTS spatial_calibrations (
    calibration_id    TEXT    PRIMARY KEY,  -- e.g. "cal_survey_001_2026-01-01"
    survey_id         TEXT    NOT NULL,
    image_filename    TEXT    NOT NULL,     -- which image contains the ArUco markers
    field_width_m     REAL    NOT NULL,     -- physical field width in meters
    field_height_m    REAL    NOT NULL,     -- physical field height in meters
    aruco_dict        TEXT    NOT NULL,     -- e.g. "DICT_4X4_50"
    marker_count      INTEGER NOT NULL,     -- how many markers were detected (must be >= 4)
    homography_json   TEXT    NOT NULL,     -- 3x3 homography matrix as JSON array
    reprojection_error REAL,               -- mean reprojection error in pixels
    is_valid          INTEGER NOT NULL DEFAULT 1,
    calibrated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

/*
┌─────────────────────────────────────────────────────┐
│                    INPUT                            │
│  Drone → Aerial Image Capture + GPS + Flight Meta   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 1: FIELD SURVEY                              │
│  ▪ Blur filter → reject unusable frames              │
│  ▪ Store images: data/images/{survey_id}/            │
│  ▪ Register plant grid (plant_id ↔ position)         │
│  ▪ Create survey record in database                  │
│  Output: Validated images + survey_id + plant_ids    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 2: PLANT DETECTION (YOLOv8)                  │
│  ▪ YOLOv8n inference → bounding boxes                │
│  ▪ Confidence threshold (≥ 0.25)                     │
│  ▪ NMS → remove duplicates                           │
│  ▪ Identity resolution → assign plant_ids            │
│  ▪ Crop individual plants from image                 │
│  Output: {plant_id, bbox [x1,y1,x2,y2], confidence} │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 3: DISEASE CLASSIFICATION (EfficientNetB0)   │
│  ▪ BGR→RGB→Resize(224×224)→float32→preprocess_input  │
│  ▪ Model.predict() → softmax(4 classes)              │
│  ▪ argmax → disease_label + confidence               │
│  ▪ Confidence < 0.60 → flag as uncertain             │
│  Output: {plant_id, disease_label, confidence}       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 4: SEVERITY ESTIMATION (OpenCV)              │
│  ▪ HSV/Lab thresholding → lesion mask                │
│  ▪ Morphological open/close → noise removal          │
│  ▪ Infected ratio = lesion_px ÷ leaf_px              │
│  ▪ Map to Low (<20%) / Medium (20–50%) / High (>50%) │
│  Output: {severity_score, severity_category}         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 5: SPATIAL MAPPING (Homography)              │
│  ▪ ArUco markers at field corners (known positions)  │
│  ▪ cv2.detectMarkers() → pixel positions of markers  │
│  ▪ cv2.findHomography() → transformation matrix H   │
│  ▪ H × (px, py) → (field_x_m, field_y_m)            │
│  Output: {plant_id, field_x_m, field_y_m}           │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  MODULE 6: DECISION ENGINE (Rule-Based)              │
│  ▪ Load dosage_rules.json                            │
│  ▪ Lookup (disease_label × severity_band) → action  │
│  ▪ Apply re-treatment interval check                 │
│  ▪ Generate: action + chemical_type + dosage         │
│  ▪ Route optimizer: minimize total travel distance   │
│  Output: {plant_id, location, action, dosage}        │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                    OUTPUT                            │
│  JSON Treatment Plan → Drone / Executor              │
│  {"plant_id":17, "location":[3.4,2.2],               │
│   "action":"spray", "dosage":9}                      │
└──────────────────────────────────────────────────────┘