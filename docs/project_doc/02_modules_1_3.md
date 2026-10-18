# Module 1 — Field survey

**Purpose.** Register a survey (a folder of images of one field) and check that the images are usable before any analysis runs.

**Input.** Images in `data/images/<survey_id>/` (`.jpg`, `.jpeg`, `.png`) plus the crop name and the field size in metres.
**Output.** One row in `surveys` and one row per image in `survey_images`.

**How it works.**

1. The survey id is validated (non-empty, a plain folder name: no `/`, `\` or leading `.`, which prevents path traversal).
2. The folder is listed; a survey with no images is rejected; a survey id can only be registered once.
3. Each image is checked by `assess_image()`: it is loaded in greyscale and its **sharpness** is measured as the variance of the Laplacian (a second-derivative edge filter). Sharp images have strong edges and a high variance; blurred images have a low one. Images scoring below **100** are flagged `is_blurry`. Images OpenCV cannot decode are flagged unreadable.
4. The survey row stores the capture date, crop (default "Tomato"), field size string (e.g. `10m x 8m`), the number of images and the creation time.

```python
def assess_image(path: Path, blur_threshold: float = BLUR_THRESHOLD) -> dict:
    """Readability + sharpness check for one image (never raises)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"image_filename": path.name, "is_readable": False, ...}
    score = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return {"image_filename": path.name, "is_readable": True,
            "is_blurry": score < blur_threshold, "blur_score": round(score, 2), ...}
```

| Item | Value |
|---|---|
| Code | `app/field_survey/service.py`, `app/field_survey/router.py` |
| Tables | `surveys` (survey_id, capture_date, crop, field_size, num_images, created_at); `survey_images` (survey_id, image_filename, width_px, height_px, blur_score, is_readable, is_blurry, checked_at) |
| API | `POST /survey/register` `{"survey_id", "crop", "field_width_m", "field_height_m"}` · `GET /survey/{survey_id}` |
| Tests | `tests/test_field_survey.py` (registration, duplicates, quality check, clean-up) |

# Module 2 — Plant detection

**Purpose.** Find every plant in each survey image, draw a bounding box around it, and give it a stable `plant_id` that all later modules use.

**Output.** Rows in `detections` (survey_id, image_filename, plant_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, model_version).

## Detector 1: YOLOv8n (trained model)

A YOLOv8-nano object detector, fine-tuned from COCO-pretrained weights on the LeafPest plant-detection dataset, is the default detector for real photos. YOLO divides the image into a grid and, in a single forward pass, predicts boxes and confidence scores for objects; detections below confidence 0.25 are discarded.

| Training setting | Value |
|---|---|
| Base weights | `yolov8n.pt` (COCO) |
| Dataset | LeafPest Plant Detection v1: 460 train / 57 validation / 29 test, one class |
| Image size / batch / epochs | 512 px / 8 / 30 |
| Optimiser / learning rate / seed | AdamW / 0.001 / 42 (deterministic) |
| Model selection | Ultralytics `best.pt` by validation fitness; test split evaluated once at the end |
| Script | `scripts/m2_train_detector.py` (writes `models/yolov8_plant_detector/best.pt` and `eval/metrics.json`) |

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation (57 images) | 0.952 | 0.748 | 0.911 | 0.956 |
| **Test (29 images)** | **0.925** | **0.805** | **0.997** | **0.912** |

mAP50 is the mean average precision when a predicted box counts as correct if it overlaps the true box by at least 50 % (intersection over union ≥ 0.5); mAP50-95 averages the same measure over stricter overlaps from 0.50 to 0.95.

## Detector 2: OpenCV vegetation fallback

A training-free detector (`ClassicalPlantDetector`) used when YOLO weights are unavailable and for the synthetic fields (where YOLO, trained on real photos, is not appropriate):

1. **Vegetation mask.** A pixel is vegetation if its HSV colour lies in hue 22–95, saturation ≥ 25, value ≥ 30 (green to yellow-green) **or** its excess-green index ExG = 2G − R − B exceeds 20.
2. **Clean-up.** Morphological opening with a small elliptical kernel (k = 1 % of the shorter image side, odd) removes speckle; closing with a 2k kernel fills lesions and gaps inside a plant; a second opening with the 2k kernel cuts thin "necks" between touching neighbours.
3. **Components.** Each connected blob between 0.04 % and 25 % of the image area is one plant.
4. **Boxes.** The blob's box is padded by 8 % and made square, so Module 3 does not distort the leaf when it resizes the crop to 224 × 224. Confidence = 0.5 + 0.5 × (blob area / box area).

`kernel_frac` (default 0.01) can be lowered when plants are small relative to the frame; the simulator's whole-field overview uses 0.004.

## Plant IDs (identity resolver)

Detectors return boxes in arbitrary order, so `assign_plant_ids()` sorts them spatially before numbering: boxes are sorted by their vertical centre; a box joins the current row if its centre is within **half the median box height** of the row's mean y; each row is then sorted left to right. Plant #1 is therefore always the top-left plant, and re-running the same image gives the same IDs. (The original version used a fixed 60-pixel band, which split tilted or high-resolution rows.) IDs continue across images of the same survey so every plant id is unique per survey.

| Item | Value |
|---|---|
| Code | `app/detection/detector.py` (both detectors, `get_detector(mode)`), `app/detection/identity_resolver.py`, `app/detection/router.py` |
| Detector modes | `yolo` (default), `auto` (YOLO if weights exist, else fallback), `classical`; or environment variable `SMARTSPRAY_DETECTOR` |
| API | `POST /detect` `{"survey_id", "detector"}` · `GET /detect/health` |
| Tests | `tests/test_detection.py` (28 tests: boxes, IDs, tilted rows, fallback, DB, API) |

# Module 3 — Disease classification

Module 3 is the core of the system. It decides, for every plant, which of four classes it belongs to, how confident that decision is, whether the confidence is high enough to act on, and which parts of the image drove the decision.

## Classes

The class order is fixed in `models/classification/class_order.json` and checked at start-up against the code (`app/classification/prediction.py`): **0 early_blight, 1 healthy, 2 late_blight, 3 septoria_leaf_spot**. If the two ever disagree the classifier refuses to load, so predictions can never be silently mislabelled.

## Model architecture

Transfer learning on **EfficientNetB0**, a convolutional network designed by compound scaling of depth, width and resolution (Tan & Le, 2019). It gives ImageNet-level features with only about 4 million parameters, so it runs comfortably on a laptop CPU or an edge device.

| Layer | Output shape | Parameters | Notes |
|---|---|---|---|
| `image` (input) | 224 × 224 × 3 | 0 | RGB, values 0–255 |
| `efficientnetb0` (backbone) | 7 × 7 × 1280 | 4,049,571 | 237 layers; ImageNet weights; includes its own rescaling and normalisation layers |
| `gap` GlobalAveragePooling2D | 1280 | 0 | one value per feature map |
| `dropout_head_1` | 1280 | 0 | rate 0.3 |
| `dense_256` Dense, ReLU | 256 | 327,936 | |
| `dropout_head_2` | 256 | 0 | rate 0.2 |
| `predictions` Dense, softmax | 4 | 1,028 | class probabilities |
| **Total** | | **4,378,535** | |

```python
inputs = keras.Input(shape=(224, 224, 3), name="image")
x = base(inputs, training=False)           # BatchNorm stays in inference mode
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout_head_1")(x)
x = layers.Dense(256, activation="relu", name="dense_256")(x)
x = layers.Dropout(0.2, name="dropout_head_2")(x)
outputs = layers.Dense(4, activation="softmax", name="predictions")(x)
```
(`app/classification/model_factory.py`)

## Training procedure

Training is done by `scripts/m3_train_classifier.py` in three phases. Loss: sparse categorical cross-entropy; optimiser: Adam; batch size 32; seed 42. Early stopping (patience 5; phase 3: 3) watches validation accuracy; ReduceLROnPlateau multiplies the learning rate by 0.5 (phase 3: 0.3) when validation loss stops improving (patience 3; phase 3: 2). A custom `ValidationSelector` callback saves the best checkpoint seen across **all** phases, judged on validation accuracy with ties broken by validation loss; the test split is not even loaded during training.

| Phase | What is trained | Learning rate | Max epochs | Epochs run | Time (2-core CPU) | Best validation accuracy |
|---|---|---|---|---|---|---|
| 1 | Head only (backbone frozen): 328,964 trainable parameters | 1e-3 | 20 | 20 | 34.4 min | 0.9667 (epoch 18) |
| 2 | Top 30 backbone layers + head; BatchNorm kept frozen (10 weighted layers, 1,812,324 trainable) | 1e-5 | 10 | 10 | 17.8 min | 0.9747 |
| 3 | Blocks 5–7 + top (from `block5a_expand_conv`, 118 layers; 41 weighted layers, 3,994,028 trainable); accepted only if validation improves | 1e-4 | 8 | 4 | 9.6 min | **0.9800** (epoch 1) |

Why BatchNorm stays frozen: with small batches, updating BatchNorm statistics during fine-tuning destroys the ImageNet statistics and hurts accuracy; keeping them frozen is standard Keras transfer-learning practice.

![Training and validation accuracy / loss across the three phases](models/classification/eval/training_curves.png)

## Model selection (validation only)

After all phases, the checkpoint with the **highest validation accuracy** (ties broken by the lowest validation loss) is selected: phase 3, epoch 1 — validation accuracy 0.9800, validation loss 0.0609. The test split is never used to choose anything.

## Inference settings chosen on validation

Two inference options were evaluated on the validation split and kept only because they helped there:

- **Flip test-time augmentation (TTA).** The model also sees the mirrored image and the two softmax outputs are averaged. Validation accuracy 0.9800 → 0.9814, validation negative log-likelihood (NLL) 0.0609 → 0.0574. **Kept.**
- **Temperature scaling** (Guo et al., 2017). Neural networks are often over- or under-confident. A single scalar T is fitted on the validation set to minimise NLL, and probabilities are re-scaled as p_cal = softmax(log p / T). Fitted **T = 0.9010** (validation NLL 0.05742 → 0.05685; expected calibration error 0.0091 → 0.0082). T < 1 means the raw model was slightly *under*-confident. Temperature scaling never changes which class wins; it only makes the confidence number trustworthy, which matters because the decision engine acts on that number.

Both settings are stored in `models/classification/inference_config.json` and loaded automatically by the backend.

## Test-set evaluation (run once)

| Metric | Value |
|---|---|
| Test images | 752 (never used in training or selection) |
| Correct / incorrect | 732 / 20 |
| **Accuracy** | **0.9734** |
| Test loss | 0.0546 |
| Macro precision / recall / **F1** | 0.9683 / 0.9682 / **0.9681** |
| Weighted F1 | 0.9733 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| early_blight | 0.924 | 0.917 | 0.921 | 120 |
| healthy | 0.990 | 1.000 | 0.995 | 190 |
| late_blight | 0.995 | 0.961 | 0.978 | 229 |
| septoria_leaf_spot | 0.964 | 0.995 | 0.979 | 213 |

**Confusion matrix** (rows = true class, columns = predicted):

| True \ Predicted | early_blight | healthy | late_blight | septoria |
|---|---|---|---|---|
| early_blight | **110** | 1 | 1 | 8 |
| healthy | 0 | **190** | 0 | 0 |
| late_blight | 8 | 1 | **220** | 0 |
| septoria | 1 | 0 | 0 | **212** |

![Confusion matrix on the 752-image test set](models/classification/eval/confusion_matrix.png)

**Production mode** (flip-TTA + temperature, settings frozen from validation): accuracy **0.9787**; expected calibration error 0.0150 → **0.0107**; 742 of 752 plants (98.7 %) have confidence ≥ 0.60 and **98.65 %** of those are correct; the other 10 are flagged uncertain (6 of the 10 were wrong, so the gate catches most errors).

![Reliability diagram: predicted confidence vs observed accuracy, before and after temperature scaling](models/classification/eval/reliability_diagram.png)

**Error analysis.** Most errors are between the two "brown lesion" diseases: 8 early-blight leaves predicted as septoria and 8 late-blight leaves predicted as early blight. These diseases look alike at early stages, and early blight is the smallest class (800 raw images). Healthy leaves are almost never confused with disease (recall 1.000), which is what protects healthy plants from being sprayed.

## How a plant is classified inside the backend

This section follows one plant through the actual code.

![Module 3 inference flow in the backend](docs/figures/fig_m3_inference.png)

**1. Request.** The API call `POST /classify {"survey_id": "demo_survey_001"}` (or the pipeline, or the simulator) calls `classify_survey(survey_id)` in `app/classification/service.py`.

**2. Fetch detections.** The service opens SQLite, ensures the tables exist, and reads every row of `detections` for the survey (written by Module 2). If there are none it stops with "Run Module 2 first". Old classification rows for the survey are deleted so a re-run does not create duplicates.

**3. Load the model once.** `get_classifier()` returns a **singleton** `DiseaseClassifier`: the first call loads `models/classification/best.keras`, validates `class_order.json` against the code, and reads `inference_config.json` (T = 0.901, TTA on, threshold 0.60). Later calls reuse the loaded model.

**4. Crop.** For each detection, the original survey image is loaded with OpenCV and the plant's bounding box is cropped (coordinates are clamped to the image; empty boxes are rejected).

**5. Pre-process** (`DiseaseClassifier._prepare`): convert OpenCV's BGR to RGB, resize to 224 × 224, convert to float32 in the 0–255 range. Pixel scaling and ImageNet normalisation are done *inside* the model by EfficientNetB0's own layers, so no manual normalisation is needed (and none can be forgotten).

**6. Forward pass with TTA** (`_forward`): the crop and its mirror image go through the network **in one batch** using a compiled `tf.function` (identical numbers to `model.predict()`, about 3× faster for single crops). The two softmax vectors are averaged.

```python
def _forward(self, batch):
    if isinstance(self._model, tf.keras.Model):
        if self._compiled is None:
            self._compiled = tf.function(lambda x: model(x, training=False),
                                         reduce_retracing=True)
        n = len(batch)
        x = np.concatenate([batch, batch[:, :, ::-1, :]]) if self._tta else batch
        out = np.asarray(self._compiled(tf.convert_to_tensor(x, tf.float32)))
        return (out[:n] + out[n:]) / 2.0 if self._tta else out
```

**7. Calibrate.** The averaged probabilities are re-scaled with the temperature:

```python
def _softmax_with_temperature(probs, temperature):
    z = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

**8. Decide.** label = class with the highest calibrated probability; confidence = that probability; `uncertain = confidence < 0.60`. The result is a dictionary:

```json
{"label": "late_blight", "confidence": 0.9999,
 "all_probs": {"early_blight": 0.0, "healthy": 0.0001,
               "late_blight": 0.9999, "septoria_leaf_spot": 0.0},
 "raw_confidence": 0.9998, "uncertain": false}
```

**9. Store.** A `ClassificationResult` is saved to the `classifications` table (survey_id, image_filename, plant_id, bbox, label, confidence, all_probs_json, model_version, is_uncertain, created_at).

**10. Explain (optional).** With `save_gradcam=true`, a Grad-CAM overlay is written to `data/outputs/<survey_id>/gradcam/plant_###.jpg`.

**11. Hand-off.** Module 4 reads the label to choose the lesion profile; Module 6 reads label, confidence and `is_uncertain` to decide the action. An uncertain plant is always sent to manual inspection, never sprayed automatically.

### Worked example (terminal)

`scripts/inspect_classifier.py` runs the exact same code path on one image and prints every intermediate value. Run it during the review:

```bash
python scripts/inspect_classifier.py --label late_blight --seed 3
python scripts/inspect_classifier.py path/to/any_leaf.jpg
python scripts/inspect_classifier.py --survey demo_survey_001 --plant 12
```

Output (abridged; class names shortened) for a test-split late-blight photo:

```text
STEP 1 · Load the trained model
model file      : models/classification/best.keras
SHA-256         : 499ceca37640900d7d2771dbac1c5f55cf30796875a56b6f4ec584918d92614e
model version   : efficientnetb0-tomato-disease-v2-701515
parameters      : 4,378,535
layers          : image -> efficientnetb0 -> gap -> dropout -> dense_256 -> dropout -> predictions
inference config: temperature T = 0.9010, flip-TTA = True, uncertainty threshold = 0.6
STEP 3 · Pre-processing
BGR -> RGB, resize to 224x224, float32 -> (224, 224, 3), range 1..221
STEP 4 · Forward pass
original image  : early 0.0000  healthy 0.0002  late 0.9998  septoria 0.0000
mirrored image  : early 0.0000  healthy 0.0001  late 0.9999  septoria 0.0000
averaged (TTA)  : early 0.0000  healthy 0.0002  late 0.9998  septoria 0.0000
STEP 5 · Confidence calibration   p_cal = softmax(log(p) / T)
calibrated      : early 0.0000  healthy 0.0001  late 0.9999  septoria 0.0000
STEP 6 · Prediction
=> late_blight at 100.0% : confident (>= threshold)
=> CORRECT (true label late_blight)
STEP 7 · Severity (Module 4)   score 0.000 -> low (valid=False: leaf fill < 0.20)
STEP 8 · Treatment decision (Module 6)
action SPRAY | dose 10.0 mL | treatment late_blight_fungicide
reason: Severity could not be measured; late_blight is treated by default. ...
STEP 9 · Grad-CAM saved: data/outputs/inspect/gradcam_<name>.jpg
```

![Saved by the script: the input photo (left) and its Grad-CAM heatmap (right) — red areas pushed the late-blight score up](docs/figures/inspect_gradcam.jpg)

## Grad-CAM explanations

Grad-CAM (Selvaraju et al., 2017) shows which regions of the image pushed the score for the predicted class up:

1. Run the backbone to get the last convolutional feature map A (7 × 7 × 1280).
2. Run the head and take the class **logit** (pre-softmax score) yᶜ. The logit is used instead of the probability so that very confident predictions still give useful gradients.
3. Compute the gradient ∂yᶜ/∂A with `tf.GradientTape`.
4. Weight each of the 1280 feature maps by the mean of its gradient: αₖ = mean(∂yᶜ/∂Aₖ).
5. Heatmap = ReLU(Σₖ αₖ Aₖ), normalised to 0–1 and upsampled to the crop size, then blended over the image with a JET colour map.

The computation is compiled once per model with `tf.function` (≈ 50 ms instead of ≈ 730 ms on a 2-core CPU) so the simulator can explain every plant as the drone photographs it. Code: `app/classification/explain.py`.

![Grad-CAM examples: the model attends to lesions on diseased leaves](models/classification/eval/gradcam_examples.png)

Grad-CAM also exposed the system's main limitation: on six real greenhouse photos, the classifier labelled most plants as diseased and the heatmaps showed it looking at yellow irrigation pipes and background rather than leaves (see Chapter 21).

## Reproducibility and safe promotion

| Record | Where |
|---|---|
| Split manifest (every file, split, label, SHA-256) | `models/classification/split_manifest.json` / `.csv` |
| Training history per phase, hyper-parameters, environment (Python 3.11.15, TensorFlow 2.21.0, Keras 3.15.0) | `models/classification/eval/training_history.json`, `history_phase*.csv`, `run_log.txt` |
| Test metrics, per-class report, confusion matrix, all predictions, misclassified list | `models/classification/eval/metrics.json`, `classification_report.*`, `confusion_matrix.*`, `test_predictions.csv`, `misclassified.csv` |
| Test-set lock | `models/classification/eval/TEST_EVALUATED.lock` |
| Promotion record: old model SHA-256 `4022becc…` backed up as `best.keras.bak-20260916-061314`; new model SHA-256 `499ceca3…` verified (output shape [None, 4], probabilities sum to 1) before copying | `models/classification/eval/promotion_record.json` |
| Executed report notebook | `notebooks/m3_classification_701515_report.ipynb` |
| Written report | `docs/MODULE3_REPORT.md` |

The ImageNet backbone weights could not be downloaded from Google's server in the build environment, so they were recovered from the frozen (never-trained) backbone of the original model; all its layers were verified as non-trainable with ImageNet normalisation values, so the starting point is the standard ImageNet initialisation.

| Item | Value |
|---|---|
| Code | `app/classification/classifier.py` (inference), `model_factory.py` (architecture), `explain.py` (Grad-CAM), `prediction.py` (data contracts, class order, threshold), `service.py` (survey workflow), `database.py`, `router.py` |
| Model files | `models/classification/best.keras`, `class_order.json`, `inference_config.json` |
| API | `POST /classify` `{"survey_id", "save_gradcam"}` · `GET /classify/results/{survey_id}` · `GET /classify/health` |
| Tests | `tests/test_classification.py` (36 tests: data contracts, class-order guard, calibration, TTA, uncertainty, DB, API) |
