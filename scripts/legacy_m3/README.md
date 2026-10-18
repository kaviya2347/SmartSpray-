# Legacy Module 3 scripts (superseded)

These files belong to the first Module 3 attempt and are kept only for the
audit trail. **Do not use them for reported numbers.**

| File | Why it is superseded |
| --- | --- |
| `analyze_splits.py`, `evaluate_classification.py` | Re-create the notebook's nested `ImageDataGenerator(validation_split=…)` splits. The notebook's 500-image "test" set is a subset of its validation set, so its 98.4 % is not a held-out test accuracy. |
| `retrain_classification.py` | A 70/15/15 attempt that was never run (none of its outputs exist). It checks overlap by file path only, so the 9 byte-identical duplicate images in the dataset could still leak across splits. |
| `smoke_test_retrained.py`, `extract_training_history.py` | Depend on the two scripts above. |
| `sample_epoch.txt`, `pytest_*.txt` | Console captures from the original Windows run. |

They also point at `models/classification/prepared_dataset/`, which now lives at
`data/datasets/plantvillage_tomato/`.

Use **`scripts/m3_train_classifier.py`** instead. It builds a de-duplicated,
seeded, stratified 70/15/15 split and writes a manifest. The test set is
evaluated exactly once.
