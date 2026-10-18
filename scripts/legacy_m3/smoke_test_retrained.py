"""Smoke-test best_retrained.keras through the PRODUCTION classifier.py."""
import sys
from pathlib import Path

# Make `app` importable regardless of where the script is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import cv2

from app.classification.classifier import DiseaseClassifier
RETRAINED = ROOT / "models/classification/evaluation/retrained"
PREPARED = ROOT / "models/classification/prepared_dataset"

# Load the untouched test split manifest
test_df = pd.read_csv(RETRAINED / "test_split.csv")

# Load the NEW model through the production class
clf = DiseaseClassifier(weights_path=ROOT / "models/classification/best_retrained.keras")
print("Model loaded via production DiseaseClassifier. ready =", clf.is_ready())
print("class_order (from classifier) =", clf.class_order)

# Predict on a few test images (they are in the untouched split)
correct = 0
total = 0
for _, row in test_df.head(20).iterrows():
    img_path = PREPARED / row["filepath"]
    img_bgr = cv2.imread(str(img_path))
    result = clf.predict(img_bgr)  # BGR uint8 crop, exactly like service.py does
    expected = row["label"]
    total += 1
    ok = result["label"] == expected
    correct += int(ok)
    print(f"  {ok and 'OK ' or 'ERR'} true={expected:22s} pred={result['label']:22s} conf={result['confidence']:.4f}")

print(f"\nSmoke-test subset accuracy (first 20 untouched test images): {correct}/{total}")
print("PRODUCTION CLASSIFIER COMPATIBILITY: PASSED" if total and correct / total >= 0.5 else "FAILED")