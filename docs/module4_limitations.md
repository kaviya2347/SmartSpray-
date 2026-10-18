# Module 4 — Disease Severity Estimation: Known Limitations

## Severity Score Definition

`severity_score` is an image-derived infected-area ratio computed as:
  `severity_score = lesion_px / leaf_px`

where:
- `leaf_px` = pixels in the filled-leaf-body mask (green component
  expanded by 21px MORPH_ELLIPSE morphological closing)
- `lesion_px` = pixels matching disease-specific HSV/Lab thresholds,
  constrained to the leaf mask, after morphological cleanup and
  connected-component size filtering

**This score is an OpenCV heuristic estimate of infected leaf-area
proportion. It is NOT a validated agronomic or clinical severity
measurement. No severity-labeled ground-truth annotations exist.**

## Severity Band Thresholds

The severity categories are:
- Low:    severity_score < 0.20
- Medium: 0.20 ≤ severity_score < 0.50
- High:   severity_score ≥ 0.50

**These thresholds are engineering design parameters from the original
SmartSpray architecture specification. They are NOT empirically calibrated
against plant pathology severity standards (e.g., Horsfall-Barratt scale)
or expert agronomic assessments.**

## Disease-Specific Limitations

### Septoria Leaf Spot
On the 60-image frozen diagnostic (seed 42), 1/60 images produced
nonzero lesion_px. Septoria's characteristic small white-centered spots
are not reliably detected by the current HSV/Lab profile.
- **Module 6 should NOT rely on severity_score for septoria treatment
  decisions.** Apply a conservative default dosage for any plant classified
  as septoria_leaf_spot by Module 3, regardless of severity_score value.

### Late Blight
On the 60-image frozen diagnostic, 25/60 images returned is_valid=False
due to fill_ratio < 0.20. This occurs when severe necrosis reduces visible
green leaf tissue below the minimum threshold for reliable leaf segmentation.
- **is_valid=False does NOT mean severity is zero or low.** For late blight
  plants with is_valid=False, Module 6 should assume high severity and apply
  maximum treatment dosage.

## is_valid=False — Module 6 Handling

When the estimator returns is_valid=False, the severity_score is 0.0 by
convention (the default from _invalid_result). This 0.0 does NOT represent
a measurement of zero infection. It represents a segmentation failure.

**Module 6 must NOT treat is_valid=False as "low severity" or "no treatment."**
Recommended Module 6 behavior for is_valid=False results:
- If disease_label is "healthy": treat as normal no-spray (healthy short-circuit
  always produces is_valid=True, so this case should not arise)
- If disease_label is any disease: apply a conservative/default dosage as if
  severity were "medium" or "high," depending on the disease

## Healthy Short-Circuit and M3 Cascade Risk

The estimator short-circuits to severity_score=0.0 for disease_label="healthy."
This means M4's correctness for healthy plants depends entirely on M3's
disease classification accuracy (96.68% on the frozen test set of 753 images).

A Module 3 misclassification of a healthy plant as early_blight or late_blight
will produce a false-positive severity score from M4, because those disease
profiles activate on healthy green tissue (confirmed by cross-probe analysis:
early_blight profile fires at mean 5,386 px and late_blight at mean 3,619 px
on healthy images).

This is a known cascade risk. It cannot be mitigated within M4.

## End-to-End Pipeline Testing

The pipeline's Stage 3 (classification) is currently not_implemented.
End-to-end automated testing of M1→M2→M3→M4 requires Stage 3 pipeline
integration, which has not yet been implemented.
