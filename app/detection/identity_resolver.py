"""
FILE PATH: SmartSpray/app/detection/identity_resolver.py

PURPOSE:
    Takes the raw list of RawDetection objects from detector.py
    and converts them into the SmartSpray structured format with plant_id assigned.

    The plant IDs assigned here are TEMPORARY — they are valid for one image only.
    They exist purely to give downstream modules (Module 3 onward) a stable key
    to refer to each detection within a response.

    Permanent, field-wide plant IDs (tied to GPS coordinates) will be managed
    by the Spatial Mapping module (Module 5).

INPUT:
    A list of RawDetection objects (from detector.py).

OUTPUT:
    A list of Python dicts in exactly this format:
    [
        {"plant_id": 1, "bbox": [x1, y1, x2, y2], "confidence": 0.97},
        {"plant_id": 2, "bbox": [x1, y1, x2, y2], "confidence": 0.95},
        ...
    ]
    This output is the direct input to Module 3 (Disease Classification).

DEPENDENCIES:
    - app.detection.detector: for the RawDetection dataclass
    - typing: Python standard library, for type hints
"""

from typing import Optional
from app.detection.detector import RawDetection


def assign_plant_ids(
    raw_detections: list[RawDetection],
    start_id: int = 1
) -> list[dict]:
    """
    Sort detections in a deterministic spatial order and assign sequential plant IDs.

    WHY SORTING?
    ------------
    If five plants appear in an image, YOLOv8 returns them in the order it found
    them during inference — which is not deterministic across different runs or
    different model versions.

    We sort the detections spatially (top-to-bottom, then left-to-right within
    each row) before assigning IDs. This means:
    - Plant #1 is always the top-left plant in the image.
    - If you re-run the same image, you always get the same IDs.
    - When this module is eventually extended to track plants across surveys,
      a consistent numbering system makes matching easier.

    SORTING ALGORITHM (adaptive rows):
    ----------------------------------
    1. Sort detections by the vertical centre of their box.
    2. Walk down the list; a detection joins the current row while its
       y-centre is within half a typical plant height (median box height) of
       the row's mean y-centre, otherwise it starts a new row.
    3. Inside each row, sort left to right by x-centre.

    (The original version used a fixed 60-pixel band, which splits one tilted
    or high-resolution row into several "rows" and makes IDs jump around.)

    Args:
        raw_detections: List of RawDetection objects from PlantDetector.detect().
        start_id:       ID to assign to the first plant (default 1).
                        Set this to the current highest plant_id + 1 when processing
                        multiple images in one survey, to get globally unique IDs
                        across the whole survey.

    Returns:
        List of dicts with keys: plant_id, bbox, confidence.
        Returns an empty list if raw_detections is empty.
    """
    if not raw_detections:
        return []

    def centre(det: RawDetection) -> tuple[float, float]:
        x1, y1, x2, y2 = det.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    heights = sorted(max(1, d.bbox[3] - d.bbox[1]) for d in raw_detections)
    tolerance = max(1.0, 0.5 * heights[len(heights) // 2])

    rows: list[list[RawDetection]] = []
    row_mean_y = 0.0
    for det in sorted(raw_detections, key=lambda d: (centre(d)[1], centre(d)[0])):
        _, cy = centre(det)
        if rows and abs(cy - row_mean_y) <= tolerance:
            rows[-1].append(det)
            row_mean_y = sum(centre(d)[1] for d in rows[-1]) / len(rows[-1])
        else:
            rows.append([det])
            row_mean_y = cy
    sorted_detections = [d for row in rows for d in sorted(row, key=lambda d: centre(d)[0])]

    # ── Step 2: assign sequential IDs ────────────────────────────────────────
    # Plant IDs start at start_id and increment by 1.
    # The format produced here is exactly what Module 3 expects.
    result = []
    for i, detection in enumerate(sorted_detections):
        result.append({
            "plant_id": start_id + i,
            "bbox": detection.bbox,
            "confidence": detection.confidence
        })

    return result


def resolve_survey_detections(
    per_image_detections: list[tuple[str, list[RawDetection]]]
) -> dict:
    """
    Process detections from ALL images in a survey into a single flat list
    with globally unique plant_ids across all images.

    Args:
        per_image_detections:
            A list of (image_filename, raw_detections) tuples.
            One tuple per image in the survey.

    Returns:
        A dict with:
        {
            "total_plants": int,
            "detections_by_image": {
                "IMG_001.jpg": [{"plant_id": 1, ...}, ...],
                "IMG_002.jpg": [{"plant_id": 8, ...}, ...],
                ...
            },
            "all_detections": [
                {"plant_id": 1, "image": "IMG_001.jpg", "bbox": [...], "confidence": 0.97},
                ...
            ]
        }

    Why globally unique IDs across images?
        Module 3 (Disease Classification) crops each plant from its image using the bbox.
        Module 5 (Spatial Mapping) converts each plant's bbox to a field coordinate.
        Both need to be able to say "plant #17" and have it mean the same plant
        regardless of which image that plant appeared in.
    """
    all_detections = []
    detections_by_image = {}
    current_id = 1  # IDs start at 1 and count up across all images

    for image_filename, raw_detections in per_image_detections:
        # assign_plant_ids() will start IDs at current_id for this image
        resolved = assign_plant_ids(raw_detections, start_id=current_id)

        # Advance the counter by however many plants were found in this image
        current_id += len(resolved)

        # Store per-image view
        detections_by_image[image_filename] = resolved

        # Add image filename to each detection for the flat all_detections list
        for d in resolved:
            all_detections.append({
                "plant_id": d["plant_id"],
                "image": image_filename,
                "bbox": d["bbox"],
                "confidence": d["confidence"]
            })

    return {
        "total_plants": len(all_detections),
        "detections_by_image": detections_by_image,
        "all_detections": all_detections
    }