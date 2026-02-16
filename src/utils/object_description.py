# src/utils/object_description.py

from typing import List, Dict, Any, Optional
import src.utils.config as config

# Ignore noisy labels
IGNORE_LABELS = {
    "Clothing", "Human arm", "Human hair", "Human leg", "Human body",
    "Human head", "Human ear", "Human eye", "Human mouth", "Human nose",
    "Human hand", "Human foot", "Human face", "Fashion accessory"
}

# Merge similar labels
MERGE_LABELS = {
    "Human face": "person", "Man": "person", "Woman": "person",
    "Boy": "person", "Girl": "person", "Person": "person",
    "Laptop computer": "laptop", "Computer keyboard": "keyboard",
    "Computer mouse": "mouse", "Mobile phone": "phone",
    "Cellular telephone": "phone", "Telephone": "phone",
    "Television": "TV", "Drink": "beverage",
}

# Priority objects
PRIORITY_LABELS = {
    "person", "Door", "Door handle", "Stairs", "Chair", "Table",
    "Car", "Bus", "Truck", "Bicycle", "Motorcycle",
    "Traffic light", "Traffic sign", "Stop sign",
    "Laptop", "laptop", "phone", "Mug", "Bottle",
    "Toilet", "Sink", "Bed", "Couch"
}


def normalize_label(label: str) -> Optional[str]:
    """Normalize labels - fast version."""
    if label in IGNORE_LABELS:
        return None
    if label in MERGE_LABELS:
        return MERGE_LABELS[label]
    return label


def direction_from_center(center, frame_width: int) -> Optional[str]:
    """Get direction from center position."""
    if center is None or frame_width <= 0:
        return None

    x = center[0]
    left_thresh = frame_width / 3
    right_thresh = 2 * frame_width / 3

    if x < left_thresh:
        return "on your left"
    elif x > right_thresh:
        return "on your right"
    else:
        return "in front of you"


def add_indefinite_article(label: str) -> str:
    """Add a/an to label."""
    if not label:
        return label
    first_letter = label[0].lower()
    return f"an {label}" if first_letter in "aeiou" else f"a {label}"


def get_confidence_threshold(label: str) -> float:
    """Get confidence threshold based on object type."""
    if label in config.SMALL_OBJECTS:
        return config.CONFIDENCE_BY_CATEGORY["small_objects"]
    elif label in PRIORITY_LABELS:
        return config.CONFIDENCE_BY_CATEGORY["priority_objects"]
    else:
        return config.CONFIDENCE_BY_CATEGORY["general_objects"]


def summarize_detections(
    detections: List[Dict[str, Any]],
    frame_width: int,
    max_items: int | None = None,
) -> str:
    """
    Fast summary - no distance estimation.
    """
    if max_items is None:
        max_items = config.MAX_SPEECH_ITEMS

    # Filter by adaptive confidence
    filtered = []
    for d in detections:
        raw_label = d.get("label", "object")
        cleaned = normalize_label(raw_label)
        if cleaned is None:
            continue

        conf = float(d.get("confidence", 0.0))
        required_conf = get_confidence_threshold(cleaned)

        if conf >= required_conf:
            filtered.append({
                "label": cleaned,
                "confidence": conf,
                "center": d.get("center"),
            })

    if not filtered:
        return "I don't see any objects clearly."

    # Sort by confidence
    filtered.sort(key=lambda x: x["confidence"], reverse=True)

    # Priority first
    priority = [d for d in filtered if d["label"] in PRIORITY_LABELS]
    non_priority = [d for d in filtered if d["label"] not in PRIORITY_LABELS]

    combined = priority + non_priority
    combined = combined[:max_items]

    # Build phrases
    phrases: list[str] = []
    for d in combined:
        label = d["label"]
        center = d["center"]
        direction = direction_from_center(center, frame_width)

        obj_phrase = add_indefinite_article(label)

        if direction:
            phrases.append(f"{obj_phrase} {direction}")
        else:
            phrases.append(obj_phrase)

    if not phrases:
        return "I don't see any objects clearly."

    # Natural sentence
    if len(phrases) == 1:
        return f"I see {phrases[0]}."
    elif len(phrases) == 2:
        return f"I see {phrases[0]} and {phrases[1]}."
    else:
        return f"I see {', '.join(phrases[:-1])}, and {phrases[-1]}."

def format_ocr_feedback(
        ocr_result: Dict[str, Any],
        low_threshold: float = 0.60,
        high_threshold: float = 0.85
    ) -> str:
        """
        Formats OCR confidence feedback with warnings or affirmations.
        
        Args:
            ocr_result: Result dict from OCREngine.extract_text_with_confidence()
            low_threshold: Below this triggers a warning
            high_threshold: Above this triggers affirmation
        
        Returns:
            Formatted feedback string
        """
        text = ocr_result.get("text", "")
        avg_conf = ocr_result.get("avg_conf", 0.0)
        count = ocr_result.get("count", 0)
        
        if not text or count == 0:
            return "❌🔍 No text detected.\n"
        
        # Format confidence level
        if avg_conf < low_threshold:
            conf_msg = f"⚠️🟠 Confidence: low (avg {avg_conf:.2f}) — Please adjust position.\n"
        elif avg_conf >= high_threshold:
            conf_msg = f"✅🟢 Confidence: high (avg {avg_conf:.2f})\n"
        else:
            conf_msg = f"⚠️🟡 Confidence: mid (avg {avg_conf:.2f})\n"
        
        return f'\n📝 Text: "{text}"\n{conf_msg}'

def _bbox_area(bbox) -> float:
    """
    bbox is (x1, y1, x2, y2)
    """
    try:
        x1, y1, x2, y2 = bbox
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        return w * h
    except Exception:
        return 0.0


def _estimate_distance_word(bbox, frame_width: int, frame_height: int) -> str:
    """
    Rough distance estimate based on bounding-box area as a % of frame area.
    (Not perfect, but good enough for Sprint 2)
    """
    if frame_width <= 0 or frame_height <= 0:
        return "unknown distance"

    area = _bbox_area(bbox)
    frame_area = float(frame_width) * float(frame_height)
    ratio = area / frame_area if frame_area > 0 else 0.0

    if ratio > 0.20:
        return "very close"
    if ratio > 0.08:
        return "near"
    return "far"


def filter_detections_by_label(detections, target_label: str):
    """
    Filter detections down to only the requested target label.
    We match against normalized labels in normalize_label().
    """
    if not target_label:
        return []

    target = target_label.strip().lower()
    matches = []

    for d in detections:
        raw_label = d.get("label", "")
        cleaned = normalize_label(raw_label)
        if cleaned is None:
            continue

        if str(cleaned).strip().lower() == target:
            matches.append(d)

    return matches


def summarize_target_location(
    detections,
    frame_width: int,
    frame_height: int,
    target_label: str,
) -> str:
    """
    Sprint 2 target-only output:
    - B: direction + rough near/far
    - C: mention count if multiple
    """
    if not target_label:
        return "I didn't understand which object you're looking for. Try saying: where is my bottle."

    matches = filter_detections_by_label(detections, target_label)
    if not matches:
        return f"I don't see a {target_label}. Want me to describe what I do see?"

    # Choose the “best” match by largest area (usually closest / most important)
    best = max(matches, key=lambda d: _bbox_area(d.get("bbox")))
    direction = direction_from_center(best.get("center"), frame_width) or "in front of you"
    dist_word = _estimate_distance_word(best.get("bbox"), frame_width, frame_height)

    if len(matches) == 1:
        return f"Your {target_label} is {direction}. It looks {dist_word}."
    else:
        return f"I see {len(matches)} {target_label}s. The closest one is {direction} and looks {dist_word}."
