
# src/utils/object_description.py

from enum import Enum
from typing import List, Dict, Any, Optional
import src.utils.config as config
import inflect

_inflect = inflect.engine()
MAX_SPEECH_ITEMS: int = 20

# Small objects that need lower confidence
SMALL_OBJECTS: set = {
    "Pen", "Pencil", "Toothbrush", "Spoon", "Fork", "Knife", 
    "Remote control", "Computer mouse", "Glasses", "Watch"
}

CONFIDENCE_BY_CATEGORY: dict = {
    "small_objects": 0.15,
    "priority_objects": 0.20,
    "general_objects": 0.25,
}

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

class Direction(Enum):
    LEFT = 1
    FRONT = 2
    RIGHT = 3

def normalize_label(label: str) -> Optional[str]:
    """
    Label normalization.

    Args: 
        label: from object detector output

    Returns: 
        normalized label based on contents of MERGE_LABELS and IGNORE_LABELS. 
        If label is in IGNORE_LABELS, returns None to indicate it should be skipped.
    """
    if label in IGNORE_LABELS:
        return None
    if label in MERGE_LABELS:
        return MERGE_LABELS[label]
    return label


def direction_from_center(center, frame_width: int) -> Direction:
    """Get direction from center position."""
    if center is None or frame_width <= 0:
        return None

    x = center[0]
    left_thresh = frame_width / 3
    right_thresh = 2 * frame_width / 3

    if x < left_thresh:
        return Direction.LEFT
    elif x > right_thresh:
        return Direction.RIGHT
    else:
        return Direction.FRONT


def add_indefinite_article(label: str) -> str:
    """Add a/an to label."""
    if not label:
        return label
    return _inflect.a(label)


def get_confidence_threshold(label: str) -> float:
    """Get confidence threshold based on object type."""
    if label in SMALL_OBJECTS:
        return CONFIDENCE_BY_CATEGORY["small_objects"]
    elif label in PRIORITY_LABELS:
        return CONFIDENCE_BY_CATEGORY["priority_objects"]
    else:
        return CONFIDENCE_BY_CATEGORY["general_objects"]

def pluralize(label: str, count: int) -> str:
    """Pluralize label correctly based on count."""
    if count == 1:
        return label

    plural = _inflect.plural_noun(label)
    return plural if plural else label

# def _count_to_word(count: int) -> str:
#     """Convert small integer counts to English words."""
#     words = {
#         2: "two", 3: "three", 4: "four", 5: "five",
#         6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
#     }
#     return words.get(count, str(count))

def summarize_detections(
    detections: List[Dict[str, Any]],
    frame_width: int,
    max_items: int = MAX_SPEECH_ITEMS,
) -> str:
    """
    Summarizes detections into a natural language description.

    Args:
        detections: List of detection dicts from object detector.
        frame_width: Width of the camera frame for direction estimation
        max_items: Maximum number of objects to include in the summary
    
    Returns:
        Natural language description of detected objects and their directions.
    """
    filtered_detections = _format_detections(detections, frame_width, max_items)

    if not filtered_detections:
        return "I don't see any objects clearly."
    else:
        return _construct_description(filtered_detections)
    

def _format_detections(
    detections: List[Dict[str, Any]],
    frame_width: int,
    max_items: int = MAX_SPEECH_ITEMS,
) -> List[Dict[str, Optional[str]]] | None:
    """
    Formats detections into a list of dicts with label, direction, and ocr_text if it is present, which is to be used for summarization.
    """
    # Filter by adaptive confidence
    filtered = []

    for d in detections:
        raw_label = d.get("label")
        normalized_label = normalize_label(raw_label)

        # continue to next detection if this one is empty or should be ignored.
        if normalized_label is None: 
            continue

        conf = float(d.get("confidence", 0.0))
        required_conf = get_confidence_threshold(normalized_label)

        if conf >= required_conf:
            filtered.append({
                "label": normalized_label,
                "confidence": conf,
                "center": d.get("center"),
                "ocr_text": d.get("ocr_text"),
            })

    if len(filtered) == 0:
        return None

    # Sort by confidence; priority first, and highest to lowest
    filtered.sort(key=lambda x: 
                  (x["confidence"] + 1 if x["label"] in PRIORITY_LABELS \
                  else x["confidence"]), 
                  reverse=True)

    filtered = filtered[:max_items] # limit to max items

    # build list of labels and directions for speech output
    filtered = [
        {
            "label": d["label"],
            "direction": direction_from_center(d["center"], frame_width),
            "ocr_text": d.get("ocr_text")
        }
        for d in filtered
    ]

    return filtered

def _construct_description(filtered_detections: List[Dict[str, Optional[str]]]) -> str:
    """Constructs a natural language description from the filtered detections.
    
    Groups detections by (label, direction, ocr_text) to eliminate redundancy.
    Example: two identical labels+directions become "two people in front of you".
    Another example: two identical labels+ocr_text+directions become "two stop signs that say STOP in front of you".
    """
    # Group by (label, direction, ocr_text) to count duplicates
    groups = {}
    for d in filtered_detections:
        label = d["label"]
        direction = d["direction"]
        ocr_text = d.get("ocr_text")
        key = (label, direction, ocr_text)
        groups[key] = groups.get(key, 0) + 1
    
    phrases = []
    for (label, direction, ocr_text), count in groups.items():
        # Build phrase based on count
        if count == 1:
            label_phrase = add_indefinite_article(label)
            verb_phrase = "that says" # singular verb phrase to match singular label_phrase, to create sentences like "a stop sign that says STOP in front of you"
        else:
            # Convert to plural: "two people", "three bottles", etc.
            plural_label = pluralize(label, count)
            count_word = _inflect.number_to_words(count)
            label_phrase = f"{count_word} {plural_label}"
            verb_phrase = "that say" # plural verb phrase to match plural label_phrase, to create sentences like "two stop signs that say STOP in front of you"

        # Add OCR text to phrase if it exists, naturally handling singular vs plural w/ "that says" vs "that say" from verb_phrase.
        if ocr_text is not None:
            label_phrase = f'{label_phrase} {verb_phrase} {ocr_text}'
        
        # Add direction
        match direction:
            case Direction.LEFT:
                phrase = f"{label_phrase} to your left"
            case Direction.RIGHT:
                phrase = f"{label_phrase} to your right"
            case Direction.FRONT:
                phrase = f"{label_phrase} in front of you"
            case None:
                phrase = label_phrase
        
        phrases.append(phrase)
    
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