
# src/utils/object_description.py

from enum import Enum
from typing import Counter, List, Dict, Any, Optional
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
    "general_objects": 0.20, # 0.25,
}

# Ignore noisy labels
IGNORE_LABELS = {
    "Clothing", "Human arm", "Human hair", "Human leg", "Human body",
    "Human head", "Human ear", "Human eye", "Human mouth", "Human nose",
    "Human hand", "Human foot", "Human face", "Fashion accessory"
}

# Merge similar labels
# Currency labels are normalized so that both front and back variants
# produce a clean spoken form like "five dollar bill".
MERGE_LABELS = {
    # --- US currency normalization ---
    "one-front":     "one dollar bill",
    "one-back":      "one dollar bill",
    "five-front":    "five dollar bill",
    "five-back":     "five dollar bill",
    "ten-front":     "ten dollar bill",
    "ten-back":      "ten dollar bill",
    "twenty-front":  "twenty dollar bill",
    "twenty-back":   "twenty dollar bill",
    "fifty-front":   "fifty dollar bill",
    "fifty-back":    "fifty dollar bill",
    "hundred-front": "hundred dollar bill",
    "hundred-back":  "hundred dollar bill",
    # "Human face": "person", "Man": "person", "Woman": "person",
    # "Boy": "person", "Girl": "person", "Person": "person",
    # "Laptop computer": "laptop", "Computer keyboard": "keyboard",
    # "Computer mouse": "mouse", "Mobile phone": "phone",
    # "Cellular telephone": "phone", "Telephone": "phone",
    # "Television": "TV", "Drink": "beverage",
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
    detections_lists: List[List[Dict[str, Any]]],
    frame_width: int,
    max_items: int = MAX_SPEECH_ITEMS,
) -> str:
    """
    Summarizes detections into a natural language description.

    Args:
        detections_lists: List of lists of detection dicts from object detector.
        frame_width: Width of the camera frame for direction estimation
        max_items: Maximum number of objects to include in the summary
    
    Returns:
        Natural language description of detected objects and their directions.
    """
    formatted_detections = _format_detections(detections_lists, frame_width, max_items)

    if not formatted_detections:
        return "I don't see any objects clearly."
    else:
        return _construct_description(formatted_detections)


def _format_detections(
    detections_lists: List[List[Dict[str, Any]]],
    frame_width: int,
    max_items: int = MAX_SPEECH_ITEMS,
) -> List[Dict[str, Optional[str]]] | None:
    """
    Formats detections into a list of dicts with label, direction, and ocr_text if it is present, which is to be used for summarization.
    """

    # used to hold filtered detections from each frame, which will be condensed into a single list describing detections across all frames 
    filtered_detections_list = []

    for detections_list in detections_lists:
        # Filter by adaptive confidence
        filtered_detections_in_current_frame = []

        for d in detections_list:
            raw_label = d.get("label")
            normalized_label = normalize_label(raw_label)

            # continue to next detection if this one is empty or should be ignored.
            if normalized_label is None: 
                continue

            conf = float(d.get("confidence", 0.0))
            required_conf = get_confidence_threshold(normalized_label)

            if conf >= required_conf:
                filtered_detections_in_current_frame.append({
                    "label": normalized_label,
                    "confidence": conf,
                    "bbox": d.get("bbox"),
                    "center": d.get("center"),
                    "track_id": d.get("track_id"),
                    "ocr_text": d.get("ocr_text"),
                    "ocr_avg_confidence": d.get("ocr_avg_confidence", 0.0)
                })

        filtered_detections_list.append(filtered_detections_in_current_frame)

    if all(len(lst) == 0 for lst in filtered_detections_list):
        return None

    # condense detections across frames into a single list describing unique detected objects.
    # print(f"[ObjDesc] Filtered detections for {len(filtered_detections_list)} frames: {filtered_detections_list} \n")
    condensed_detections = _condense_detections(filtered_detections_list)

    # Sort by confidence; priority first, and highest to lowest
    condensed_detections.sort(key=lambda x: 
                  (x["confidence"] + 1 if x["label"] in PRIORITY_LABELS \
                  else x["confidence"]), 
                  reverse=True)

    condensed_detections = condensed_detections[:max_items] # limit to max items

    # build list of labels and directions for speech output
    condensed_detections = [
        {
            "label": d["label"],
            "direction": direction_from_center(d["center"], frame_width),
            "ocr_text": d.get("ocr_text")
        }
        for d in condensed_detections
    ]

    return condensed_detections

def _condense_detections(filtered_detections_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Condenses detections across frames into a single list describing unique detected objects.
    """
    # at each track_id key, the value is a dict with these keys:
    # "label": normalized label with best confidence across frames for the object. (e.g. "person", "five dollar bill", etc.)
    # "confidence": confidence of that label
    # "center": the center coordinates of the object's bounding box in the most RECENT frame where it was detected, used for direction estimation.
    # "ocr_text": if the object has associated OCR text, the best OCR text across frames, determined by highest OCR average confidence.
    # "ocr_avg_confidence": the best average OCR confidence across frames for text read within the object's bounding box(es).
    best_candidates_for_track_id: Dict[int, Dict[str, Any]] = {}

    # you can instead do "for filtered_detections in filtered_detections_list:" if you want to remove the first print
    for i, filtered_detections in enumerate(filtered_detections_list): 
        merged_detections =  _merge_detections(filtered_detections) # merge detections with the same track id within the current frame, also deletes detections for N/A track ids.
        # print(f"[ObjDesc] Merged detections for frame {i}: {merged_detections} \n")

        for d in merged_detections:
            curr_id = d.get("track_id")

            if curr_id not in best_candidates_for_track_id:
                best_candidates_for_track_id[curr_id] = {
                    "label": d.get("label"),
                    "confidence": d.get("confidence", 0.0),
                    "ocr_text": d.get("ocr_text"),
                    "ocr_avg_confidence": d.get("ocr_avg_confidence", 0.0),
                    "center": d.get("center")
                }
            else:
                if d.get("confidence", 0.0) > best_candidates_for_track_id[curr_id].get("confidence", 0.0):
                    best_candidates_for_track_id[curr_id]["label"] = d.get("label")
                    best_candidates_for_track_id[curr_id]["confidence"] = d.get("confidence", 0.0)

                if d.get("ocr_avg_confidence", 0.0) > best_candidates_for_track_id[curr_id].get("ocr_avg_confidence", 0.0):
                    best_candidates_for_track_id[curr_id]["ocr_text"] = d.get("ocr_text")
                    best_candidates_for_track_id[curr_id]["ocr_avg_confidence"] = d.get("ocr_avg_confidence", 0.0)

                # always update center to most recent detection
                best_candidates_for_track_id[curr_id]["center"] = d.get("center")

    condensed_detections = list(best_candidates_for_track_id.values())
    # print(f"[ObjDesc] Condensed detections across frames: {condensed_detections}")
    return condensed_detections

def _merge_detections(filtered_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    given list of detections for one frame, removes detections for N/A track ids
    and merges detections the rest of the track ids to a single detection per track id.
    for track ids that have multiple detections:
    - if there are multiple labels for the same track id, keep the label that is associated with the highest confidence for that object.
    - keep the highest yolo and ocr confidence across detections for the same track id.
    - if there are multiple ocr_text values for the same track id, concatenate them with "and".
    - keep the bbox with the largest area across detections for the same track id.
    - keep that bbox's center for direction estimation.
    """
    track_id_counter = Counter() # tracks occurrences of a track_id across detections in the current frame

    # first pass to count track_id occurrences and filter out N/A track ids
    for d in filtered_detections:
        track_id = d.get("track_id")
        if track_id != 'N/A':
            track_id_counter[track_id] += 1
            
    track_ids_to_normalize = {track_id for track_id, count in track_id_counter.items() if count > 1}

    new_dets_for_track_ids_to_normalize: Dict[int, Dict[str, Any]] = {track_id: {
        "label": None,
        "confidence": 0.0,
        "bbox": None,
        "bbox_area": 0.0, # just for calculation purposes, not included in final output
        "center": None,
        "track_id": track_id,
        "ocr_text": None,
        "ocr_avg_confidence": 0.0
    } for track_id in track_ids_to_normalize}

    for d in filtered_detections:
        track_id = d.get("track_id")

        if track_id in track_ids_to_normalize:
            curr_label = d.get("label", None)
            curr_conf = d.get("confidence", 0.0)
            curr_bbox = d.get("bbox", None)
            curr_bbox_area = _bbox_area(curr_bbox) if curr_bbox else 0.0
            curr_center = d.get("center", None)
            curr_ocr_text = d.get("ocr_text", None)
            curr_ocr_avg_conf = d.get("ocr_avg_confidence", 0.0)

            # print(f"[ObjDesc] Merging track_id {track_id}: current label/conf {curr_label}/{curr_conf}, bbox area {curr_bbox_area}, center {curr_center}, ocr text/conf {curr_ocr_text}/{curr_ocr_avg_conf}\n")

            if curr_conf > new_dets_for_track_ids_to_normalize[track_id]["confidence"]:
                new_dets_for_track_ids_to_normalize[track_id]["label"] = curr_label
                new_dets_for_track_ids_to_normalize[track_id]["confidence"] = curr_conf
            
            if curr_bbox_area > new_dets_for_track_ids_to_normalize[track_id]["bbox_area"]:
                new_dets_for_track_ids_to_normalize[track_id]["bbox"] = curr_bbox
                new_dets_for_track_ids_to_normalize[track_id]["bbox_area"] = curr_bbox_area
                new_dets_for_track_ids_to_normalize[track_id]["center"] = curr_center

            if curr_ocr_text is not None:
                if new_dets_for_track_ids_to_normalize[track_id]["ocr_text"] is not None:
                    new_dets_for_track_ids_to_normalize[track_id]["ocr_text"] = f"{curr_ocr_text} and {new_dets_for_track_ids_to_normalize[track_id]['ocr_text']}"
                else:
                    new_dets_for_track_ids_to_normalize[track_id]["ocr_text"] = curr_ocr_text 

                if curr_ocr_avg_conf > new_dets_for_track_ids_to_normalize[track_id]["ocr_avg_confidence"]:
                    new_dets_for_track_ids_to_normalize[track_id]["ocr_avg_confidence"] = curr_ocr_avg_conf

            # print(f"[ObjDesc] Updated merged detection for track_id {track_id}: {new_dets_for_track_ids_to_normalize[track_id]}\n")

    for k, _ in new_dets_for_track_ids_to_normalize.items():
        del new_dets_for_track_ids_to_normalize[k]["bbox_area"] # remove from final output because it was just for calculation purposes

    # for track ids that were merged, remove all detections with those track ids from the original list and replace with the new merged detection for that track id.
    merged_detections = [d for d in filtered_detections if d.get("track_id") not in track_ids_to_normalize and d.get("track_id") != 'N/A']
    merged_detections.extend(new_dets_for_track_ids_to_normalize.values())
    
    return merged_detections

def _bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box."""
    if bbox is None or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

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