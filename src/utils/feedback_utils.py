"""
Made by Eric Leon

Utility functions for formatting user feedback messages.
"""
from typing import Dict, Any


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
