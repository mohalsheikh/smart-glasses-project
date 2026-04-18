"""
Advanced Color Analysis and Text Understanding
Uses OpenAI GPT-4o for color identification, text translation, and visual comprehension
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import base64
from collections import Counter

from openai import OpenAI
import src.utils.config as config

client = OpenAI()


class ColorTextAnalyzer:
    """
    Advanced visual analysis combining:
    - Dominant color extraction
    - Color naming with GPT-4o
    - Text extraction and translation
    - Scene color description
    """
    
    # Extended color names
    COLOR_NAMES = {
        # Basic colors
        "red": [(255, 0, 0), (200, 0, 0), (180, 0, 0)],
        "green": [(0, 255, 0), (0, 200, 0), (0, 180, 0)],
        "blue": [(0, 0, 255), (0, 0, 200), (0, 0, 180)],
        "yellow": [(255, 255, 0), (255, 200, 0)],
        "orange": [(255, 165, 0), (255, 140, 0)],
        "purple": [(128, 0, 128), (148, 0, 211)],
        "pink": [(255, 192, 203), (255, 105, 180)],
        "brown": [(165, 42, 42), (139, 69, 19)],
        "black": [(0, 0, 0), (30, 30, 30)],
        "white": [(255, 255, 255), (240, 240, 240)],
        "gray": [(128, 128, 128), (169, 169, 169)],
        # Extended colors
        "cyan": [(0, 255, 255)],
        "magenta": [(255, 0, 255)],
        "navy": [(0, 0, 128)],
        "teal": [(0, 128, 128)],
        "olive": [(128, 128, 0)],
        "maroon": [(128, 0, 0)],
        "lime": [(0, 255, 0)],
        "aqua": [(0, 255, 255)],
        "silver": [(192, 192, 192)],
        "gold": [(255, 215, 0)],
        "beige": [(245, 245, 220)],
        "ivory": [(255, 255, 240)],
        "tan": [(210, 180, 140)],
        "khaki": [(240, 230, 140)],
        "coral": [(255, 127, 80)],
        "salmon": [(250, 128, 114)],
        "peach": [(255, 218, 185)],
        "lavender": [(230, 230, 250)],
        "indigo": [(75, 0, 130)],
        "turquoise": [(64, 224, 208)],
    }
    
    def __init__(self):
        print("🎨 ColorTextAnalyzer initialized")
    
    def extract_dominant_colors(
        self,
        frame: np.ndarray,
        n_colors: int = 5,
        sample_fraction: float = 0.3
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """Extract dominant colors from frame using k-means clustering"""
        try:
            # Resize for faster processing
            h, w = frame.shape[:2]
            if w > 300:
                scale = 300 / w
                frame = cv.resize(frame, None, fx=scale, fy=scale)
            
            # Sample pixels
            pixels = frame.reshape(-1, 3)
            n_samples = int(len(pixels) * sample_fraction)
            indices = np.random.choice(len(pixels), n_samples, replace=False)
            pixels = pixels[indices]
            
            # K-means clustering
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.2)
            _, labels, centers = cv.kmeans(
                pixels.astype(np.float32),
                n_colors,
                None,
                criteria,
                10,
                cv.KMEANS_PP_CENTERS
            )
            
            # Count occurrences
            label_counts = Counter(labels.flatten())
            total = len(labels)
            
            # Sort by frequency
            results = []
            for i in range(n_colors):
                color = tuple(map(int, centers[i]))
                percentage = label_counts[i] / total
                results.append((color, percentage))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except Exception as e:
            print(f"⚠️ Color extraction error: {e}")
            return []
    
    def get_basic_color_name(self, bgr_color: Tuple[int, int, int]) -> str:
        """Get basic color name from BGR using Euclidean distance"""
        try:
            min_dist = float('inf')
            closest_name = "unknown"
            
            for name, references in self.COLOR_NAMES.items():
                for ref_color in references:
                    # BGR to RGB
                    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                    dist = np.linalg.norm(np.array(rgb_color) - np.array(ref_color))
                    if dist < min_dist:
                        min_dist = dist
                        closest_name = name
            
            return closest_name
        except:
            return "unknown"
    
    def describe_colors_gpt4o(self, frame: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Use GPT-4o to describe colors in natural language"""
        try:
            # Crop to region if specified
            if region:
                x1, y1, x2, y2 = region
                frame = frame[y1:y2, x1:x2]
            
            if frame.size == 0:
                return "Unable to analyze colors."
            
            # Encode image
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = client.chat.completions.create(
                model=config.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You describe colors in images naturally and concisely for a blind person. Focus on dominant colors and notable color patterns. Keep it brief (1-2 sentences)."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What are the main colors in this image? Describe them naturally."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ GPT-4o color description error: {e}")
            return "Unable to describe colors."
    
    def describe_colors_simple(self, frame: np.ndarray) -> str:
        """Simple color description without AI"""
        colors = self.extract_dominant_colors(frame, n_colors=3)
        
        if not colors:
            return "Unable to identify colors."
        
        descriptions = []
        for color, percentage in colors:
            if percentage > 0.15:  # Only mention significant colors
                name = self.get_basic_color_name(color)
                descriptions.append(name)
        
        if not descriptions:
            return "Mixed colors."
        
        if len(descriptions) == 1:
            return f"Mostly {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"Mostly {descriptions[0]} and {descriptions[1]}."
        else:
            return f"Mostly {descriptions[0]}, {descriptions[1]}, and {descriptions[2]}."
    
    def analyze_color_of_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], use_ai: bool = True) -> str:
        """Analyze color of a specific object"""
        try:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return "Unable to analyze object color."
            
            if use_ai and config.OPENAI_API_KEY_PRESENT:
                return self.describe_colors_gpt4o(frame, (x1, y1, x2, y2))
            else:
                crop = frame[y1:y2, x1:x2]
                return self.describe_colors_simple(crop)
                
        except Exception as e:
            return "Unable to analyze object color."
    
    def translate_visible_text(
        self,
        text: str,
        target_language: str = "en",
        source_language: str = "auto"
    ) -> Dict[str, str]:
        """Translate text using GPT-4o"""
        try:
            if not text or not text.strip():
                return {"original": "", "translated": "", "language": "unknown"}
            
            response = client.chat.completions.create(
                model=config.OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a translator. Translate the given text to {target_language}. Also identify the source language. Format: SOURCE_LANG|TRANSLATION"
                    },
                    {
                        "role": "user",
                        "content": f"Translate this text: {text}"
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            parts = result.split('|', 1)
            
            if len(parts) == 2:
                return {
                    "original": text,
                    "translated": parts[1].strip(),
                    "language": parts[0].strip()
                }
            else:
                return {
                    "original": text,
                    "translated": result,
                    "language": "unknown"
                }
                
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return {
                "original": text,
                "translated": text,
                "language": "unknown"
            }
    
    def analyze_text_context(self, frame: np.ndarray, detected_text: str) -> str:
        """Use GPT-4o to understand context of detected text"""
        try:
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = client.chat.completions.create(
                model=config.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You help blind people understand text in context. Briefly explain what the text is (sign, label, document, etc.) and its likely purpose."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"This text was detected: '{detected_text}'. What kind of text is this and what's its purpose?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Text context analysis error: {e}")
            return "Text detected but unable to analyze context."
    
    def identify_brand_or_product(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Identify brand or product using GPT-4o Vision"""
        try:
            if bbox:
                x1, y1, x2, y2 = bbox
                frame = frame[y1:y2, x1:x2]
            
            if frame.size == 0:
                return "Unable to identify."
            
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = client.chat.completions.create(
                model=config.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You identify brands, products, and logos. Be specific but concise. If you're not confident, say so."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What brand or product is this? Be specific."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Brand identification error: {e}")
            return "Unable to identify brand or product."
    
    def describe_clothing_colors(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> str:
        """Describe clothing colors for a person"""
        try:
            x1, y1, x2, y2 = person_bbox
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return "Unable to see clothing."
            
            # Split into upper and lower body regions
            h = person_crop.shape[0]
            upper = person_crop[:int(h*0.5), :]
            lower = person_crop[int(h*0.5):, :]
            
            upper_colors = self.extract_dominant_colors(upper, n_colors=2)
            lower_colors = self.extract_dominant_colors(lower, n_colors=2)
            
            upper_desc = self.get_basic_color_name(upper_colors[0][0]) if upper_colors else "unknown"
            lower_desc = self.get_basic_color_name(lower_colors[0][0]) if lower_colors else "unknown"
            
            return f"{upper_desc} top, {lower_desc} bottom"
            
        except Exception as e:
            return "Unable to describe clothing."
