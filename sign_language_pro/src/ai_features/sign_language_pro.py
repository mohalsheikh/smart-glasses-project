"""
Sign Language Interpreter Pro — ML-Powered ASL Recognition
============================================================

Drop-in replacement for sign_language_interpreter.py.
Uses trained PyTorch models instead of rule-based pattern matching.

Recognition pipeline:
  1. MediaPipe extracts hand landmarks (21 per hand, 3D)
  2. StaticSignNet classifies alphabet/numbers from single frames
  3. DynamicSignNet classifies word-level signs from temporal sequences
  4. GPT-4o Vision handles unknown/ambiguous signs as fallback
  5. SentenceBuilder assembles signs into natural language

Models needed (in models/ directory):
  - alphabet_model.pt  → trained by train_alphabet.py
  - word_model.pt      → trained by train_words.py

Falls back to rule-based recognition if models aren't found.

Author: VisionAssist AI Team
Version: 2.0 (ML-Powered)
"""

from __future__ import annotations

import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
import time
import math
import threading
import base64
import json
from typing import List, Dict, Any, Optional, Tuple, Deque, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from pathlib import Path

# Re-export existing types for backward compatibility
from src.ai_features.sign_language_interpreter import (
    SignCategory,
    SignConfidence,
    InterpreterMode,
    HandLandmarks,
    RecognizedSign,
    SignSequence,
    InterpreterState,
    HandAnalysisEngine,
    SignRecognitionEngine,
    ASLSignsDatabase,
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def _find_model_dir() -> Path:
    """Find the models directory."""
    candidates = [
        Path("models"),
        Path("src/models"),
        Path(__file__).parent / "models",
        Path(__file__).parent.parent / "models",
        Path(__file__).parent.parent.parent / "models",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Default: create in project root
    default = Path("models")
    default.mkdir(exist_ok=True)
    return default


class MLSignClassifier:
    """
    Wraps trained PyTorch models for sign classification.
    Loads alphabet_model.pt and word_model.pt if available.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "auto"):
        self.model_dir = Path(model_dir) if model_dir else _find_model_dir()
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )

        self.alphabet_model = None
        self.word_model = None
        self.alphabet_labels: Dict[int, str] = {}
        self.word_vocab: List[str] = []
        self.has_alphabet = False
        self.has_words = False

        self._load_models()

    def _load_models(self):
        """Load trained models if available."""
        # Try to import model architectures
        try:
            # First try from training directory
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training" / "sign_language"))
            from models import StaticSignNet, DynamicSignNet, normalize_landmarks
            self._normalize_fn = normalize_landmarks
        except ImportError:
            try:
                from src.ai_features.sign_language_models import StaticSignNet, DynamicSignNet, normalize_landmarks
                self._normalize_fn = normalize_landmarks
            except ImportError:
                print("⚠️  Model architectures not found. Using rule-based fallback.")
                return

        # Load alphabet model
        alphabet_path = self.model_dir / "alphabet_model.pt"
        if alphabet_path.exists():
            try:
                checkpoint = torch.load(str(alphabet_path), map_location=self.device, weights_only=True)
                num_classes = checkpoint.get("num_classes", 36)
                self.alphabet_labels = checkpoint.get("label_map", {})
                self.alphabet_model = StaticSignNet(num_classes=num_classes).to(self.device)
                self.alphabet_model.load_state_dict(checkpoint["model_state_dict"])
                self.alphabet_model.eval()
                self.has_alphabet = True
                acc = checkpoint.get("val_acc", 0)
                print(f"✅ Alphabet model loaded ({num_classes} classes, {acc:.0%} accuracy)")
            except Exception as e:
                print(f"⚠️  Failed to load alphabet model: {e}")

        # Load word model
        word_path = self.model_dir / "word_model.pt"
        if word_path.exists():
            try:
                checkpoint = torch.load(str(word_path), map_location=self.device, weights_only=True)
                num_classes = checkpoint.get("num_classes", 100)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                max_seq = checkpoint.get("max_seq_len", 64)
                self.word_vocab = checkpoint.get("vocab", [])
                self.word_model = DynamicSignNet(
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    max_seq_len=max_seq,
                ).to(self.device)
                self.word_model.load_state_dict(checkpoint["model_state_dict"])
                self.word_model.eval()
                self.has_words = True
                acc = checkpoint.get("val_acc", 0)
                print(f"✅ Word model loaded ({num_classes} words, {acc:.0%} accuracy)")
            except Exception as e:
                print(f"⚠️  Failed to load word model: {e}")

        if not self.has_alphabet and not self.has_words:
            print("⚠️  No trained models found in", self.model_dir)
            print("   Train with: python training/sign_language/train_alphabet.py")
            print("   Using rule-based fallback for now.")

    def classify_static(self, landmarks_21x3: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify a single-frame static sign (alphabet/number).

        Args:
            landmarks_21x3: (21, 3) hand landmark array

        Returns:
            (sign, confidence, alternatives)
        """
        if not self.has_alphabet:
            return "", 0.0, []

        # Normalize
        normalized = self._normalize_fn(landmarks_21x3)
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.alphabet_model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        # Top predictions
        topk = torch.topk(probs, min(5, len(probs)))
        best_idx = topk.indices[0].item()
        best_conf = topk.values[0].item()
        best_sign = self.alphabet_labels.get(best_idx, str(best_idx))

        alternatives = [
            (self.alphabet_labels.get(idx.item(), str(idx.item())), val.item())
            for idx, val in zip(topk.indices[1:4], topk.values[1:4])
        ]

        return best_sign, best_conf, alternatives

    def classify_sequence(
        self,
        sequence: np.ndarray,
        length: int,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify a temporal sign sequence (word-level).

        Args:
            sequence: (T, 126) landmark sequence (both hands)
            length: actual sequence length

        Returns:
            (word, confidence, alternatives)
        """
        if not self.has_words:
            return "", 0.0, []

        max_seq = self.word_model.max_seq_len

        # Pad or truncate
        T = sequence.shape[0]
        if T > max_seq:
            indices = np.linspace(0, T - 1, max_seq, dtype=int)
            sequence = sequence[indices]
            length = max_seq
        elif T < max_seq:
            pad = np.zeros((max_seq - T, sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)

        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([length]).to(self.device)

        with torch.no_grad():
            logits = self.word_model(tensor, length_tensor)
            probs = F.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, min(5, len(probs)))
        best_idx = topk.indices[0].item()
        best_conf = topk.values[0].item()
        best_word = self.word_vocab[best_idx] if best_idx < len(self.word_vocab) else str(best_idx)

        alternatives = [
            (self.word_vocab[idx.item()] if idx.item() < len(self.word_vocab) else str(idx.item()), val.item())
            for idx, val in zip(topk.indices[1:4], topk.values[1:4])
        ]

        return best_word, best_conf, alternatives


# =============================================================================
# GPT-4o VISION FALLBACK
# =============================================================================

class GPTVisionFallback:
    """
    Uses GPT-4o Vision to classify signs that the local model can't handle.
    Only triggered when local confidence is low.
    """

    def __init__(self, min_interval: float = 3.0):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = self.api_key is not None
        self.min_interval = min_interval
        self.last_call_time = 0.0
        self._client = None

        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                print("🌐 GPT-4o Vision fallback enabled")
            except ImportError:
                self.enabled = False

    def classify_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Send a frame to GPT-4o for sign classification.

        Returns:
            (recognized_sign, confidence_estimate)
        """
        if not self.enabled or not self._client:
            return "", 0.0

        now = time.time()
        if now - self.last_call_time < self.min_interval:
            return "", 0.0

        self.last_call_time = now

        try:
            # Encode frame
            _, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            b64_image = base64.b64encode(buffer).decode("utf-8")

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert ASL interpreter. Identify the ASL sign being shown. "
                            "Respond with ONLY a JSON object: "
                            '{\"sign\": \"<sign>\", \"confidence\": <0.0-1.0>, \"type\": \"letter|word|phrase\"}'
                            " If no clear sign, use {\"sign\": \"\", \"confidence\": 0.0, \"type\": \"none\"}"
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",
                                }
                            },
                            {
                                "type": "text",
                                "text": "What ASL sign is being shown?"
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.1,
            )

            text = response.choices[0].message.content.strip()
            # Parse JSON from response
            if text.startswith("{"):
                data = json.loads(text)
            else:
                # Try to extract JSON from text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                else:
                    return "", 0.0

            sign = data.get("sign", "")
            confidence = float(data.get("confidence", 0.0))
            return sign, confidence

        except Exception as e:
            return "", 0.0


# =============================================================================
# SENTENCE BUILDER
# =============================================================================

class SentenceBuilder:
    """
    Assembles recognized signs into natural sentences.
    Uses GPT to convert ASL grammar to English when enough signs accumulate.
    """

    def __init__(self, min_signs: int = 3, timeout: float = 3.0):
        self.sign_buffer: List[Tuple[str, float]] = []  # (sign, timestamp)
        self.min_signs = min_signs
        self.timeout = timeout
        self.last_sentence = ""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass

    def add_sign(self, sign: str, timestamp: float):
        """Add a recognized sign to the buffer."""
        # Don't add duplicates in rapid succession
        if self.sign_buffer and self.sign_buffer[-1][0] == sign:
            time_diff = timestamp - self.sign_buffer[-1][1]
            if time_diff < 0.5:
                return

        self.sign_buffer.append((sign, timestamp))

    def check_sentence(self, current_time: float) -> Optional[str]:
        """Check if we have enough signs to form a sentence."""
        if len(self.sign_buffer) < self.min_signs:
            return None

        # Check for pause
        if self.sign_buffer:
            time_since_last = current_time - self.sign_buffer[-1][1]
            if time_since_last < self.timeout:
                return None

        # Build sentence
        signs = [s for s, _ in self.sign_buffer]
        sentence = self._assemble(signs)
        self.sign_buffer.clear()
        self.last_sentence = sentence
        return sentence

    def _assemble(self, signs: List[str]) -> str:
        """Assemble signs into an English sentence."""

        # Try GPT for ASL→English grammar conversion
        if self._client and len(signs) >= 2:
            try:
                sign_str = " ".join(signs)
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Convert ASL gloss (sign sequence) into natural English. "
                                "ASL has different grammar than English. "
                                "Respond with ONLY the English sentence, nothing else."
                            )
                        },
                        {"role": "user", "content": f"ASL gloss: {sign_str}"}
                    ],
                    max_tokens=100,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass

        # Simple fallback: just join with spaces
        return " ".join(s.replace("_", " ") for s in signs)

    def clear(self):
        self.sign_buffer.clear()


# =============================================================================
# SIGN LANGUAGE INTERPRETER PRO (Main Class)
# =============================================================================

import sys  # needed for model import path manipulation

class SignLanguageInterpreterPro:
    """
    Production sign language interpreter with ML-powered recognition.

    Drop-in replacement for SignLanguageInterpreter with enhanced accuracy.
    Falls back gracefully when trained models aren't available.
    """

    def __init__(
        self,
        mode: InterpreterMode = InterpreterMode.CONTINUOUS,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        speech_callback: Optional[Callable[[str], None]] = None,
        model_dir: Optional[str] = None,
        enable_gpt_fallback: bool = True,
        enable_sentence_builder: bool = True,
        speak_letters: bool = True,
        speak_words: bool = True,
        word_pause_threshold: float = 1.0,
        confirmation_threshold: float = 0.75,
        uncertain_threshold: float = 0.55,
        enable_visual_feedback: bool = True,
    ):
        # Configuration
        self.mode = mode
        self.speech_callback = speech_callback
        self.speak_letters = speak_letters
        self.speak_words = speak_words
        self.word_pause_threshold = word_pause_threshold
        self.confirmation_threshold = confirmation_threshold
        self.uncertain_threshold = uncertain_threshold
        self.enable_visual_feedback = enable_visual_feedback

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # ML classifier
        self.ml_classifier = MLSignClassifier(model_dir=model_dir)

        # Rule-based fallback
        self.hand_analyzer = HandAnalysisEngine()
        self.rule_engine = SignRecognitionEngine()

        # GPT Vision fallback
        self.gpt_fallback = GPTVisionFallback() if enable_gpt_fallback else None

        # Sentence builder
        self.sentence_builder = SentenceBuilder() if enable_sentence_builder else None

        # State
        self.state = InterpreterState(mode=mode)
        self.last_frame_time = time.time()

        # Buffers
        self.letter_buffer: List[Tuple[str, float, float]] = []
        self.word_buffer: str = ""
        self.last_stable_sign: str = ""
        self.last_stable_time: float = 0.0
        self.sign_hold_start: float = 0.0
        self.sign_stability_frames: int = 0

        # Sequence buffer for dynamic signs
        self._sequence_buffer_left: Deque[np.ndarray] = deque(maxlen=90)   # ~3 sec at 30fps
        self._sequence_buffer_right: Deque[np.ndarray] = deque(maxlen=90)
        self._sequence_timestamps: Deque[float] = deque(maxlen=90)
        self._last_word_check: float = 0.0
        self._word_check_interval: float = 0.5  # Check for words every 0.5s

        # Speech management
        self.last_spoken_letter: str = ""
        self.last_spoken_time: float = 0.0
        self.min_letter_speak_interval: float = 1.5
        self._last_spoken_word: str = ""
        self._last_word_time: float = 0.0
        self._speech_lock = threading.Lock()

        # Stats
        self._stats = {
            "ml_predictions": 0,
            "rule_predictions": 0,
            "gpt_predictions": 0,
            "total_signs": 0,
            "sentences_built": 0,
        }

        # Visual colors
        self.colors = {
            'hand_landmarks': (0, 255, 128),
            'text_fg': (255, 255, 255),
            'high_confidence': (0, 255, 0),
            'medium_confidence': (0, 255, 255),
            'low_confidence': (0, 165, 255),
            'uncertain': (0, 0, 255),
            'panel_bg': (20, 20, 20),
            'ml_badge': (0, 200, 0),
            'rule_badge': (0, 165, 255),
            'gpt_badge': (255, 100, 0),
        }

        print("🤟 Sign Language Interpreter Pro v2.0 initialized")
        print(f"   ├─ Mode: {mode.value}")
        print(f"   ├─ ML Models: {'✅' if self.ml_classifier.has_alphabet or self.ml_classifier.has_words else '❌ (rule-based)'}")
        print(f"   ├─ GPT Fallback: {'✅' if self.gpt_fallback and self.gpt_fallback.enabled else '❌'}")
        print(f"   ├─ Sentence Builder: {'✅' if self.sentence_builder else '❌'}")
        print(f"   └─ Device: {self.ml_classifier.device}")

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None,
    ) -> Tuple[List[RecognizedSign], np.ndarray]:
        """
        Process a video frame for sign language recognition.

        Args:
            frame: BGR image frame
            detections: Optional object detections (unused, API compat)

        Returns:
            (recognized_signs, annotated_frame)
        """
        current_time = time.time()
        self.last_frame_time = current_time

        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        recognized_signs: List[RecognizedSign] = []

        # Convert to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        # Track hands
        left_hand: Optional[HandLandmarks] = None
        right_hand: Optional[HandLandmarks] = None
        left_raw: Optional[np.ndarray] = None
        right_raw: Optional[np.ndarray] = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                handedness = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score

                # Extract raw landmarks
                raw_lm = np.array(
                    [[l.x, l.y, l.z] for l in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                # Process through hand analyzer (for rule-based and features)
                hand_data = self.hand_analyzer.process_landmarks(
                    hand_landmarks, handedness, w, h
                )
                hand_data.confidence = confidence

                if handedness.lower() == "left":
                    left_hand = hand_data
                    left_raw = raw_lm
                else:
                    right_hand = hand_data
                    right_raw = raw_lm

                # Draw landmarks
                if self.enable_visual_feedback:
                    self.mp_draw.draw_landmarks(
                        annotated_frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

        # Buffer landmarks for sequence classification
        self._sequence_buffer_left.append(left_raw if left_raw is not None else np.zeros((21, 3), dtype=np.float32))
        self._sequence_buffer_right.append(right_raw if right_raw is not None else np.zeros((21, 3), dtype=np.float32))
        self._sequence_timestamps.append(current_time)

        # --- ALPHABET / NUMBER RECOGNITION ---
        if self.mode in [InterpreterMode.FINGERSPELLING, InterpreterMode.CONTINUOUS]:
            primary_hand = right_hand if right_hand else left_hand
            primary_raw = right_raw if right_raw is not None else left_raw

            if primary_hand and primary_raw is not None:
                sign, conf, alts, source = self._classify_static(primary_hand, primary_raw)

                if sign and conf > 0.55:
                    self._handle_sign_recognition(sign, conf, SignCategory.ALPHABET, current_time)

                    recognized = RecognizedSign(
                        sign=sign,
                        category=SignCategory.ALPHABET if sign.isalpha() else SignCategory.NUMBER,
                        confidence=conf,
                        confidence_level=self._get_confidence_level(conf),
                        timestamp=current_time,
                        hand_used=primary_hand.handedness,
                        alternatives=alts,
                    )
                    recognized_signs.append(recognized)

        # --- WORD RECOGNITION ---
        if self.mode in [InterpreterMode.WORD_SIGNS, InterpreterMode.CONTINUOUS]:
            # Check for word-level signs periodically
            if current_time - self._last_word_check > self._word_check_interval:
                self._last_word_check = current_time
                word, word_conf, word_alts = self._classify_dynamic()

                if word and word_conf > 0.6:
                    if word != self._last_spoken_word or (current_time - self._last_word_time) > 3.0:
                        recognized = RecognizedSign(
                            sign=word.replace("_", " "),
                            category=SignCategory.COMMON_WORD,
                            confidence=word_conf,
                            confidence_level=self._get_confidence_level(word_conf),
                            timestamp=current_time,
                            hand_used="both" if left_hand and right_hand else ("right" if right_hand else "left"),
                            alternatives=word_alts,
                        )
                        recognized_signs.append(recognized)
                        self._speak_sign(word.replace("_", " "), word_conf, is_word=True)
                        self._last_spoken_word = word
                        self._last_word_time = current_time

                        # Add to sentence builder
                        if self.sentence_builder:
                            self.sentence_builder.add_sign(word, current_time)

            # Also check rule-based common signs
            if left_hand or right_hand:
                motion = self.hand_analyzer.detect_motion('right' if right_hand else 'left')
                rule_sign, rule_conf = self.rule_engine.recognize_common_sign(
                    left_hand, right_hand, motion
                )
                if rule_sign and rule_conf > 0.85:
                    if rule_sign != self._last_spoken_word or (current_time - self._last_word_time) > 3.0:
                        recognized = RecognizedSign(
                            sign=rule_sign.replace("_", " "),
                            category=SignCategory.COMMON_WORD,
                            confidence=rule_conf,
                            confidence_level=self._get_confidence_level(rule_conf),
                            timestamp=current_time,
                        )
                        recognized_signs.append(recognized)
                        self._speak_sign(rule_sign.replace("_", " "), rule_conf, is_word=True)
                        self._last_spoken_word = rule_sign
                        self._last_word_time = current_time

        # --- GPT FALLBACK ---
        if (self.gpt_fallback and self.gpt_fallback.enabled
                and not recognized_signs
                and (left_hand or right_hand)):
            gpt_sign, gpt_conf = self.gpt_fallback.classify_frame(frame)
            if gpt_sign and gpt_conf > 0.5:
                recognized = RecognizedSign(
                    sign=gpt_sign,
                    category=SignCategory.COMMON_WORD,
                    confidence=gpt_conf,
                    confidence_level=self._get_confidence_level(gpt_conf),
                    timestamp=current_time,
                )
                recognized_signs.append(recognized)
                self._stats["gpt_predictions"] += 1

        # --- SENTENCE CHECK ---
        if self.sentence_builder:
            sentence = self.sentence_builder.check_sentence(current_time)
            if sentence:
                self._stats["sentences_built"] += 1
                if self.speech_callback:
                    self.speech_callback(f"Sentence: {sentence}")

        # --- CHECK WORD COMPLETION ---
        self._check_word_completion(current_time)

        # --- VISUAL FEEDBACK ---
        if self.enable_visual_feedback:
            self._draw_status_panel(annotated_frame, recognized_signs, left_hand, right_hand)

        return recognized_signs, annotated_frame

    def _classify_static(
        self, hand_data: HandLandmarks, raw_landmarks: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]], str]:
        """
        Classify a static sign using ML model with rule-based fallback.
        Returns: (sign, confidence, alternatives, source)
        """
        # Try ML model first
        if self.ml_classifier.has_alphabet:
            ml_sign, ml_conf, ml_alts = self.ml_classifier.classify_static(raw_landmarks)
            if ml_conf > 0.5:
                self._stats["ml_predictions"] += 1
                return ml_sign, ml_conf, ml_alts, "ml"

        # Rule-based fallback
        letter, letter_conf, alts = self.rule_engine.recognize_alphabet(hand_data)
        number, number_conf = self.rule_engine.recognize_number(hand_data)

        if letter_conf > number_conf:
            self._stats["rule_predictions"] += 1
            return letter, letter_conf, alts, "rule"
        elif number_conf > 0.5:
            self._stats["rule_predictions"] += 1
            return number, number_conf, [], "rule"

        return "", 0.0, [], "none"

    def _classify_dynamic(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Classify a dynamic sign from the sequence buffer."""
        if not self.ml_classifier.has_words:
            return "", 0.0, []

        if len(self._sequence_buffer_left) < 10:  # Need at least ~0.3s
            return "", 0.0, []

        # Build sequence
        left_seq = np.array(list(self._sequence_buffer_left))
        right_seq = np.array(list(self._sequence_buffer_right))

        T = left_seq.shape[0]
        left_flat = left_seq.reshape(T, -1)    # (T, 63)
        right_flat = right_seq.reshape(T, -1)  # (T, 63)
        combined = np.concatenate([left_flat, right_flat], axis=1)  # (T, 126)

        word, conf, alts = self.ml_classifier.classify_sequence(combined, T)
        return word, conf, alts

    def _handle_sign_recognition(
        self, sign: str, confidence: float, category: SignCategory, current_time: float
    ):
        """Handle sign for word building."""
        if not (len(sign) == 1 and (sign.isalpha() or sign.isdigit())):
            return

        if sign == self.last_stable_sign:
            self.sign_stability_frames += 1
            if self.sign_stability_frames >= 3:
                hold_duration = current_time - self.sign_hold_start
                if hold_duration > 0.2:
                    if len(self.letter_buffer) == 0 or self.letter_buffer[-1][0] != sign:
                        self.letter_buffer.append((sign, confidence, current_time))
                        self.word_buffer += sign
                        self._stats["total_signs"] += 1
                        if self.speak_letters:
                            self._speak_sign(sign, confidence, is_word=False)
                        if self.sentence_builder:
                            self.sentence_builder.add_sign(sign, current_time)
        else:
            self.last_stable_sign = sign
            self.last_stable_time = current_time
            self.sign_hold_start = current_time
            self.sign_stability_frames = 1

    def _check_word_completion(self, current_time: float):
        """Check if a pause indicates word completion."""
        if not self.word_buffer:
            return

        if current_time - self.last_stable_time > self.word_pause_threshold:
            word = self.word_buffer.lower()
            if self.speak_words and len(word) > 1:
                self._speak_word(word)
            self.word_buffer = ""
            self.letter_buffer.clear()

    def _speak_sign(self, sign: str, confidence: float, is_word: bool = False):
        """Speak a recognized sign."""
        with self._speech_lock:
            current_time = time.time()

            if not is_word:
                if current_time - self.last_spoken_time < self.min_letter_speak_interval:
                    return
                if sign == self.last_spoken_letter:
                    return

            if confidence >= self.confirmation_threshold:
                text = sign.upper() if not is_word else sign
            elif confidence >= self.uncertain_threshold:
                text = f"I think {sign}"
            else:
                text = f"Maybe {sign}"

            if self.speech_callback:
                try:
                    self.speech_callback(text)
                except Exception:
                    pass

            self.last_spoken_letter = sign
            self.last_spoken_time = current_time

    def _speak_word(self, word: str):
        """Speak a completed word."""
        with self._speech_lock:
            if self.speech_callback:
                try:
                    self.speech_callback(f"Word: {word}")
                except Exception:
                    pass

    def _get_confidence_level(self, confidence: float) -> SignConfidence:
        if confidence >= 0.85:
            return SignConfidence.HIGH
        elif confidence >= 0.65:
            return SignConfidence.MEDIUM
        elif confidence >= 0.45:
            return SignConfidence.LOW
        return SignConfidence.UNCERTAIN

    def _draw_status_panel(
        self, frame: np.ndarray, signs: List[RecognizedSign],
        left_hand, right_hand
    ):
        """Draw enhanced status panel with ML/rule indicators."""
        h, w = frame.shape[:2]
        panel_w = 300
        px = w - panel_w - 10
        py = 10

        # Background
        overlay = frame.copy()
        cv.rectangle(overlay, (px, py), (w - 10, py + 220), self.colors['panel_bg'], -1)
        cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv.rectangle(frame, (px, py), (w - 10, py + 220), (100, 100, 100), 1)

        # Title with ML badge
        title = "SIGN INTERPRETER PRO"
        badge = "ML" if self.ml_classifier.has_alphabet else "RULE"
        badge_color = self.colors['ml_badge'] if self.ml_classifier.has_alphabet else self.colors['rule_badge']

        cv.putText(frame, title, (px + 10, py + 25), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv.rectangle(frame, (px + 210, py + 10), (px + 250, py + 30), badge_color, -1)
        cv.putText(frame, badge, (px + 215, py + 25), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        yo = py + 50

        # Hands
        hands = f"Hands: {'L' if left_hand else '-'} {'R' if right_hand else '-'}"
        cv.putText(frame, hands, (px + 10, yo), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        yo += 25

        # Mode
        cv.putText(frame, f"Mode: {self.mode.value}", (px + 10, yo), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        yo += 30

        # Current recognition
        if signs:
            best = max(signs, key=lambda s: s.confidence)
            color = {
                SignConfidence.HIGH: self.colors['high_confidence'],
                SignConfidence.MEDIUM: self.colors['medium_confidence'],
                SignConfidence.LOW: self.colors['low_confidence'],
            }.get(best.confidence_level, self.colors['uncertain'])

            cv.putText(frame, best.sign.upper(), (px + 10, yo + 30), cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

            # Confidence bar
            bar_w = int(220 * best.confidence)
            cv.rectangle(frame, (px + 10, yo + 45), (px + 10 + bar_w, yo + 55), color, -1)
            cv.rectangle(frame, (px + 10, yo + 45), (px + 230, yo + 55), (100, 100, 100), 1)
            cv.putText(frame, f"{best.confidence:.0%}", (px + 240, yo + 55), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            yo += 70

        # Word buffer
        if self.word_buffer:
            cv.putText(frame, "Spelling:", (px + 10, yo), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv.putText(frame, self.word_buffer.upper(), (px + 80, yo), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -------------------------------------------------------------------------
    # Public API (backward compatible)
    # -------------------------------------------------------------------------

    def get_current_word(self) -> str:
        return self.word_buffer

    def clear_buffer(self):
        self.word_buffer = ""
        self.letter_buffer.clear()
        self.last_stable_sign = ""
        if self.sentence_builder:
            self.sentence_builder.clear()

    def set_mode(self, mode: InterpreterMode):
        self.mode = mode
        self.state.mode = mode
        self.clear_buffer()

    def describe_for_speech(self, signs: List[RecognizedSign], frame_width: int) -> str:
        if not signs:
            if self.word_buffer:
                return f"Currently spelling: {self.word_buffer}"
            return "No signing detected."

        best = max(signs, key=lambda s: s.confidence)
        if best.category == SignCategory.COMMON_WORD:
            return f"Sign: {best.sign}"

        conf_phrase = ""
        if best.confidence_level == SignConfidence.UNCERTAIN:
            conf_phrase = "I'm not sure, but I think they signed "
        elif best.confidence_level == SignConfidence.LOW:
            conf_phrase = "It looks like "

        if best.category == SignCategory.ALPHABET:
            return f"{conf_phrase}letter {best.sign}"
        elif best.category == SignCategory.NUMBER:
            return f"{conf_phrase}number {best.sign}"
        return f"{conf_phrase}{best.sign}"

    def get_summary(self, signs: List[RecognizedSign]) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'current_word': self.word_buffer,
            'signs_detected': len(signs),
            'best_sign': signs[0].sign if signs else None,
            'best_confidence': signs[0].confidence if signs else 0.0,
            'is_signing': len(signs) > 0,
            'ml_available': self.ml_classifier.has_alphabet or self.ml_classifier.has_words,
            'stats': self._stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    def request_repeat(self):
        if self.speech_callback:
            self.speech_callback("I'm not sure, can you sign that again slower?")

    def request_spell(self):
        if self.speech_callback:
            self.speech_callback("Could you spell that out letter by letter?")

    def __del__(self):
        try:
            if self.hands:
                self.hands.close()
        except:
            pass


# =============================================================================
# FACTORY (backward compatible)
# =============================================================================

# Alias for backward compatibility
SignLanguageInterpreter = SignLanguageInterpreterPro


def create_sign_interpreter(
    mode: str = "continuous",
    speech_callback: Optional[Callable[[str], None]] = None,
    model_dir: Optional[str] = None,
    enable_gpt_fallback: bool = True,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
    speak_letters: bool = True,
    speak_words: bool = True,
    enable_visual_feedback: bool = True,
) -> SignLanguageInterpreterPro:
    """
    Factory function to create a SignLanguageInterpreterPro.
    """
    mode_map = {
        'fingerspelling': InterpreterMode.FINGERSPELLING,
        'word_signs': InterpreterMode.WORD_SIGNS,
        'continuous': InterpreterMode.CONTINUOUS,
        'learning': InterpreterMode.LEARNING,
    }

    return SignLanguageInterpreterPro(
        mode=mode_map.get(mode.lower(), InterpreterMode.CONTINUOUS),
        speech_callback=speech_callback,
        model_dir=model_dir,
        enable_gpt_fallback=enable_gpt_fallback,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        speak_letters=speak_letters,
        speak_words=speak_words,
        enable_visual_feedback=enable_visual_feedback,
    )


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤟 Sign Language Interpreter Pro — Test Mode")
    print("=" * 60)

    def test_speech(text: str):
        print(f"🔊 {text}")

    interpreter = create_sign_interpreter(
        mode="continuous",
        speech_callback=test_speech,
        enable_visual_feedback=True,
    )

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🎮 Controls: Q=quit, C=clear, 1=fingerspell, 2=words, 3=continuous")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        signs, annotated = interpreter.process_frame(frame)
        cv.imshow("Sign Language Interpreter Pro", annotated)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            interpreter.clear_buffer()
        elif key == ord('1'):
            interpreter.set_mode(InterpreterMode.FINGERSPELLING)
        elif key == ord('2'):
            interpreter.set_mode(InterpreterMode.WORD_SIGNS)
        elif key == ord('3'):
            interpreter.set_mode(InterpreterMode.CONTINUOUS)

    cap.release()
    cv.destroyAllWindows()
    print(f"\n📊 Stats: {interpreter.get_stats()}")
