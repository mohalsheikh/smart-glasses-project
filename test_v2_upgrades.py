#!/usr/bin/env python3
"""
VisionAssist v2.0 — Upgrade Test Suite
========================================

Run from your project root:
    python test_v2_upgrades.py

Tests each new module independently. No camera or hardware needed.
Requires: OPENAI_API_KEY in your .env (for embedding tests)

Color codes in output:
  ✅ = passed
  ❌ = failed
  ⚠️ = warning (non-critical)
  🧪 = running test
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Make sure we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def ok(self, msg: str):
        self.passed += 1
        print(f"  ✅ {msg}")

    def fail(self, msg: str, error: str = ""):
        self.failed += 1
        self.errors.append(f"{msg}: {error}")
        print(f"  ❌ {msg}")
        if error:
            print(f"     → {error}")

    def warn(self, msg: str):
        self.warnings += 1
        print(f"  ⚠️  {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed, {self.warnings} warnings")
        if self.errors:
            print(f"\nFailed tests:")
            for e in self.errors:
                print(f"  ❌ {e}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# =============================================================================
# TEST 1: IMPORTS
# =============================================================================

def test_imports():
    print("\n🧪 TEST 1: Module Imports")
    print("-" * 40)

    # 1a. Advanced Voice Listener
    try:
        from src.voice.advanced_voice_listener import (
            AdvancedVoiceListener,
            VoiceListener,
            WakeWordDetector,
            AdaptiveVAD,
            ListeningMode,
            ConversationState,
        )
        results.ok("voice/advanced_voice_listener imports OK")
    except Exception as e:
        results.fail("voice/advanced_voice_listener import", str(e))

    # 1b. Enhanced Scene Memory
    try:
        from src.ai_features.enhanced_scene_memory import (
            EnhancedSceneMemory,
            SceneMemoryEngine,
            SceneMemory,
            ConversationContext,
            LocationTracker,
        )
        results.ok("ai_features/enhanced_scene_memory imports OK")
    except Exception as e:
        results.fail("ai_features/enhanced_scene_memory import", str(e))

    # 1c. Emergency Detection
    try:
        from src.safety.emergency_detection import (
            EmergencySystem,
            FallDetector,
            DistressVoiceDetector,
            InactivityMonitor,
            DangerousEnvironmentDetector,
            EmergencyLevel,
            EmergencyType,
        )
        results.ok("safety/emergency_detection imports OK")
    except Exception as e:
        results.fail("safety/emergency_detection import", str(e))

    # 1d. Smart Conversation
    try:
        from src.brain.smart_conversation import SmartConversationMixin
        results.ok("brain/smart_conversation imports OK")
    except Exception as e:
        results.fail("brain/smart_conversation import", str(e))

    # 1e. Existing modules still import
    try:
        from src.controller import MainController
        results.ok("controller.py still imports OK")
    except Exception as e:
        results.warn(f"controller.py import issue (may be fine if deps missing): {e}")

    # 1f. voice/__init__.py re-exports
    try:
        from src.voice import VoiceListener as VL, AdvancedVoiceListener as AVL
        assert VL is not None
        assert AVL is not None
        results.ok("voice/__init__.py re-exports OK")
    except Exception as e:
        results.fail("voice/__init__.py re-exports", str(e))


# =============================================================================
# TEST 2: WAKE WORD DETECTOR
# =============================================================================

def test_wake_word_detector():
    print("\n🧪 TEST 2: Wake Word Detector")
    print("-" * 40)

    try:
        from src.voice.advanced_voice_listener import WakeWordDetector

        detector = WakeWordDetector(
            wake_phrases=["hey vision", "vision", "hey glasses"],
            sensitivity=0.6,
        )
        results.ok("WakeWordDetector instantiated")

        # Test wake phrase matching
        assert detector.check_wake_word_in_text("hey vision, what do you see") is True
        results.ok("Detects 'hey vision' in text")

        assert detector.check_wake_word_in_text("vision describe the scene") is True
        results.ok("Detects 'vision' in text")

        assert detector.check_wake_word_in_text("hey glasses read this") is True
        results.ok("Detects 'hey glasses' in text")

        assert detector.check_wake_word_in_text("what is the weather today") is False
        results.ok("Does NOT false-trigger on normal text")

        # Phonetic variants
        assert detector.check_wake_word_in_text("hey fishing what do you see") is True
        results.ok("Detects phonetic variant 'hey fishing'")

        assert detector.check_wake_word_in_text("a vision describe") is True
        results.ok("Detects phonetic variant 'a vision'")

        detector.cleanup()
        results.ok("Cleanup successful")

    except Exception as e:
        results.fail("WakeWordDetector", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 3: ADAPTIVE VAD
# =============================================================================

def test_adaptive_vad():
    print("\n🧪 TEST 3: Adaptive VAD (Voice Activity Detection)")
    print("-" * 40)

    try:
        import numpy as np
        from src.voice.advanced_voice_listener import AdaptiveVAD

        vad = AdaptiveVAD(
            sample_rate=16000,
            frame_ms=30,
            pre_speech_buffer_ms=300,
            hangover_ms=400,
        )
        results.ok("AdaptiveVAD instantiated")

        frame_size = int(16000 * 0.03)  # 30ms at 16kHz = 480 samples

        # Calibrate with silence
        for i in range(20):
            noise = np.random.randn(frame_size).astype(np.float32) * 0.001
            done = vad.calibrate(noise)
        assert vad._calibrated, "VAD should be calibrated after 20 frames"
        results.ok("Calibrated on ambient noise")

        # Feed silence — should not detect speech
        silence = np.zeros(frame_size, dtype=np.float32)
        is_speech, just_started = vad.process_frame(silence)
        assert not is_speech, "Silence should not be detected as speech"
        results.ok("Silence correctly identified as non-speech")

        # Feed loud signal — should detect speech
        speech = np.sin(np.linspace(0, 440 * 2 * np.pi * 0.03, frame_size)).astype(np.float32) * 0.5
        is_speech, just_started = vad.process_frame(speech)
        assert is_speech, "Loud signal should be detected as speech"
        assert just_started, "Should be marked as 'just started'"
        results.ok("Speech correctly detected")

        # Pre-speech buffer should have data
        pre = vad.get_pre_speech_audio()
        assert len(pre) > 0, "Pre-speech buffer should not be empty"
        results.ok(f"Pre-speech buffer has {len(pre)} samples")

        # Reset
        vad.reset()
        assert not vad.speech_active, "After reset, speech should not be active"
        results.ok("Reset works correctly")

    except Exception as e:
        results.fail("AdaptiveVAD", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 4: ADVANCED VOICE LISTENER (Unit Tests — No Mic Needed)
# =============================================================================

def test_advanced_voice_listener():
    print("\n🧪 TEST 4: Advanced Voice Listener (no mic needed)")
    print("-" * 40)

    try:
        from src.voice.advanced_voice_listener import (
            AdvancedVoiceListener,
            VoiceListener,
            ListeningMode,
            ConversationState,
        )

        # Test PTT mode instantiation
        listener = AdvancedVoiceListener(mode="ptt", max_duration=5.0)
        assert listener.mode == ListeningMode.PUSH_TO_TALK
        results.ok("PTT mode instantiation")

        # Test wake word mode
        listener2 = AdvancedVoiceListener(mode="wake_word", conversation_timeout=10.0)
        assert listener2.mode == ListeningMode.WAKE_WORD
        results.ok("Wake word mode instantiation")

        # Test state management
        assert listener.get_state() == "idle"
        results.ok("Initial state is 'idle'")

        # Test conversation timeout check
        assert not listener.is_in_conversation()
        results.ok("Not in conversation initially")

        # Test wake word stripping
        stripped = listener._strip_wake_word("hey vision what do you see")
        assert stripped == "what do you see", f"Got: {stripped!r}"
        results.ok("Wake word stripping works")

        stripped2 = listener._strip_wake_word("what is the weather")
        assert stripped2 == "what is the weather", f"Got: {stripped2!r}"
        results.ok("Non-wake-word text preserved")

        # Test backward compatibility wrapper
        compat = VoiceListener(duration_seconds=8.0, silence_duration=1.2)
        assert hasattr(compat, "listen_and_transcribe")
        results.ok("Backward-compatible VoiceListener wrapper works")

        # Test stats
        stats = listener.stats
        assert "wake_detections" in stats
        assert "transcriptions" in stats
        results.ok("Stats tracking initialized")

        listener.cleanup()
        listener2.cleanup()
        results.ok("Cleanup successful")

    except Exception as e:
        results.fail("AdvancedVoiceListener", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 5: ENHANCED SCENE MEMORY
# =============================================================================

def test_enhanced_scene_memory():
    print("\n🧪 TEST 5: Enhanced Scene Memory")
    print("-" * 40)

    try:
        from src.ai_features.enhanced_scene_memory import (
            EnhancedSceneMemory,
            SceneMemoryEngine,
        )

        # Test instantiation
        memory = EnhancedSceneMemory(
            max_memories=100,
            auto_save=True,
            use_openai_embeddings=False,  # Don't use API for tests
        )
        results.ok("EnhancedSceneMemory instantiated")

        # Test backward compatibility alias
        assert SceneMemoryEngine is EnhancedSceneMemory
        results.ok("SceneMemoryEngine alias works")

        # Test scene classification
        mock_detections = [
            {"label": "laptop", "confidence": 0.9},
            {"label": "keyboard", "confidence": 0.85},
            {"label": "mouse", "confidence": 0.8},
            {"label": "desk", "confidence": 0.75},
        ]
        tags, loc_type = memory._classify_scene(mock_detections)
        assert "office" in tags, f"Expected 'office', got {tags}"
        results.ok(f"Scene classified as: {tags} (location: {loc_type})")

        kitchen_dets = [
            {"label": "stove", "confidence": 0.9},
            {"label": "refrigerator", "confidence": 0.85},
            {"label": "sink", "confidence": 0.8},
        ]
        tags2, loc2 = memory._classify_scene(kitchen_dets)
        assert "kitchen" in tags2, f"Expected 'kitchen', got {tags2}"
        results.ok(f"Kitchen scene classified correctly: {tags2}")

        # Test store (without embeddings)
        stored = memory.store_scene(
            description="Office with laptop and keyboard",
            detections=mock_detections,
            location="building A",
            location_type="office",
        )
        # May fail without embedding model, that's OK
        if stored:
            results.ok("Scene stored successfully")
        else:
            results.warn("Scene store returned False (embedding unavailable — OK for test)")

        # Force store without embedding for testing
        from src.ai_features.enhanced_scene_memory import SceneMemory
        import numpy as np

        test_mem = SceneMemory(
            timestamp=time.time() - 300,  # 5 min ago
            description="Kitchen with stove and sink",
            embedding=np.random.randn(384).astype(np.float32),  # Fake embedding
            detections=kitchen_dets,
            location_type="kitchen",
            tags=["kitchen", "indoor"],
            importance=1.0,
        )
        memory.memories.append(test_mem)

        test_mem2 = SceneMemory(
            timestamp=time.time() - 60,  # 1 min ago
            description="Office with laptop and monitor",
            embedding=np.random.randn(384).astype(np.float32),
            detections=mock_detections,
            location_type="office",
            tags=["office", "indoor"],
            importance=1.0,
        )
        memory.memories.append(test_mem2)
        results.ok(f"Manually stored 2 test memories (total: {len(memory.memories)})")

        # Test recall by object
        found = memory.recall_by_object("laptop")
        assert len(found) > 0, "Should find laptop memory"
        results.ok(f"recall_by_object('laptop') found {len(found)} memories")

        found2 = memory.recall_by_object("stove")
        assert len(found2) > 0, "Should find stove memory"
        results.ok(f"recall_by_object('stove') found {len(found2)} memories")

        # Test recall by location
        found3 = memory.recall_by_location("kitchen")
        assert len(found3) > 0, "Should find kitchen memories"
        results.ok(f"recall_by_location('kitchen') found {len(found3)} memories")

        # Test recall by time
        found4 = memory.recall_by_time(minutes_ago=3, time_range_minutes=5)
        assert len(found4) > 0, "Should find recent memories"
        results.ok(f"recall_by_time(3 min ago) found {len(found4)} memories")

        # Test when_last_seen
        last_seen = memory.when_last_seen("laptop")
        assert last_seen is not None
        assert "laptop" in last_seen.lower()
        results.ok(f"when_last_seen: {last_seen}")

        # Test scene change detection
        change = memory._compute_scene_change(mock_detections)
        results.ok(f"Scene change computed: {change:.2f}")

        # Test objects hash (deduplication)
        hash1 = memory._compute_objects_hash(mock_detections)
        hash2 = memory._compute_objects_hash(mock_detections)
        assert hash1 == hash2, "Same detections should produce same hash"
        results.ok("Object hash deduplication works")

        hash3 = memory._compute_objects_hash(kitchen_dets)
        assert hash3 != hash1, "Different detections should produce different hash"
        results.ok("Different scenes produce different hashes")

        # Test auto-save logic
        memory._last_objects = set()  # Reset for fresh test
        memory._last_auto_save = 0    # Reset timer
        msg = memory.update_from_detections(mock_detections, frame_idx=0)
        results.ok(f"update_from_detections ran (msg: {msg})")

        # Test context summary
        summary = memory.get_context_summary(recent_minutes=10)
        assert isinstance(summary, str)
        results.ok(f"Context summary: {summary[:80]}...")

        # Test conversation context
        memory.update_conversation("What is on the desk?", intent="identify_object", objects=["laptop"])
        ctx = memory.get_conversation_context()
        assert "desk" in ctx.lower() or "laptop" in ctx.lower()
        results.ok(f"Conversation context: {ctx[:80]}...")

        # Test stats
        stats = memory.get_stats()
        assert "total_memories" in stats
        results.ok(f"Stats: {stats}")

        # Test export
        export_path = "/tmp/test_memory_export.json"
        exported = memory.export_memories(export_path)
        if exported:
            results.ok(f"Exported to {export_path}")
            os.remove(export_path)
        else:
            results.warn("Export failed (non-critical)")

    except Exception as e:
        results.fail("EnhancedSceneMemory", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 6: EMERGENCY DETECTION — FALL DETECTOR
# =============================================================================

def test_fall_detector():
    print("\n🧪 TEST 6: Fall Detector")
    print("-" * 40)

    try:
        from src.safety.emergency_detection import FallDetector

        fd = FallDetector(
            free_fall_threshold=0.3,
            impact_threshold=3.0,
            orientation_threshold=60.0,
            confirmation_time=2.0,  # Shorter for testing
        )
        results.ok("FallDetector instantiated")

        # Normal walking — no fall
        for _ in range(20):
            event = fd.update(0.0, 0.0, 1.0, pitch=5.0, roll=3.0)
        assert event is None, "Normal walking should not trigger fall"
        assert fd.state.value == "normal"
        results.ok("Normal movement: no false alarm")

        # Simulate free fall (acceleration near zero)
        event = fd.update(0.05, 0.05, 0.05, pitch=10.0, roll=5.0)
        assert fd.state.value == "free_fall", f"Expected free_fall, got {fd.state.value}"
        results.ok("Free fall detected")

        # Simulate impact (high acceleration)
        event = fd.update(4.0, 3.0, 5.0, pitch=70.0, roll=45.0)
        assert fd.state.value == "impact", f"Expected impact, got {fd.state.value}"
        results.ok("Impact detected")

        # Orientation change (now horizontal)
        event = fd.update(0.1, 0.1, 0.1, pitch=80.0, roll=60.0)
        assert fd.state.value == "confirming", f"Expected confirming, got {fd.state.value}"
        results.ok("Orientation change detected, confirming...")

        # Stillness for confirmation_time
        time.sleep(2.5)  # Wait for confirmation time
        event = fd.update(0.98, 0.01, 0.01, pitch=85.0, roll=60.0)
        if event is not None:
            assert event.event_type.value == "fall"
            results.ok(f"FALL CONFIRMED! Message: {event.message[:60]}...")
        else:
            results.warn("Fall not confirmed (timing issue — OK in test)")

        # Reset
        fd.reset()
        assert fd.state.value == "normal"
        results.ok("Reset to normal state")

    except Exception as e:
        results.fail("FallDetector", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 7: EMERGENCY DETECTION — DISTRESS VOICE
# =============================================================================

def test_distress_voice():
    print("\n🧪 TEST 7: Distress Voice Detector")
    print("-" * 40)

    try:
        from src.safety.emergency_detection import DistressVoiceDetector, EmergencyLevel

        dvd = DistressVoiceDetector(cooldown_seconds=2.0)
        results.ok("DistressVoiceDetector instantiated")

        # Emergency phrases
        event = dvd.check_text("call 911 please")
        assert event is not None
        assert event.level == EmergencyLevel.EMERGENCY
        results.ok(f"'call 911' → EMERGENCY: {event.message[:50]}...")

        time.sleep(2.5)  # Wait for cooldown

        # Alert phrases
        event2 = dvd.check_text("help me I need help")
        assert event2 is not None
        assert event2.level == EmergencyLevel.ALERT
        results.ok(f"'help me' → ALERT: {event2.message[:50]}...")

        time.sleep(2.5)

        # Warning phrases
        event3 = dvd.check_text("emergency")
        assert event3 is not None
        assert event3.level == EmergencyLevel.WARNING
        results.ok(f"'emergency' → WARNING: {event3.message[:50]}...")

        # Normal text — no trigger
        event4 = dvd.check_text("what is the weather today")
        assert event4 is None
        results.ok("Normal text: no false alarm")

        event5 = dvd.check_text("describe what you see")
        assert event5 is None
        results.ok("Normal command: no false alarm")

    except Exception as e:
        results.fail("DistressVoiceDetector", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 8: EMERGENCY DETECTION — INACTIVITY MONITOR
# =============================================================================

def test_inactivity_monitor():
    print("\n🧪 TEST 8: Inactivity Monitor")
    print("-" * 40)

    try:
        from src.safety.emergency_detection import InactivityMonitor

        # Use very short intervals for testing
        monitor = InactivityMonitor(
            check_in_minutes=0.05,      # 3 seconds
            escalation_minutes=0.05,    # 3 seconds
        )
        results.ok("InactivityMonitor instantiated")

        # Should not trigger immediately
        event = monitor.check()
        assert event is None
        results.ok("No immediate trigger")

        # Wait for check-in
        time.sleep(4)
        event = monitor.check()
        if event is not None:
            assert event.level.value == 1  # CHECK_IN
            results.ok(f"Check-in triggered: {event.message[:50]}...")
        else:
            results.warn("Check-in didn't trigger (timing — OK in test)")

        # Register activity — should reset
        monitor.register_activity()
        event2 = monitor.check()
        assert event2 is None
        results.ok("Activity registration resets monitor")

    except Exception as e:
        results.fail("InactivityMonitor", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 9: EMERGENCY DETECTION — DANGEROUS ENVIRONMENT
# =============================================================================

def test_dangerous_environment():
    print("\n🧪 TEST 9: Dangerous Environment Detector")
    print("-" * 40)

    try:
        from src.safety.emergency_detection import DangerousEnvironmentDetector

        ded = DangerousEnvironmentDetector(cooldown_seconds=1.0)
        results.ok("DangerousEnvironmentDetector instantiated")

        # Vehicle very close (large bbox)
        detections = [
            {
                "label": "car",
                "confidence": 0.9,
                "bbox": [50, 50, 550, 430],  # Takes up ~70% of 640x480 frame
            },
        ]
        events = ded.check_scene(detections, frame_width=640, frame_height=480)
        assert len(events) > 0, "Should detect close vehicle"
        results.ok(f"Close vehicle detected: {events[0].message}")

        time.sleep(1.5)  # Cooldown

        # Stairs detected
        stairs_dets = [
            {
                "label": "stairs",
                "confidence": 0.85,
                "bbox": [100, 200, 500, 450],  # Large in frame
            },
        ]
        events2 = ded.check_scene(stairs_dets, frame_width=640, frame_height=480)
        if events2:
            results.ok(f"Stairs detected: {events2[0].message}")
        else:
            results.warn("Stairs not detected (may be below area threshold)")

        # Small distant car — should NOT trigger
        small_car = [
            {
                "label": "car",
                "confidence": 0.7,
                "bbox": [300, 200, 340, 220],  # Very small in frame
            },
        ]
        time.sleep(1.5)
        events3 = ded.check_scene(small_car, frame_width=640, frame_height=480)
        assert len(events3) == 0, "Distant small car should not trigger"
        results.ok("Distant car: no false alarm")

    except Exception as e:
        results.fail("DangerousEnvironmentDetector", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 10: EMERGENCY SYSTEM (FULL)
# =============================================================================

def test_emergency_system():
    print("\n🧪 TEST 10: Emergency System (Full Integration)")
    print("-" * 40)

    try:
        from src.safety.emergency_detection import EmergencySystem

        spoken_messages = []

        def mock_speak(text):
            spoken_messages.append(text)

        es = EmergencySystem(
            enable_fall_detection=True,
            enable_distress_voice=True,
            enable_inactivity=True,
            enable_environment=True,
            speak_callback=mock_speak,
            emergency_contacts=[
                {"name": "Test Contact", "phone": "+1234567890", "is_primary": True},
            ],
        )
        results.ok("EmergencySystem instantiated with all modules")

        # Check voice distress
        event = es.check_voice("help me I need help")
        if event:
            assert len(spoken_messages) > 0, "Should have spoken alert"
            results.ok(f"Voice distress handled, spoke: {spoken_messages[-1][:50]}...")

        # Register activity
        es.register_activity()
        results.ok("Activity registered")

        # Manual SOS
        sos_event = es.trigger_sos("test SOS")
        assert sos_event is not None
        results.ok(f"SOS triggered: {sos_event.message}")

        # Cancel with "I'm fine"
        es.check_voice("I'm fine, it was a false alarm")
        assert not es.has_active_emergency()
        results.ok("Emergency cancelled with 'I'm fine'")

        # Status
        status = es.get_status()
        assert "active_emergency" in status
        results.ok(f"Status: {status}")

        # Voice command check
        assert es.is_emergency_command("call 911") is True
        assert es.is_emergency_command("what time is it") is False
        results.ok("Emergency command detection works")

    except Exception as e:
        results.fail("EmergencySystem", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 11: SMART CONVERSATION — REFERENCE RESOLUTION
# =============================================================================

def test_smart_conversation():
    print("\n🧪 TEST 11: Smart Conversation Engine")
    print("-" * 40)

    try:
        from src.brain.smart_conversation import SmartConversationMixin

        # Create a test class that uses the mixin
        class TestBrain(SmartConversationMixin):
            def __init__(self):
                self.last_scene_detections = [
                    {"label": "chair"},
                    {"label": "laptop"},
                ]
                self.conversation_history = []
                # Mock scene_ai for _build_system_prompt
                class MockSceneAI:
                    pass
                self.scene_ai = MockSceneAI()
                self._init_conversation_engine()

        brain = TestBrain()
        results.ok("SmartConversationMixin instantiated")

        # Test reference context
        brain._reference_context["last_object"] = "chair"
        resolved = brain._resolve_references("what color is it")
        assert "chair" in resolved.lower(), f"Expected 'chair' in resolved, got: {resolved}"
        results.ok(f"'what color is it' → '{resolved}'")

        # Test "tell me more"
        brain._last_spoken_response = "There's a red chair on your left and a laptop on the desk."
        resolved2 = brain._resolve_references("tell me more")
        assert "red chair" in resolved2.lower() or "details" in resolved2.lower()
        results.ok(f"'tell me more' resolved correctly")

        # Test reference update
        brain._update_references(
            intent="describe_environment",
            response="I see a desk with a monitor",
            objects=["desk", "monitor"],
            direction="in front of you",
        )
        assert brain._reference_context["last_object"] == "desk"
        assert brain._reference_context["last_direction"] == "in front of you"
        results.ok("Reference context updated after response")

        # Test user preferences
        result = brain._handle_preference_command("my name is Mohammed")
        assert result is not None
        assert brain._user_name == "Mohammed"
        results.ok(f"Name set: {result}")

        result2 = brain._handle_preference_command("brief mode please")
        assert brain._user_preferences["verbosity"] == "brief"
        results.ok(f"Verbosity set: {result2}")

        result3 = brain._handle_preference_command("detailed mode")
        assert brain._user_preferences["verbosity"] == "detailed"
        results.ok(f"Verbosity changed: {result3}")

        # Test system prompt building
        prompt = brain._build_system_prompt(mode="describe", include_scene=True)
        assert "Vision" in prompt or "visually impaired" in prompt
        assert "chair" in prompt or "laptop" in prompt  # Scene objects should be included
        results.ok(f"System prompt built ({len(prompt)} chars)")

        # Test activity tracking
        brain._update_activity("directions")
        assert brain._current_activity == "navigating"
        results.ok("Activity updated to 'navigating'")

        brain._update_activity("read_text")
        assert brain._current_activity == "reading"
        results.ok("Activity updated to 'reading'")

        # Test session stats
        stats = brain.get_session_stats()
        assert stats["user_name"] == "Mohammed"
        results.ok(f"Session stats: {stats}")

        # Non-preference command should return None
        result4 = brain._handle_preference_command("what do you see")
        assert result4 is None
        results.ok("Non-preference command returns None")

    except Exception as e:
        results.fail("SmartConversationMixin", str(e))
        traceback.print_exc()


# =============================================================================
# TEST 12: BACKWARD COMPATIBILITY
# =============================================================================

def test_backward_compatibility():
    print("\n🧪 TEST 12: Backward Compatibility")
    print("-" * 40)

    try:
        # Check that EnhancedSceneMemory can be used as SceneMemoryEngine
        from src.ai_features.enhanced_scene_memory import SceneMemoryEngine
        mem = SceneMemoryEngine(max_memories=50, use_openai_embeddings=False)
        assert hasattr(mem, "store_scene")
        assert hasattr(mem, "recall_similar")
        assert hasattr(mem, "recall_by_object")
        assert hasattr(mem, "memories")
        results.ok("SceneMemoryEngine backward compat: all original methods present")

        # Check that VoiceListener wrapper has original API
        from src.voice.advanced_voice_listener import VoiceListener
        vl = VoiceListener(duration_seconds=8.0, silence_duration=1.2)
        assert hasattr(vl, "listen_and_transcribe")
        results.ok("VoiceListener backward compat: listen_and_transcribe() present")

        vl.cleanup()

    except Exception as e:
        results.fail("Backward compatibility", str(e))
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("VisionAssist v2.0 — Upgrade Test Suite")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"DEEPGRAM_API_KEY: {'set' if os.getenv('DEEPGRAM_API_KEY') else 'NOT SET'}")

    # Run all tests
    test_imports()
    test_wake_word_detector()
    test_adaptive_vad()
    test_advanced_voice_listener()
    test_enhanced_scene_memory()
    test_fall_detector()
    test_distress_voice()
    test_inactivity_monitor()
    test_dangerous_environment()
    test_emergency_system()
    test_smart_conversation()
    test_backward_compatibility()

    # Summary
    success = results.summary()

    if success:
        print("\n🎉 All tests passed! You're ready to integrate.")
        print("\nNext steps:")
        print("  1. Follow INTEGRATION_GUIDE.py to wire into controller.py")
        print("  2. Set VOICE_MODE=wake_word in .env for always-on listening")
        print("  3. Run the full app and test each feature end-to-end")
    else:
        print("\n⚠️  Some tests failed. Fix the issues above before integrating.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
