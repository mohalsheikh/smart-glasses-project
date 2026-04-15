# INTEGRATION_GUIDE.md
# VisionAssist v2.0 — Integration Guide
# ========================================
#
# This guide shows you exactly how to integrate the 4 new upgraded modules
# into your existing codebase. Each section shows what to change and where.
#
# NEW MODULES:
# 1. src/voice/advanced_voice_listener.py  — Wake word + smart VAD + conversation mode
# 2. src/ai_features/enhanced_scene_memory.py — Auto-save + temporal + context memory
# 3. src/safety/emergency_detection.py — Fall detection + distress + SOS
# 4. src/brain/smart_conversation.py — Natural dialogue + references + proactive
#
# ==============================================================================

# ==============================================================================
# STEP 1: FILE PLACEMENT
# ==============================================================================
#
# Copy the new files into your project:
#
#   src/
#   ├── voice/                          ← NEW FOLDER
#   │   ├── __init__.py                 ← NEW
#   │   └── advanced_voice_listener.py  ← NEW
#   ├── ai_features/
#   │   ├── enhanced_scene_memory.py    ← NEW (alongside existing scene_memory.py)
#   │   └── ... (existing files)
#   ├── safety/
#   │   ├── emergency_detection.py      ← NEW
#   │   └── ... (existing files)
#   ├── brain/
#   │   ├── smart_conversation.py       ← NEW
#   │   └── ... (existing files)
#   └── controller.py                   ← MODIFY (changes below)

# ==============================================================================
# STEP 2: MODIFY controller.py — IMPORTS
# ==============================================================================
#
# Replace/add these imports at the top of controller.py:

"""
# --- REPLACE this import ---
# OLD:
from src.voice_listener import VoiceListener

# NEW:
from src.voice.advanced_voice_listener import AdvancedVoiceListener as VoiceListener

# --- REPLACE this import ---
# OLD:
from src.ai_features.scene_memory import SceneMemoryEngine

# NEW:
from src.ai_features.enhanced_scene_memory import EnhancedSceneMemory as SceneMemoryEngine

# --- ADD this new import ---
from src.safety.emergency_detection import EmergencySystem
"""

# ==============================================================================
# STEP 3: MODIFY controller.py — __init__ METHOD
# ==============================================================================
#
# In MainController.__init__(), ADD these after existing initialization:

"""
    def __init__(self):
        # ... existing init code ...

        # --- UPGRADE: Replace VoiceListener with AdvancedVoiceListener ---
        self.voice_listener = VoiceListener(
            duration_seconds=10.0,
            sample_rate=16000,
            silence_duration=1.2,
        )
        # The new VoiceListener is backward-compatible — same API!
        # To enable wake word mode, set env: VOICE_MODE=wake_word
        # To enable conversation mode timeout: CONVERSATION_TIMEOUT=15

        # --- UPGRADE: Replace SceneMemoryEngine with EnhancedSceneMemory ---
        self.scene_memory = SceneMemoryEngine(
            max_memories=1000,
            auto_save=True,                # NEW: auto-saves scene changes
            auto_save_interval=10.0,       # NEW: minimum seconds between saves
            min_change_threshold=0.3,      # NEW: min scene change to trigger save
        )

        # --- NEW: Emergency Detection System ---
        self.emergency = EmergencySystem(
            enable_fall_detection=True,     # Uses IMU data (BNO085)
            enable_distress_voice=True,     # Listens for "help", "emergency"
            enable_inactivity=True,         # Checks in after 30min silence
            enable_environment=True,        # Detects dangerous objects
            speak_callback=self._emergency_speak,
            emergency_contacts=[
                # Add your emergency contacts here:
                # {"name": "Mom", "phone": "+1234567890", "is_primary": True},
            ],
        )

        print("🆕 VisionAssist v2.0 upgrades loaded!")
        print("   ✅ Advanced Voice (wake word + conversation mode)")
        print("   ✅ Enhanced Scene Memory (auto-save + temporal)")
        print("   ✅ Emergency Detection (fall + distress + SOS)")
        print("   ✅ Smart Conversation (references + proactive)")
"""

# ==============================================================================
# STEP 4: ADD EMERGENCY SPEAK CALLBACK
# ==============================================================================

"""
    def _emergency_speak(self, text: str):
        '''Emergency speech callback — bypasses queue, speaks immediately.'''
        if not text:
            return
        # For emergencies, interrupt current speech and speak immediately
        if hasattr(self.speech, 'interrupt'):
            self.speech.interrupt()
        self._speak_blocking(text, meta={"source": "emergency", "priority": "high"})
"""

# ==============================================================================
# STEP 5: MODIFY THE MAIN LOOP — ADD AUTO-SAVE AND EMERGENCY CHECKS
# ==============================================================================
#
# In the main run() loop, after detection, ADD these lines:

"""
        # Inside the main while True loop, after:
        #   self.last_detections = detections
        #   self.last_annotated = annotated_frame

        # --- NEW: Auto-save scene memory ---
        if self.scene_memory and hasattr(self.scene_memory, 'update_from_detections'):
            auto_msg = self.scene_memory.update_from_detections(
                detections=detections,
                frame_idx=frame_idx,
            )
            if auto_msg:
                print(f"🧠 {auto_msg}")

        # --- NEW: Emergency environment check (every 10 frames) ---
        if frame_idx % 10 == 0 and self.emergency:
            danger_events = self.emergency.check_scene(
                detections=detections,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            # Danger alerts are handled by the emergency system's speak callback

        # --- NEW: Inactivity check (every 300 frames ≈ every 10 seconds) ---
        if frame_idx % 300 == 0 and self.emergency:
            self.emergency.check_inactivity()
"""

# ==============================================================================
# STEP 6: MODIFY VOICE INTERACTION — ADD EMERGENCY + PREFERENCE COMMANDS
# ==============================================================================
#
# In _handle_voice_interaction() worker function, ADD checks before routing:

"""
        def worker():
            # ... existing code up to transcription ...

            if not text:
                self._speak_blocking("I didn't catch that. Try again.")
                return

            # --- NEW: Register activity for inactivity monitor ---
            if self.emergency:
                self.emergency.register_activity()

            # --- NEW: Check for emergency commands FIRST (highest priority) ---
            if self.emergency and self.emergency.is_emergency_command(text):
                response = self.emergency.handle_emergency_command(text)
                if response:
                    self._speak_blocking(response, meta={"source": "emergency"})
                    return

            # --- NEW: Check distress in voice (even non-commands) ---
            if self.emergency:
                distress = self.emergency.check_voice(text)
                # If distress detected, the emergency system handles speaking
                if distress and distress.level.value >= 3:
                    return  # Emergency system took over

            # ... rest of existing command routing ...
            # (sign language, toggle warnings, read mode, doc commands, brain)
"""

# ==============================================================================
# STEP 7: MODIFY AssistantBrain — ADD SMART CONVERSATION
# ==============================================================================
#
# In src/brain/assistant_brain_impl.py, add SmartConversationMixin:

"""
# ADD IMPORT:
from src.brain.smart_conversation import SmartConversationMixin

# MODIFY CLASS DEFINITION:
class AssistantBrain(
    SmartConversationMixin,      # ← ADD THIS
    IntentDetectionMixin,
    VisionHandlersMixin,
    NavigationHandlersMixin,
    SystemHandlersMixin,
):

    def __init__(self, ...):
        # ... existing init ...

        # ADD AT END OF __init__:
        self._init_conversation_engine()

    # MODIFY handle_query to use smart features:
    def handle_query(self, text, *, frame, detections=None):
        if not text or not text.strip():
            return "I didn't hear anything. Could you repeat that?"

        text = text.strip()

        # --- NEW: Check preference commands ---
        pref_response = self._handle_preference_command(text)
        if pref_response:
            self._append_history("user", text)
            self._append_history("assistant", pref_response)
            return pref_response

        # --- NEW: Resolve references ("it", "that", "tell me more") ---
        resolved_text = self._resolve_references(text)
        if resolved_text != text:
            print(f"  → Resolved: {resolved_text!r}")

        # ... rest of existing handle_query, but use resolved_text
        #     instead of text for intent detection ...

        self._append_history("user", text)

        intent_data = self._detect_intents(resolved_text)
        primary_intent = intent_data.get("primary_intent", "general_question")

        # --- NEW: Update activity tracking ---
        self._update_activity(primary_intent)

        # ... existing handler routing ...

        # --- NEW: Update references after response ---
        objects = [d.get("label", "") for d in (detections or [])[:5]]
        self._update_references(primary_intent, final_answer, objects=objects)

        # --- NEW: Enhance response ---
        final_answer = self._enhance_response(final_answer, primary_intent)

        self._append_history("assistant", final_answer)
        return final_answer
"""

# ==============================================================================
# STEP 8: ENVIRONMENT VARIABLES (.env)
# ==============================================================================
#
# Add these to your .env file:

"""
# Voice Mode: "ptt" (push-to-talk) or "wake_word" (always-on) or "continuous"
VOICE_MODE=ptt

# Conversation timeout (seconds) — how long to stay in conversation after wake word
CONVERSATION_TIMEOUT=15

# Porcupine wake word engine (optional — get free key at console.picovoice.ai)
PORCUPINE_ACCESS_KEY=

# Emergency contacts (configured in code, see Step 3)
"""

# ==============================================================================
# STEP 9: ADDITIONAL DEPENDENCIES (requirements.txt)
# ==============================================================================
#
# Add these to your requirements:

"""
# Wake word detection (optional but recommended)
pvporcupine>=3.0.0

# Existing deps (make sure these are present)
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.24.0
openai>=1.3.0
deepgram-sdk>=3.0.0     # optional, for better transcription
webrtcvad>=2.0.10        # optional, for better VAD
sentence-transformers>=2.2.0  # optional, for local embeddings
"""

# ==============================================================================
# STEP 10: TESTING CHECKLIST
# ==============================================================================
#
# After integration, test each feature:
#
# □ Basic voice still works (press 'v', speak, get response)
# □ Wake word detection (set VOICE_MODE=wake_word, say "hey vision")
# □ Conversation mode (after wake word, speak again without wake word)
# □ Scene memory auto-save (check logs for "🧠" messages)
# □ "What did I see earlier?" query
# □ "When did I last see [object]?" query
# □ "Tell me more" reference resolution
# □ Emergency: say "help me" or "emergency"
# □ Emergency: say "I'm fine" to cancel
# □ Preference: "call me [name]", "brief mode", "detailed mode"
# □ Proactive hints appear occasionally
# □ Keyboard shortcuts still work (d, v, r, s, etc.)
#
# ==============================================================================
# QUICK START — MINIMAL INTEGRATION
# ==============================================================================
#
# If you want to start with just ONE upgrade at a time:
#
# 1. EASIEST: Enhanced Scene Memory
#    - Just swap the import in controller.py
#    - Add update_from_detections() call in the main loop
#    - Everything else is automatic
#
# 2. MEDIUM: Emergency System
#    - Add the import and initialization
#    - Add check_scene() and check_inactivity() in main loop
#    - Add emergency command check in voice handler
#
# 3. MEDIUM: Advanced Voice
#    - Swap the import (backward compatible!)
#    - Set VOICE_MODE=wake_word when ready for always-on
#
# 4. ADVANCED: Smart Conversation
#    - Requires modifying AssistantBrain class
#    - Add mixin + init + reference resolution
#    - Most impactful but most code changes
