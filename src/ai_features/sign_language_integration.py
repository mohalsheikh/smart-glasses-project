"""
Sign Language Mode Integration for MainController
==================================================

This file contains the code changes needed to integrate the Sign Language
Interpreter into the main controller. Copy these changes into your
existing controller.py file.

Integration Steps:
1. Add imports at top of controller.py
2. Add initialization in __init__
3. Add mode toggle handling
4. Add keyboard shortcut for activation
5. Add processing in main loop

Author: VisionAssist AI Team
"""

# =============================================================================
# STEP 1: Add these imports at the top of controller.py (after existing imports)
# =============================================================================

IMPORTS_TO_ADD = '''
from src.ai_features.sign_language_interpreter import (
    SignLanguageInterpreter,
    InterpreterMode,
    create_sign_interpreter,
)
'''

# =============================================================================
# STEP 2: Add these lines in MainController.__init__ (after human_analyzer init)
# =============================================================================

INIT_CODE_TO_ADD = '''
        # ----------------------------
        # Sign Language Interpreter
        # ----------------------------
        self.sign_interpreter = None
        self._sign_mode_enabled = False
        
        try:
            # Create sign interpreter with speech callback
            self.sign_interpreter = create_sign_interpreter(
                mode="continuous",
                speech_callback=self._speak_sign_callback,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                speak_letters=True,
                speak_words=True,
                enable_visual_feedback=True,
            )
            print("🤟 Sign Language Interpreter enabled (press 'g' to toggle)")
        except Exception as e:
            print(f"⚠️ Sign Language Interpreter not available: {e}")
'''

# =============================================================================
# STEP 3: Add this callback method to MainController class
# =============================================================================

CALLBACK_METHOD = '''
    def _speak_sign_callback(self, text: str) -> None:
        """Callback for sign language interpreter to speak recognized signs."""
        if not text:
            return
        
        # Use non-blocking speech
        def speak_thread():
            try:
                self._speak_blocking(text, meta={"source": "sign_interpreter"})
            except Exception as e:
                print(f"⚠️ Sign speech error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()
'''

# =============================================================================
# STEP 4: Add keyboard handling in main loop (inside the key handling section)
# =============================================================================

KEY_HANDLING_CODE = '''
                elif key == ord("g"):
                    # Toggle sign language mode
                    self._sign_mode_enabled = not self._sign_mode_enabled
                    state = "ON" if self._sign_mode_enabled else "OFF"
                    print(f"🤟 Sign language mode: {state}")
                    self._speak_blocking(
                        f"Sign language interpreter {state.lower()}.", 
                        meta={"source": "system"}
                    )
'''

# =============================================================================
# STEP 5: Add processing in main loop (after human analyzer processing)
# =============================================================================

PROCESSING_CODE = '''
                    # Process sign language if enabled
                    if (self._sign_mode_enabled and 
                        self.sign_interpreter is not None):
                        try:
                            signs, annotated_frame = self.sign_interpreter.process_frame(
                                annotated_frame, 
                                detections
                            )
                            if signs:
                                # Log sign detection
                                best_sign = max(signs, key=lambda s: s.confidence)
                                log_ai({
                                    "type": "sign_detected",
                                    "sign": best_sign.sign,
                                    "confidence": best_sign.confidence,
                                    "category": best_sign.category.value,
                                })
                        except Exception as e:
                            pass  # Don't crash on sign recognition errors
'''


def print_integration_guide():
    """Print a step-by-step integration guide."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SIGN LANGUAGE INTERPRETER - INTEGRATION GUIDE                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

This guide explains how to integrate the Sign Language Interpreter into your
VisionAssist Smart Glasses controller.

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Add Imports (top of controller.py)                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Add after the existing imports (around line 40):

    from src.ai_features.sign_language_interpreter import (
        SignLanguageInterpreter,
        InterpreterMode,
        create_sign_interpreter,
    )

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Initialize Sign Interpreter (in __init__, after human_analyzer)     │
└──────────────────────────────────────────────────────────────────────────────┘

Add around line 370 (after human analyzer initialization):

        # Sign Language Interpreter
        self.sign_interpreter = None
        self._sign_mode_enabled = False
        
        try:
            self.sign_interpreter = create_sign_interpreter(
                mode="continuous",
                speech_callback=self._speak_sign_callback,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                speak_letters=True,
                speak_words=True,
                enable_visual_feedback=True,
            )
            print("🤟 Sign Language Interpreter enabled (press 'g' to toggle)")
        except Exception as e:
            print(f"⚠️ Sign Language Interpreter not available: {e}")

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Add Speech Callback Method (new method in MainController)           │
└──────────────────────────────────────────────────────────────────────────────┘

Add this method to the MainController class:

    def _speak_sign_callback(self, text: str) -> None:
        \"\"\"Callback for sign language interpreter to speak recognized signs.\"\"\"
        if not text:
            return
        
        def speak_thread():
            try:
                self._speak_blocking(text, meta={"source": "sign_interpreter"})
            except Exception as e:
                print(f"⚠️ Sign speech error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Add Keyboard Handler (in run() method, key handling section)        │
└──────────────────────────────────────────────────────────────────────────────┘

Add around line 1460 (after the 'h' key handler for human visualization):

                elif key == ord("g"):
                    # Toggle sign language mode
                    self._sign_mode_enabled = not self._sign_mode_enabled
                    state = "ON" if self._sign_mode_enabled else "OFF"
                    print(f"🤟 Sign language mode: {state}")
                    self._speak_blocking(
                        f"Sign language interpreter {state.lower()}.", 
                        meta={"source": "system"}
                    )

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Add Processing in Main Loop (after human analyzer processing)       │
└──────────────────────────────────────────────────────────────────────────────┘

Add around line 1355 (after the human analyzer block):

                    # Process sign language if enabled
                    if self._sign_mode_enabled and self.sign_interpreter is not None:
                        try:
                            signs, annotated_frame = self.sign_interpreter.process_frame(
                                annotated_frame, detections
                            )
                            if signs:
                                best_sign = max(signs, key=lambda s: s.confidence)
                                log_ai({
                                    "type": "sign_detected",
                                    "sign": best_sign.sign,
                                    "confidence": best_sign.confidence,
                                    "category": best_sign.category.value,
                                })
                        except Exception as e:
                            pass  # Don't crash on sign recognition errors

┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Update Help Text (in __init__)                                      │
└──────────────────────────────────────────────────────────────────────────────┘

Update the controls print statement around line 340:

        print("🎛 Controls: 'q' quit | 'd' describe | 'v' voice | 'r' read | 's' toggle warnings")
        print("📖 Reading mode keys: '1' offline | '2' hybrid | '3' AI-only | 'm' cycle mode")
        print("🤟 Sign language: 'g' toggle sign mode")

╔══════════════════════════════════════════════════════════════════════════════╗
║                              USAGE INSTRUCTIONS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Run the smart glasses system as usual
2. Press 'g' to enable Sign Language Interpreter mode
3. The system will now recognize ASL signs and speak them aloud
4. Press 'g' again to disable sign language mode

SUPPORTED SIGNS:
- All 26 ASL alphabet letters (A-Z)
- Numbers 0-9
- 50+ common words/phrases including:
  • Greetings: hello, goodbye, please, thank you, sorry
  • Questions: what, where, who, when, why, how
  • Common words: yes, no, help, stop, more, want, need, like
  • And many more!

TIPS FOR BEST RECOGNITION:
- Hold signs steady for at least 200ms
- Keep hands in camera view
- Good lighting improves accuracy
- Fingerspelling works best with clear hand movements

╔══════════════════════════════════════════════════════════════════════════════╗
║                           VOICE COMMAND INTEGRATION                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

You can also add voice command support for sign language mode.
Add these phrases to your intent detection system:

- "enable sign language" / "turn on sign mode" → Enable sign interpreter
- "disable sign language" / "turn off sign mode" → Disable sign interpreter  
- "what did they sign" → Describe last recognized sign
- "clear sign buffer" → Clear the current word buffer
- "switch to fingerspelling" → Change to fingerspelling-only mode
- "switch to word signs" → Change to word signs mode

""")


if __name__ == "__main__":
    print_integration_guide()
