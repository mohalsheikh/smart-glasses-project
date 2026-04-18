# ---------------------------------------------------------------------------
# Sign Language Interpreter Settings
# ---------------------------------------------------------------------------
# Add these settings to your src/utils/config.py file

# Enable/disable sign language interpreter feature
SIGN_LANGUAGE_ENABLED: bool = os.getenv("SIGN_LANGUAGE_ENABLED", "1") == "1"

# Recognition mode: "fingerspelling" | "word_signs" | "continuous"
SIGN_LANGUAGE_MODE: str = os.getenv("SIGN_LANGUAGE_MODE", "continuous").strip().lower()

# Minimum detection confidence for hand detection (0.0-1.0)
SIGN_DETECTION_CONFIDENCE: float = float(os.getenv("SIGN_DETECTION_CONFIDENCE", "0.7"))

# Minimum tracking confidence for hand tracking (0.0-1.0)
SIGN_TRACKING_CONFIDENCE: float = float(os.getenv("SIGN_TRACKING_CONFIDENCE", "0.5"))

# Confidence threshold for automatic speech output (0.0-1.0)
# Signs with confidence above this will be spoken with certainty
SIGN_CONFIRMATION_THRESHOLD: float = float(os.getenv("SIGN_CONFIRMATION_THRESHOLD", "0.75"))

# Confidence threshold below which uncertainty prompts are used
SIGN_UNCERTAIN_THRESHOLD: float = float(os.getenv("SIGN_UNCERTAIN_THRESHOLD", "0.55"))

# Whether to speak individual letters during fingerspelling
SIGN_SPEAK_LETTERS: bool = os.getenv("SIGN_SPEAK_LETTERS", "1") == "1"

# Whether to speak completed words after pauses
SIGN_SPEAK_WORDS: bool = os.getenv("SIGN_SPEAK_WORDS", "1") == "1"

# Pause duration (seconds) to trigger word completion
SIGN_WORD_PAUSE_THRESHOLD: float = float(os.getenv("SIGN_WORD_PAUSE_THRESHOLD", "1.0"))

# Minimum time between speaking letters (seconds)
SIGN_MIN_LETTER_INTERVAL: float = float(os.getenv("SIGN_MIN_LETTER_INTERVAL", "0.5"))

# Minimum time to hold a sign before recognition (seconds)
SIGN_MIN_HOLD_TIME: float = float(os.getenv("SIGN_MIN_HOLD_TIME", "0.15"))

# Number of consistent frames required for stable recognition
SIGN_STABILITY_FRAMES: int = int(os.getenv("SIGN_STABILITY_FRAMES", "3"))

# Whether to show visual feedback overlay
SIGN_VISUAL_FEEDBACK: bool = os.getenv("SIGN_VISUAL_FEEDBACK", "1") == "1"

# Maximum number of hands to track simultaneously
SIGN_MAX_HANDS: int = int(os.getenv("SIGN_MAX_HANDS", "2"))
