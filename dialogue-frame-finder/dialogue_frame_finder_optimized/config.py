"""
Global configuration — shared mutable state for the pipeline.

All stage functions read from this module; ``main()`` writes to it once at
startup.  Modules import it as ``from . import config`` (or ``import config``
in the single-file variant) and access values as ``config.DEVICE`` etc.

Pattern: module-level variables are set by ``main.main()`` before any stage
runs.  No locks needed — the pipeline is single-threaded.
"""

# ---- device ---------------------------------------------------------------

DEVICE: str = "cpu"

# ---- paths ----------------------------------------------------------------

WORK_DIR: str = "./output"
VIDEO_PATH: str = ""
AUDIO_PATH: str = ""
AUDIO_SLICE_PATH: str = ""
FRAME_OUT_PATH: str = ""

# ---- thresholds -----------------------------------------------------------

MATCH_THRESHOLD: float = 80.0
COARSE_MATCH_THRESHOLD: float = 50.0
CANDIDATE_WINDOW_BUFFER_SEC: float = 45.0

# ---- tiering (seconds of *speech*) ----------------------------------------

SHORT_MAX_SEC: int = 180     # < 3 min  speech → "short"  tier
MEDIUM_MAX_SEC: int = 1200   # < 20 min speech → "medium" tier
                              # >= 20 min speech → "long"   tier

# ---- models ---------------------------------------------------------------

TIER_MODEL_MAP: dict = {}
LONG_COARSE_MODEL: str = "tiny.en"
LONG_FINE_MODEL: str = "tiny.en"
CPU_THREADS: int = 4

# ---- cookies (yt-dlp) -----------------------------------------------------

COOKIES_FROM_BROWSER: str = None
COOKIES_FILE: str = None

# ---- per-session caches (populated at runtime, never serialised) -----------

_MODEL_CACHE: dict = {}
_VAD_CACHE: dict = {}
