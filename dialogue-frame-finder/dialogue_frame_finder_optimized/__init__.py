"""
Dialogue Frame Finder — Optimized Edition (tiny.en, CPU)
========================================================
Find the exact video frame where a spoken dialogue line occurs.

Pipeline (unchanged from the original):
  Download → Metadata → Audio → VAD → Tier → Transcribe → Match → Frame

Optimizations over the original:
  * Default CPU model: tiny.en (English-only, ~3.7x faster than base.en)
  * CPU threads: min(8, os.cpu_count()) instead of CT2 default 4
  * beam_size=5 (free on tiny.en, recovers greedy accuracy loss)
  * condition_on_previous_text=False (prevents hallucination loops)
  * clip_timestamps bug fix (fine pass now actually works)

Usage:
  python -m dialogue_frame_finder_optimized \
      --source "https://ok.ru/videoembed/248244667877" \
      --query  "My mind rebels at stagnation" \
      --outdir ./output
"""

__version__ = "1.0.0"
