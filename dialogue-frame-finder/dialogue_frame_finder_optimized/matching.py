"""
Stage 5 — Dialogue Matching
=============================
Exact substring match first, rapidfuzz token_sort_ratio fuzzy fallback second.
Returns matched_text, similarity_score, and a word-level target_timestamp.
"""

from rapidfuzz import fuzz

from . import config
from .timing import timed


@timed
def match_dialogue(segments, target_text: str, threshold: float = None) -> dict:
    """
    Match *target_text* against transcribed segments.

    Pass 1 — exact substring match (score 100.0).
    Pass 2 — rapidfuzz ``token_sort_ratio`` fuzzy fallback.

    Returns a dict with ``status`` in
    ``{"success", "partial_match", "no_match"}``.
    """
    if threshold is None:
        threshold = config.MATCH_THRESHOLD

    target_norm = target_text.strip().lower()

    # --- Pass 1: exact substring match ---
    for seg in segments:
        seg_text_norm = (seg.text or "").strip().lower()
        if target_norm in seg_text_norm:
            target_words = target_norm.split()
            match_start_time = seg.start

            words = seg.words if seg.words is not None else []

            for i in range(len(words)):
                current_words = words[i:i + len(target_words)]

                if not all(
                    w is not None and getattr(w, "word", None) is not None
                    for w in current_words
                ):
                    continue

                window = " ".join(w.word.strip().lower() for w in current_words)
                if window.startswith(target_words[0]):
                    if getattr(words[i], "start", None) is not None:
                        match_start_time = words[i].start
                    break

            return {
                "status": "success",
                "matched_text": (seg.text or "").strip(),
                "similarity_score": 100.0,
                "target_timestamp": match_start_time,
            }

    # --- Pass 2: rapidfuzz fuzzy fallback ---
    best_score = -1.0
    best_seg = None
    for seg in segments:
        score = fuzz.token_sort_ratio(target_norm, (seg.text or "").strip().lower())
        if score > best_score:
            best_score = score
            best_seg = seg

    if best_seg is None:
        return {"status": "no_match", "reason": "No segments available to compare"}

    result = {
        "matched_text": (best_seg.text or "").strip(),
        "similarity_score": best_score,
        "target_timestamp": best_seg.start,
    }
    result["status"] = "partial_match" if best_score >= threshold else "no_match"
    return result
