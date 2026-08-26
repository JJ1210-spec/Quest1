"""
Pipeline Orchestrator
=====================
Guard-clause chain — short-circuits on the first stage failure.
Every stage's wall-clock duration is collected in ``timings``.
"""

import time

from . import config
from .download import acquire_video
from .metadata import get_video_metadata
from .audio import extract_audio, extract_audio_slice
from .vad import run_vad, classify_tier
from .transcribe import transcribe
from .matching import match_dialogue
from .frame import extract_frame


def run_pipeline(video_url_or_path: str, target_dialogue: str) -> dict:
    """Run the full 7-stage pipeline end to end."""
    pipeline_start = time.perf_counter()
    timings: dict = {}

    def record(stage_name: str, stage_result: dict):
        timings[stage_name] = stage_result.pop("_stage_duration_sec", None)

    # --- Stage 1: Video acquisition ---
    video_result = acquire_video(video_url_or_path, config.VIDEO_PATH)
    record("acquire_video", video_result)
    if video_result["status"] != "ok":
        return {**video_result, "timings": timings}
    video_path = video_result["path"]

    # --- Stage 2: Metadata ---
    metadata_result = get_video_metadata(video_path)
    record("get_video_metadata", metadata_result)
    if metadata_result["status"] != "ok":
        return {**metadata_result, "timings": timings}

    # --- Stage 3: Audio extraction ---
    audio_result = extract_audio(video_path, config.AUDIO_PATH)
    record("extract_audio", audio_result)
    if audio_result["status"] != "ok":
        return {**audio_result, "timings": timings}

    # --- Stage 3b: VAD ---
    vad_result = run_vad(audio_result["path"])
    record("run_vad", vad_result)
    if vad_result["status"] != "ok":
        return {**vad_result, "timings": timings}

    tier_info = classify_tier(vad_result["total_speech_sec"])
    print(
        f"[INFO] Tier: {tier_info['tier']}  |  "
        f"model: {tier_info['model_size']}  |  "
        f"speech: {vad_result['total_speech_sec']}s"
    )

    # --- Stage 4+5: Transcription + matching ---
    if tier_info["tier"] != "long":
        # Short / medium: single transcription pass
        transcript_result = transcribe(
            audio_result["path"],
            model_size=tier_info["model_size"],
            speech_segments=vad_result["speech_segments"],
        )
        record("transcribe", transcript_result)
        if transcript_result["status"] != "ok":
            return {**transcript_result, "timings": timings}

        match_result = match_dialogue(transcript_result["segments"], target_dialogue)
        record("match_dialogue", match_result)

    else:
        # Long: coarse pass → candidate window → fine pass
        # NOTE: coarse pass uses vad_filter=True (not clip_timestamps from
        # Silero VAD).  Constraining faster-whisper to pre-computed VAD
        # windows changes segment boundaries and can cause the coarse pass
        # to miss the target dialogue entirely.
        coarse_result = transcribe(
            audio_result["path"],
            model_size=config.LONG_COARSE_MODEL,
            speech_segments=None,
        )
        record("transcribe_coarse", coarse_result)
        if coarse_result["status"] != "ok":
            return {**coarse_result, "timings": timings}

        coarse_match = match_dialogue(
            coarse_result["segments"], target_dialogue,
            threshold=config.COARSE_MATCH_THRESHOLD,
        )
        record("match_dialogue_coarse", coarse_match)

        if coarse_match["status"] not in ("success", "partial_match"):
            timings["total"] = round(time.perf_counter() - pipeline_start, 3)
            return {
                "status": coarse_match["status"],
                "query": target_dialogue,
                "closest_text": coarse_match.get("matched_text"),
                "similarity_score": coarse_match.get("similarity_score"),
                "reason": coarse_match.get("reason", "Coarse pass found no usable candidate"),
                "tier_info": tier_info,
                "timings": timings,
            }

        candidate_time = coarse_match["target_timestamp"]
        window_start = max(0.0, candidate_time - config.CANDIDATE_WINDOW_BUFFER_SEC)
        window_end   = min(
            metadata_result["duration_sec"],
            candidate_time + config.CANDIDATE_WINDOW_BUFFER_SEC,
        )

        slice_result = extract_audio_slice(
            audio_result["path"], window_start, window_end, config.AUDIO_SLICE_PATH
        )
        record("extract_audio_slice", slice_result)
        if slice_result["status"] != "ok":
            return {**slice_result, "timings": timings}

        fine_result = transcribe(
            slice_result["path"], model_size=config.LONG_FINE_MODEL, speech_segments=None
        )
        record("transcribe_fine", fine_result)

        if fine_result["status"] != "ok":
            print(
                f"[WARN] Fine-pass failed ({fine_result.get('reason')}); "
                "falling back to coarse-pass timestamp."
            )
            match_result = coarse_match
            match_result["status"] = "partial_match"
            match_result["note"] = (
                f"Fine-pass failed: {fine_result.get('reason')}. "
                "Using coarse-pass timestamp (lower confidence)."
            )
        else:
            fine_match = match_dialogue(fine_result["segments"], target_dialogue)
            record("match_dialogue_fine", fine_match)

            if fine_match["status"] not in ("success", "partial_match"):
                match_result = coarse_match
                match_result["status"] = "partial_match"
                match_result["note"] = (
                    "Fine-pass verification failed; "
                    "using coarse-pass estimate (lower confidence)."
                )
            else:
                fine_match["target_timestamp"] += slice_result["window_start_sec"]
                match_result = fine_match

    # --- No match ---
    if match_result["status"] not in ("success", "partial_match"):
        timings["total"] = round(time.perf_counter() - pipeline_start, 3)
        return {
            "status": match_result["status"],
            "query": target_dialogue,
            "closest_text": match_result.get("matched_text"),
            "similarity_score": match_result.get("similarity_score"),
            "reason": match_result.get("reason"),
            "timings": timings,
        }

    # --- Stage 6+7: Frame extraction ---
    frame_result = extract_frame(
        video_path,
        match_result["target_timestamp"],
        metadata_result["fps"],
        config.FRAME_OUT_PATH,
    )
    record("extract_frame", frame_result)
    if frame_result["status"] != "ok":
        return {**frame_result, "timings": timings}

    timings["total"] = round(time.perf_counter() - pipeline_start, 3)

    return {
        "status": match_result["status"],
        "query": target_dialogue,
        "matched_text": match_result["matched_text"],
        "similarity_score": match_result["similarity_score"],
        "timestamp_sec": round(match_result["target_timestamp"], 3),
        "note": match_result.get("note"),
        "frame_number": frame_result["frame_number"],
        "frame_image_path": frame_result["path"],
        "video_metadata": {
            "fps": metadata_result["fps"],
            "duration_sec": metadata_result["duration_sec"],
            "is_vfr": metadata_result["is_vfr"],
        },
        "tier_info": {
            "tier": tier_info["tier"],
            "model_size": tier_info["model_size"],
            "total_speech_sec": vad_result["total_speech_sec"],
        },
        "timings": timings,
    }
