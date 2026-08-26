"""
Stage 3b — Voice Activity Detection + Tier Classification
============================================================
Silero VAD (via torch.hub) produces speech segments and total speech
duration, which drive the tier classifier and are reused by transcription.
"""

import torch

from . import config
from .timing import timed


def load_vad_model():
    """Load Silero VAD (once per session, cached)."""
    if "model" not in config._VAD_CACHE:
        print("[INFO] Loading Silero VAD ...")
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        config._VAD_CACHE["model"] = vad_model
        config._VAD_CACHE["utils"] = utils
    else:
        print("[INFO] Reusing cached Silero VAD")
    return config._VAD_CACHE["model"], config._VAD_CACHE["utils"]


@timed
def run_vad(audio_path: str) -> dict:
    """Run Silero VAD → speech segments + total speech duration."""
    from .audio import read_pcm_wav  # local import to avoid circular

    try:
        vad_model, utils = load_vad_model()
        (get_speech_timestamps, *_rest) = utils

        wav = read_pcm_wav(audio_path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)

        if not speech_timestamps:
            return {"status": "no_audio_track", "reason": "VAD found no speech in audio"}

        total_speech_sec = sum(
            (seg["end"] - seg["start"]) / 16000 for seg in speech_timestamps
        )
        return {
            "status": "ok",
            "speech_segments": speech_timestamps,
            "total_speech_sec": round(total_speech_sec, 2),
        }
    except Exception as e:
        return {"status": "transcription_failed", "reason": f"VAD failed: {e}"}


def classify_tier(total_speech_sec: float) -> dict:
    """Map total speech duration → tier name + model size."""
    if total_speech_sec < config.SHORT_MAX_SEC:
        tier = "short"
    elif total_speech_sec < config.MEDIUM_MAX_SEC:
        tier = "medium"
    else:
        tier = "long"
    return {"tier": tier, "model_size": config.TIER_MODEL_MAP[tier]}
