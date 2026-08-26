"""
Stage 4 — Transcription
========================
Load faster-whisper models (cached) and transcribe with word-level
timestamps.  The clip_timestamps bug fix lives here.
"""

from faster_whisper import WhisperModel

from . import config
from .timing import timed


def load_whisper_model(model_size: str) -> WhisperModel:
    """Load (or reuse) a faster-whisper model.  int8 on CPU, float16 on CUDA."""
    if model_size not in config._MODEL_CACHE:
        compute = "float16" if config.DEVICE == "cuda" else "int8"
        print(f"[INFO] Loading faster-whisper '{model_size}' on {config.DEVICE} ({compute}) ...")
        kwargs = {"cpu_threads": config.CPU_THREADS} if config.DEVICE == "cpu" else {}
        config._MODEL_CACHE[model_size] = WhisperModel(
            model_size, device=config.DEVICE, compute_type=compute, **kwargs
        )
    else:
        print(f"[INFO] Reusing cached faster-whisper '{model_size}'")
    return config._MODEL_CACHE[model_size]


@timed
def transcribe(audio_path: str, model_size: str = "tiny.en", speech_segments=None) -> dict:
    """
    Transcribe audio with word-level timestamps.

    When speech_segments (from Silero VAD) are provided they are passed as
    clip_timestamps so faster-whisper's internal VAD is skipped — VAD runs
    only once per pipeline invocation.

    BUG FIX: faster-whisper's clip_timestamps default is ``"0"`` (a string),
    not ``None``.  Passing ``None`` explicitly crashes ``generate_segments``
    with ``'NoneType' object is not iterable``.  We only pass it when we
    actually have VAD segments; otherwise the library default applies.
    """
    try:
        model = load_whisper_model(model_size)

        clip_timestamps = None
        if speech_segments:
            clip_timestamps = []
            for seg in speech_segments:
                clip_timestamps.extend([seg["start"] / 16000, seg["end"] / 16000])

        # Only include clip_timestamps when non-None to avoid the bug.
        transcribe_kwargs = dict(
            word_timestamps=True,
            vad_filter=(clip_timestamps is None),
            beam_size=5,
            condition_on_previous_text=False,
        )
        if clip_timestamps is not None:
            transcribe_kwargs["clip_timestamps"] = clip_timestamps

        segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

        # Consume the lazy generator with per-segment safety so one corrupt
        # segment is skipped instead of aborting the whole transcription.
        segments = []
        for seg in segments_gen:
            try:
                _ = seg.words  # materialise; surfaces any latent error early
                segments.append(seg)
            except Exception:
                continue

        if not segments:
            return {"status": "transcription_failed", "reason": "No speech detected in audio"}

        return {
            "status": "ok",
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
        }
    except Exception as e:
        return {"status": "transcription_failed", "reason": str(e)}
