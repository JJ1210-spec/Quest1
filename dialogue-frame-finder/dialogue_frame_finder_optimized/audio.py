"""
Stage 3 — Audio Extraction
============================
Extract mono 16 kHz WAV audio, and a short slice for the fine pass.
Also contains the WAV reader used by Silero VAD.
"""

import json
import subprocess
import wave

import torch

from .timing import timed


@timed
def extract_audio(video_path: str, audio_path: str) -> dict:
    """Verify an audio stream exists, then extract mono 16 kHz WAV."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "json", video_path],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            return {"status": "no_audio_track", "reason": "No audio stream found in video"}

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-vn", audio_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "ok", "path": audio_path}
    except subprocess.CalledProcessError as e:
        return {"status": "audio_extraction_failed", "reason": e.stderr or str(e)}
    except Exception as e:
        return {"status": "audio_extraction_failed", "reason": str(e)}


@timed
def extract_audio_slice(audio_path: str, start_sec: float, end_sec: float, out_path: str) -> dict:
    """Extract a short audio window around the coarse candidate timestamp."""
    try:
        start_sec = max(0.0, start_sec)
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start_sec), "-to", str(end_sec),
            "-ar", "16000", "-ac", "1", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "ok", "path": out_path, "window_start_sec": start_sec}
    except subprocess.CalledProcessError as e:
        return {"status": "audio_extraction_failed", "reason": e.stderr or str(e)}
    except Exception as e:
        return {"status": "audio_extraction_failed", "reason": str(e)}


def read_pcm_wav(audio_path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """Read a mono 16-bit PCM WAV into a float32 tensor (for Silero VAD).

    Bypasses torchaudio's TorchCodec backend (which needs platform-specific
    FFmpeg DLLs that may not be present) by reading with the standard library.
    """
    with wave.open(audio_path, "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Expected mono WAV audio")
        if wav_file.getframerate() != sampling_rate:
            raise ValueError(
                f"Expected {sampling_rate} Hz WAV audio, got {wav_file.getframerate()} Hz"
            )
        if wav_file.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV audio")
        raw_samples = wav_file.readframes(wav_file.getnframes())

    return torch.frombuffer(raw_samples, dtype=torch.int16).clone().float() / 32768.0
