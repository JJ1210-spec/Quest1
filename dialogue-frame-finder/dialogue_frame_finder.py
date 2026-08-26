"""
dialogue_frame_finder.py
========================
Find the exact video frame that matches a spoken dialogue line.

Pipeline
--------
1. Download (yt-dlp) or load a local video
2. Extract metadata (fps, duration, VFR flag)
3. Extract mono 16 kHz WAV audio (ffmpeg)
4. Run Silero VAD  ->  speech segments + total speech duration
5. Classify audio length into a tier  ->  pick transcription strategy
   - short  (< 3 min speech)  : single pass with chosen model
   - medium (< 20 min speech) : single pass with chosen model
   - long   (>= 20 min speech): coarse pass (small) -> candidate window -> fine pass
6. Match the target dialogue (exact substring -> rapidfuzz fuzzy fallback)
7. Extract the frame at the matched timestamp (+ nearby frames) -> save as JPEG

   Frame extraction uses ffmpeg's own timestamp-based seeking rather than
   OpenCV's frame-count seeking. This is correct on variable-frame-rate (VFR)
   video (ffmpeg seeks by actual PTS, not an assumed constant fps), and is
   also typically much faster on containers where OpenCV's seek degrades
   (e.g. some WebM/VP9 downloads from yt-dlp).

Usage
-----
    python dialogue_frame_finder.py \\
        --source  "https://ok.ru/video/248244667877" \\
        --query   "My mind rebels at stagnation" \\
        --outdir  ./output

    # local file, custom model, CPU forced
    python dialogue_frame_finder.py \\
        --source  /path/to/movie.mp4 \\
        --query   "Elementary, my dear Watson" \\
        --outdir  ./output \\
        --model   small \\
        --device  cpu
"""

import os
import sys
import json
import time
import shutil
import argparse
import functools
import subprocess
import wave
import cv2
import torch
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the exact video frame for a spoken dialogue line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", required=True,
        help="URL (yt-dlp supported) or local path to the video file.",
    )
    parser.add_argument(
        "--query", required=True,
        help="The dialogue line to search for (partial or full sentence).",
    )
    parser.add_argument(
        "--outdir", default="./output",
        help="Directory where the matched frame JPEG(s) and JSON result are saved. "
             "Created automatically if it does not exist. (default: ./output)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the faster-whisper model for short/medium tier. "
             "E.g. tiny, base, small, medium, large-v3. "
             "Default is 'small' on CPU, 'large-v3' on GPU.",
    )
    parser.add_argument(
        "--coarse-model", default="small",
        help="Model used for the coarse pass on long-tier audio. (default: small)",
    )
    parser.add_argument(
        "--fine-model", default="small",
        help="Model used for the fine pass on the candidate window of long-tier audio. "
             "(default: small -- pass large-v3 for higher accuracy)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda' or 'cpu'. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=80.0,
        help="Rapidfuzz token_sort_ratio score (0-100) above which a fuzzy match is "
             "accepted. (default: 80.0)",
    )
    parser.add_argument(
        "--coarse-threshold", type=float, default=50.0,
        help="Match threshold for the coarse pass on long-tier audio. (default: 50.0)",
    )
    parser.add_argument(
        "--window-buffer", type=float, default=45.0,
        help="Seconds of padding on each side of the coarse timestamp when slicing "
             "the fine-pass audio window. (default: 45)",
    )
    parser.add_argument(
        "--frame-neighbors", type=int, default=2,
        help="Number of extra frames to extract on each side of the matched "
             "timestamp, as a hedge against seek/timing imprecision. "
             "0 extracts only the single matched frame. (default: 2)",
    )
    parser.add_argument(
        "--cookies-from-browser", default=None,
        metavar="BROWSER",
        help="Pass your browser cookies to yt-dlp to handle age-restricted or "
             "login-required videos. E.g. chrome, firefox, edge, brave. "
             "Example: --cookies-from-browser chrome",
    )
    parser.add_argument(
        "--cookies", default=None,
        metavar="FILE",
        help="Path to a Netscape-format cookies.txt file for yt-dlp. "
             "Alternative to --cookies-from-browser.",
    )
    return parser.parse_args()


# =============================================================================
# GLOBALS  (populated from args in main())
# =============================================================================

DEVICE: str = "cpu"
WORK_DIR: str = "./output"
VIDEO_PATH: str = ""
AUDIO_PATH: str = ""
AUDIO_SLICE_PATH: str = ""
FRAME_OUT_PATH: str = ""

MATCH_THRESHOLD: float = 80.0
COARSE_MATCH_THRESHOLD: float = 50.0
CANDIDATE_WINDOW_BUFFER_SEC: float = 45.0
FRAME_NEIGHBOR_COUNT: int = 2

SHORT_MAX_SEC: int = 180    # < 3 min  speech -> "short"  tier
MEDIUM_MAX_SEC: int = 1200  # < 20 min speech -> "medium" tier
                             # >= 20 min speech -> "long"   tier

TIER_MODEL_MAP: dict = {}
LONG_COARSE_MODEL: str = "small"
LONG_FINE_MODEL: str = "small"

COOKIES_FROM_BROWSER: str = None
COOKIES_FILE: str = None

# Per-session model caches
_MODEL_CACHE: dict = {}
_VAD_CACHE: dict = {}


# =============================================================================
# MODEL LOADERS
# =============================================================================

def load_whisper_model(model_size: str) -> WhisperModel:
    if model_size not in _MODEL_CACHE:
        # int8 on CPU (fast, low memory); float16 on CUDA (fast, accurate)
        compute = "float16" if DEVICE == "cuda" else "int8"
        print(f"[INFO] Loading faster-whisper '{model_size}' on {DEVICE} ({compute}) ...")
        _MODEL_CACHE[model_size] = WhisperModel(
            model_size, device=DEVICE, compute_type=compute
        )
    else:
        print(f"[INFO] Reusing cached faster-whisper '{model_size}'")
    return _MODEL_CACHE[model_size]


def load_vad_model():
    if "model" not in _VAD_CACHE:
        print("[INFO] Loading Silero VAD ...")
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _VAD_CACHE["model"] = vad_model
        _VAD_CACHE["utils"] = utils
    else:
        print("[INFO] Reusing cached Silero VAD")
    return _VAD_CACHE["model"], _VAD_CACHE["utils"]


# =============================================================================
# TIMING DECORATOR
# =============================================================================

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round(time.perf_counter() - start, 3)
        if isinstance(result, dict):
            result["_stage_duration_sec"] = elapsed
        return result
    return wrapper


# =============================================================================
# STAGE 1 -- VIDEO ACQUISITION
# =============================================================================

@timed
def acquire_video(source: str, dest: str) -> dict:
    """Download via yt-dlp (URL) or verify a local path exists.

    yt-dlp is called as ``python -m yt_dlp`` so it works even when the
    standalone ``yt-dlp`` binary is not on PATH (only the pip package is
    installed).  Falls back to the bare ``yt-dlp`` command as a secondary
    option for environments where the binary is on PATH but the module is not.

    If extracting browser cookies fails (for example, because Chrome has its
    cookie database locked), a public video is retried without cookies.  This
    prevents an optional authentication aid from blocking a normal download.

    A stale file at ``dest`` from a previous run is removed before downloading,
    so a new --source always produces a fresh download instead of yt-dlp
    silently skipping past an existing file.

    Format selector ``bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best``
    ensures YouTube (and most other sites) always return a playable file.
    """
    try:
        if source.startswith("http://") or source.startswith("https://"):
            print(f"[INFO] Downloading video from {source} ...")

            # Format selector: prefer mp4 video+audio merge; fall back to best available
            fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

            cookie_args = []
            if COOKIES_FROM_BROWSER:
                cookie_args.extend(["--cookies-from-browser", COOKIES_FROM_BROWSER])
            elif COOKIES_FILE:
                cookie_args.extend(["--cookies", COOKIES_FILE])

            # Chrome often locks its cookie database while it is running.  Cookies
            # are optional for public videos, so retry anonymously if extraction
            # fails rather than ending the pipeline before yt-dlp reaches YouTube.
            download_attempts = [(cookie_args, "with browser cookies")]
            if COOKIES_FROM_BROWSER:
                download_attempts.append(([], "without browser cookies"))

            errors = []
            cookie_copy_failed = False

            if os.path.exists(dest):
                os.remove(dest)  # avoid reusing a stale file from a previous run

            for attempt_index, (attempt_args, attempt_label) in enumerate(download_attempts):
                if attempt_index and not cookie_copy_failed:
                    break
                if attempt_index:
                    print(
                        "[WARN] Could not read browser cookies. Retrying "
                        "without cookies (works for public videos) ..."
                    )

                # Primary: python -m yt_dlp  (works when only the pip package is installed)
                # Secondary: bare yt-dlp     (works when the standalone binary is on PATH)
                for cmd_base in (
                    [sys.executable, "-m", "yt_dlp"],
                    ["yt-dlp"],
                ):
                    cmd = cmd_base + ["-f", fmt, "--force-overwrites"] + attempt_args + ["-o", dest, source]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        # yt-dlp may produce a file with a different extension when merging
                        # (e.g. .mkv).  Walk outdir to find what was actually written.
                        if os.path.exists(dest):
                            return {"status": "ok", "path": dest}
                        # Search for any file yt-dlp wrote (it may have chosen a different ext)
                        out_dir = os.path.dirname(dest)
                        base    = os.path.splitext(os.path.basename(dest))[0]
                        for fname in os.listdir(out_dir):
                            if fname.startswith(base):
                                actual = os.path.join(out_dir, fname)
                                return {"status": "ok", "path": actual}
                        return {"status": "ok", "path": dest}
                    except FileNotFoundError:
                        continue  # try the next command variant
                    except subprocess.CalledProcessError as e:
                        error_text = e.stderr or str(e)
                        errors.append(f"{attempt_label}: {error_text}")
                        if (
                            attempt_args
                            and COOKIES_FROM_BROWSER
                            and "cookie database" in error_text.lower()
                        ):
                            cookie_copy_failed = True

            if errors:
                return {"status": "download_failed", "reason": "\n".join(errors)}

            return {
                "status": "download_failed",
                "reason": "yt-dlp not found. Install it with: pip install yt-dlp",
            }
        else:
            if not os.path.exists(source):
                return {"status": "download_failed", "reason": f"Local file not found: {source}"}
            return {"status": "ok", "path": source}
    except subprocess.CalledProcessError as e:
        return {"status": "download_failed", "reason": e.stderr or str(e)}
    except Exception as e:
        return {"status": "download_failed", "reason": str(e)}


# =============================================================================
# STAGE 2 -- METADATA EXTRACTION
# =============================================================================

@timed
def get_video_metadata(video_path: str) -> dict:
    """Extract fps, frame count, duration, and VFR flag via OpenCV + ffprobe."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "metadata_failed", "reason": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        cap.release()

        if not fps or fps <= 0:
            return {"status": "metadata_failed", "reason": "Invalid or unreadable fps"}

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate",
             "-of", "json", video_path],
            capture_output=True, text=True, check=True,
        )
        probe_data = json.loads(probe.stdout)
        r_rate   = probe_data["streams"][0]["r_frame_rate"]
        avg_rate = probe_data["streams"][0]["avg_frame_rate"]
        is_vfr   = r_rate != avg_rate

        return {
            "status": "ok",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "is_vfr": is_vfr,
        }
    except Exception as e:
        return {"status": "metadata_failed", "reason": str(e)}


# =============================================================================
# STAGE 3 -- AUDIO EXTRACTION
# =============================================================================

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


# =============================================================================
# STAGE 3b -- VOICE ACTIVITY DETECTION
# =============================================================================

def read_pcm_wav(audio_path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """Read the PCM WAV created by ``extract_audio`` without torchaudio.

    Silero's bundled ``read_audio`` utility now routes through torchaudio's
    TorchCodec backend in recent releases.  That backend needs platform-specific
    FFmpeg DLLs and can fail even though the command-line ffmpeg used above is
    working.  Our input is always a mono 16-bit PCM WAV, so the standard library
    is sufficient and avoids that unrelated dependency chain.
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

    # clone() makes the tensor writable after constructing it from immutable bytes.
    return torch.frombuffer(raw_samples, dtype=torch.int16).clone().float() / 32768.0


@timed
def run_vad(audio_path: str) -> dict:
    """Run Silero VAD -> speech segments + total speech duration for tiering."""
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
    """Map total speech duration -> tier name + model size."""
    if total_speech_sec < SHORT_MAX_SEC:
        tier = "short"
    elif total_speech_sec < MEDIUM_MAX_SEC:
        tier = "medium"
    else:
        tier = "long"
    return {"tier": tier, "model_size": TIER_MODEL_MAP[tier]}


# =============================================================================
# STAGE 4 -- TRANSCRIPTION
# =============================================================================

@timed
def transcribe(audio_path: str, model_size: str = "small", speech_segments=None) -> dict:
    """
    Transcribe audio with word-level timestamps.

    When speech_segments (from Silero VAD) are provided they are passed as
    clip_timestamps so faster-whisper's internal VAD is skipped -- VAD runs
    only once per pipeline invocation.

    clip_timestamps format: flat list [s1, e1, s2, e2, ...] in seconds (floats).
    """
    try:
        model = load_whisper_model(model_size)

        clip_timestamps = None
        if speech_segments:
            clip_timestamps = []
            for seg in speech_segments:
                clip_timestamps.extend([seg["start"] / 16000, seg["end"] / 16000])

        segments_gen, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=(clip_timestamps is None),
            clip_timestamps=clip_timestamps,
        )

        # model.transcribe() is a lazy generator; errors (e.g. None word objects
        # from malformed audio) only surface during iteration. Iterate per-segment
        # so a single corrupt segment does not abort the whole transcription.
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


# =============================================================================
# STAGE 5 -- DIALOGUE MATCHING
# =============================================================================

@timed
def match_dialogue(segments, target_text: str, threshold: float = None) -> dict:
    """
    Exact substring match first, rapidfuzz token_sort_ratio fuzzy fallback second.
    Returns matched_text, similarity_score, target_timestamp (seconds).
    """
    if threshold is None:
        threshold = MATCH_THRESHOLD

    target_norm = target_text.strip().lower()

    # --- Pass 1: exact substring match ---
    for seg in segments:
        seg_text_norm = (seg.text or "").strip().lower()
        if target_norm in seg_text_norm:
            target_words = target_norm.split()
            match_start_time = seg.start  # segment start as safe fallback

            # seg.words can be None even when seg.text is populated
            words = seg.words if seg.words is not None else []

            for i in range(len(words)):
                current_words = words[i:i + len(target_words)]

                # Individual word objects can also be None in some faster-whisper builds
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


# =============================================================================
# STAGE 6 + 7 -- FRAME EXTRACTION  (Group C: ffmpeg PTS-based seeking)
# =============================================================================

@timed
def extract_frame(video_path: str, timestamp_sec: float, fps: float, out_path: str,
                   neighbor_count: int = 2) -> dict:
    """
    Extract the frame at timestamp_sec, plus `neighbor_count` frames on each
    side, using ffmpeg's own timestamp-based seeking.

    Why ffmpeg instead of OpenCV's CAP_PROP_POS_FRAMES:
      - ffmpeg seeks by actual PTS (presentation timestamp), which is correct
        on variable-frame-rate (VFR) video. OpenCV's frame-count seek assumes
        constant fps (frame_number = round(timestamp * fps)), which drifts on
        VFR video.
      - ffmpeg's seeking is also typically far faster on containers where
        OpenCV's non-keyframe seeking degrades badly (observed: 25s+ for a
        single frame on some yt-dlp-downloaded streams).

    Seek strategy: a coarse seek (-ss before -i, fast but imprecise) lands
    ~2s before the target, then a fine seek (-ss after -i, slower but frame
    accurate) covers the remaining gap. This is the standard ffmpeg
    fast+accurate seek pattern -- much cheaper than a fully-accurate seek
    from the start of the file, and much more precise than a coarse seek alone.

    `fps` is still used to compute an *approximate* frame_number for display/
    logging purposes only -- it is not used to locate the frame.
    """
    try:
        base_dir = os.path.dirname(out_path)
        os.makedirs(base_dir, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(out_path))
        frame_step = (1.0 / fps) if fps else 0.042  # ~24fps spacing as a safe fallback

        offsets = list(range(-neighbor_count, neighbor_count + 1))  # e.g. [-2,-1,0,1,2]
        nearby_frames = []

        for offset in offsets:
            t = max(0.0, timestamp_sec + offset * frame_step)
            suffix = "primary" if offset == 0 else f"offset{offset:+d}"
            frame_path = os.path.join(base_dir, f"{base_name}_{suffix}{ext}")

            coarse_seek = max(0.0, t - 2.0)
            fine_seek = t - coarse_seek

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(coarse_seek),
                "-i", video_path,
                "-ss", str(fine_seek),
                "-frames:v", "1",
                "-q:v", "2",
                frame_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(frame_path):
                nearby_frames.append({
                    "offset": offset,
                    "timestamp_sec": round(t, 3),
                    "path": frame_path,
                })

        if not nearby_frames:
            return {
                "status": "frame_extraction_failed",
                "reason": "ffmpeg could not extract any frame near the target timestamp",
            }

        # Copy the primary (offset 0) frame to the canonical out_path, or the
        # closest available offset if offset 0 somehow failed to extract.
        primary = next((f for f in nearby_frames if f["offset"] == 0), nearby_frames[len(nearby_frames) // 2])
        if primary["path"] != out_path:
            shutil.copyfile(primary["path"], out_path)

        approx_frame_number = round(timestamp_sec * fps) if fps else None

        return {
            "status": "ok",
            "frame_number": approx_frame_number,  # approximate; informational only
            "path": out_path,
            "nearby_frames": nearby_frames,
        }
    except Exception as e:
        return {"status": "frame_extraction_failed", "reason": str(e)}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(video_url_or_path: str, target_dialogue: str) -> dict:
    """
    Guard-clause chain -- short-circuits and returns on the first failure.
    Every stage wall-clock duration is collected in 'timings'.
    """
    pipeline_start = time.perf_counter()
    timings: dict = {}
    all_segments = []

    def record(stage_name: str, stage_result: dict):
        timings[stage_name] = stage_result.pop("_stage_duration_sec", None)

    # Stage 1: Video acquisition
    video_result = acquire_video(video_url_or_path, VIDEO_PATH)
    record("acquire_video", video_result)
    if video_result["status"] != "ok":
        return {**video_result, "timings": timings}
    video_path = video_result["path"]

    # Stage 2: Metadata
    metadata_result = get_video_metadata(video_path)
    record("get_video_metadata", metadata_result)
    if metadata_result["status"] != "ok":
        return {**metadata_result, "timings": timings}

    # Stage 3: Audio extraction
    audio_result = extract_audio(video_path, AUDIO_PATH)
    record("extract_audio", audio_result)
    if audio_result["status"] != "ok":
        return {**audio_result, "timings": timings}

    # Stage 3b: VAD
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

    # Stage 4 + 5: Transcription + matching
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
        all_segments = transcript_result["segments"]
        match_result = match_dialogue(transcript_result["segments"], target_dialogue)
        record("match_dialogue", match_result)

    else:
        # Long: coarse pass -> candidate window -> fine pass
        coarse_result = transcribe(
            audio_result["path"],
            model_size=LONG_COARSE_MODEL,
            speech_segments=vad_result["speech_segments"],
        )
        record("transcribe_coarse", coarse_result)
        if coarse_result["status"] != "ok":
            return {**coarse_result, "timings": timings}
        all_segments = coarse_result["segments"]
        coarse_match = match_dialogue(
            coarse_result["segments"], target_dialogue, threshold=COARSE_MATCH_THRESHOLD
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
                "video_metadata": {
                    "fps": metadata_result["fps"],
                    "duration_sec": metadata_result["duration_sec"],
                    "is_vfr": metadata_result["is_vfr"],
                },
                "transcript_debug": [
                    {"start": round(s.start, 2), "end": round(s.end, 2),
                     "text": (s.text or "").strip()}
                    for s in all_segments
                ],
                "timings": timings,
            }

        candidate_time = coarse_match["target_timestamp"]
        window_start = max(0.0, candidate_time - CANDIDATE_WINDOW_BUFFER_SEC)
        window_end   = min(
            metadata_result["duration_sec"],
            candidate_time + CANDIDATE_WINDOW_BUFFER_SEC,
        )

        slice_result = extract_audio_slice(
            audio_result["path"], window_start, window_end, AUDIO_SLICE_PATH
        )
        record("extract_audio_slice", slice_result)
        if slice_result["status"] != "ok":
            return {**slice_result, "timings": timings}

        fine_result = transcribe(
            slice_result["path"], model_size=LONG_FINE_MODEL, speech_segments=None
        )
        record("transcribe_fine", fine_result)

        # If fine pass fails (e.g. short/noisy slice), fall back to coarse result
        # rather than aborting -- the frame is still extracted at the coarse timestamp.
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

    # No match
    if match_result["status"] not in ("success", "partial_match"):
        timings["total"] = round(time.perf_counter() - pipeline_start, 3)
        return {
            "status": match_result["status"],
            "query": target_dialogue,
            "closest_text": match_result.get("matched_text"),
            "similarity_score": match_result.get("similarity_score"),
            "video_metadata": {
                "fps": metadata_result["fps"],
                "duration_sec": metadata_result["duration_sec"],
                "is_vfr": metadata_result["is_vfr"],
            },
            "transcript_debug": [
                {"start": round(s.start, 2), "end": round(s.end, 2),
                 "text": (s.text or "").strip()}
                for s in all_segments
            ],
            "reason": match_result.get("reason"),
            "timings": timings,
        }

    # Stage 6 + 7: Frame extraction (ffmpeg PTS-based, VFR-safe, + nearby frames)
    frame_result = extract_frame(
        video_path,
        match_result["target_timestamp"],
        metadata_result["fps"],
        FRAME_OUT_PATH,
        neighbor_count=FRAME_NEIGHBOR_COUNT,
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
        "frame_number_is_approximate": bool(metadata_result["is_vfr"]),
        "frame_image_path": frame_result["path"],
        "nearby_frames": frame_result.get("nearby_frames"),
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


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    global DEVICE, WORK_DIR, VIDEO_PATH, AUDIO_PATH, AUDIO_SLICE_PATH, FRAME_OUT_PATH
    global MATCH_THRESHOLD, COARSE_MATCH_THRESHOLD, CANDIDATE_WINDOW_BUFFER_SEC
    global TIER_MODEL_MAP, LONG_COARSE_MODEL, LONG_FINE_MODEL
    global COOKIES_FROM_BROWSER, COOKIES_FILE, FRAME_NEIGHBOR_COUNT

    args = parse_args()

    COOKIES_FROM_BROWSER = args.cookies_from_browser
    COOKIES_FILE = args.cookies

    # Device
    if args.device:
        DEVICE = args.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")

    # Default model -- small on CPU (friendly for local testing), large-v3 on GPU
    default_model = "small" if DEVICE == "cpu" else "large-v3"
    user_model = args.model or default_model

    TIER_MODEL_MAP = {
        "short":  user_model,
        "medium": user_model,
        "long":   user_model,  # reference only; long tier uses coarse/fine models below
    }
    LONG_COARSE_MODEL = args.coarse_model
    LONG_FINE_MODEL   = args.fine_model
    FRAME_NEIGHBOR_COUNT = max(0, args.frame_neighbors)

    # Paths
    WORK_DIR         = os.path.abspath(args.outdir)
    os.makedirs(WORK_DIR, exist_ok=True)
    VIDEO_PATH       = os.path.join(WORK_DIR, "input_video.mp4")
    AUDIO_PATH       = os.path.join(WORK_DIR, "audio.wav")
    AUDIO_SLICE_PATH = os.path.join(WORK_DIR, "audio_candidate_slice.wav")
    FRAME_OUT_PATH   = os.path.join(WORK_DIR, "matched_frame.jpg")

    # Thresholds
    MATCH_THRESHOLD             = args.match_threshold
    COARSE_MATCH_THRESHOLD      = args.coarse_threshold
    CANDIDATE_WINDOW_BUFFER_SEC = args.window_buffer

    # Run
    print(f"[INFO] Query  : {args.query}")
    print(f"[INFO] Source : {args.source}")
    print(f"[INFO] Output : {WORK_DIR}")
    print()

    result = run_pipeline(args.source, args.query)

    # Print result (filter None values for a clean JSON output)
    clean = {k: v for k, v in result.items() if v is not None}
    print()
    print(json.dumps(clean, indent=2))

    # Save result JSON alongside the frame
    result_json_path = os.path.join(WORK_DIR, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[INFO] Result saved  : {result_json_path}")

    if result.get("frame_image_path") and os.path.exists(result["frame_image_path"]):
        print(f"[INFO] Frame image   : {result['frame_image_path']}")
    if result.get("nearby_frames"):
        print(f"[INFO] Nearby frames : {len(result['nearby_frames'])} saved alongside it")

    return 0 if result.get("status") in ("success", "partial_match") else 1


if __name__ == "__main__":
    sys.exit(main())