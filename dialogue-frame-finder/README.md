
# Dialogue Frame Finder

Find the **exact video frame** where a spoken dialogue line occurs — from any
URL supported by **yt-dlp** (YouTube, Vimeo, ok.ru, …) or a local video file.
Runs fully offline on **CPU only** — no GPU required.

Give it a video and a quote:

```bash
python -m dialogue_frame_finder_optimized \
    --source "https://ok.ru/videoembed/248244667877" \
    --query  "My mind rebels at stagnation" \
    --outdir ./output
```

… and it returns the matched frame, the timestamp, and a full JSON report:

```json
{
  "status": "partial_match",
  "query": "My mind rebels at stagnation",
  "matched_text": "My mind rebels, it's stagnation, give me problems, give me work, give me the most",
  "similarity_score": 78.9,
  "timestamp_sec": 324.77,
  "frame_number": 7787,
  "frame_image_path": "output/matched_frame.jpg"
}
```

## How it works

```
Video / URL
    │
    ▼
yt-dlp / curl / local path ──► MP4          (3-layer fallback: TLS12 wrapper → yt-dlp → curl)
    │
    ▼
ffprobe ─────────────────────► FPS · duration · VFR flag
    │
    ▼
ffmpeg ──────────────────────► mono 16 kHz WAV
    │
    ▼
Silero VAD ──────────────────► speech segments · total speech seconds
    │
    ▼
Tier classifier ─────────────► short / medium / long
    │                           │
    │  short/medium:            └─ long: coarse pass (tiny.en, full audio)
    │  single pass                  → candidate window (±45 s)
    │                               → fine pass (tiny.en, window only)
    ▼
Dialogue matcher (exact substring → rapidfuzz fuzzy fallback)
    │
    ▼
OpenCV frame seek ───────────► matched_frame.jpg  +  result.json
```

The two-pass design is the core idea: the **coarse pass** only has to locate
the right ±45 s window in the full audio, and the **fine pass** re-transcribes
just that window for word-level precision — so a small model delivers
large-model accuracy at a fraction of the cost.

## Performance

Measured on an **Intel i5-1245U laptop (CPU only, no GPU)** with a
54.4-minute video, query *"My mind rebels at stagnation"*:

| Configuration | Total runtime | Match result |
|---|---|---|
| original (`small`, beam 5) | est. 45–70 min | — |
| `base.en` variant | 15.8–20.9 min | fuzzy 61.2 → frame @ 326.15 s |
| **`tiny.en` modular (this repo default)** | **5.4 min** | **`partial_match` — score 78.9, frame @ 324.77 s** |

On a short clip (23.6 s of speech): **32.9 s end-to-end** — ~2.9× faster
than the original configuration.

### Modular package structure

```
dialogue_frame_finder_optimized/
├── __init__.py        Package metadata + version
├── __main__.py        Entry point for python -m
├── config.py          All shared state (device, paths, thresholds, model cache)
├── timing.py          @timed decorator for stage-level profiling
├── download.py        Video acquisition (yt-dlp TLS12 wrapper + curl fallback)
├── metadata.py        ffprobe + OpenCV metadata extraction
├── audio.py           Audio extraction + WAV reader for Silero VAD
├── vad.py             Silero VAD model + tier classification
├── transcribe.py      faster-whisper model loading + transcription
├── matching.py        Exact substring + rapidfuzz fuzzy matcher
├── frame.py           OpenCV frame seek + JPEG export
├── pipeline.py        Orchestrator — guard-clause chain across all stages
└── main.py            CLI arg parsing + config population + entry point
```

Each module owns one pipeline stage. `config.py` holds all shared state;
`main.py` writes to it once at startup; all other modules read from it.

## Requirements

- **Python 3.9+**
- **ffmpeg + ffprobe** — system binaries, not pip packages (Step 1 below)

---

## Installation (step by step)

### Step 1 — Install ffmpeg (system binary)

| OS | Install |
|----|---------|
| Windows | `winget install ffmpeg` — **then close and reopen the terminal** so PATH updates |
| macOS   | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

Verify both binaries are on PATH (ffmpeg ships ffprobe):

```bash
ffmpeg -version
ffprobe -version
```

### Step 2 — Get the code

```bash
git clone https://github.com/<your-username>/dialogue-frame-finder.git
cd dialogue-frame-finder
```

*(or download and extract the ZIP, then `cd` into it)*

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# activate it:
.venv\Scripts\activate        # Windows (cmd/PowerShell)
source .venv/bin/activate     # macOS / Linux
```

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `faster-whisper` (+ `ctranslate2`), `opencv-python-headless`,
`torch`, `yt-dlp`, `rapidfuzz`.

### Step 5 — Run it

```bash
python -m dialogue_frame_finder_optimized \
    --source "https://ok.ru/videoembed/248244667877" \
    --query  "My mind rebels at stagnation" \
    --outdir ./output
```

**First run:** downloads the `tiny.en` model (~75 MB) to the HuggingFace
cache automatically, then reuses it. Subsequent runs start instantly.

**Expected console output:**

```
[INFO] Using device: cpu (threads: 8)
[INFO] Tier: long  |  model: tiny.en  |  speech: 1209.69s
...
{
  "status": "partial_match",
  "matched_text": "My mind rebels, it's stagnation, give me problems, give me work, give me the most",
  "similarity_score": 78.9,
  "timestamp_sec": 324.77,
  "frame_number": 7787,
  ...
}
[INFO] Frame image   : ...\output\matched_frame.jpg
```

Results land in `--outdir`:

```
output/
├── matched_frame.jpg   ← the exact video frame
└── result.json         ← full structured result
```

---

## Usage examples

### URL source

```bash
python -m dialogue_frame_finder_optimized \
    --source "https://ok.ru/videoembed/248244667877" \
    --query  "My mind rebels at stagnation" \
    --outdir ./output
```

### Local file

```bash
python -m dialogue_frame_finder_optimized \
    --source /path/to/movie.mp4 \
    --query  "Elementary, my dear Watson" \
    --outdir ./output
```

### Tuning knobs

```bash
# more coarse-pass accuracy for noisy audio
python -m dialogue_frame_finder_optimized \
    --source ./movie.mp4 --query "..." --outdir ./output \
    --coarse-model base.en --fine-model small.en

# cap CPU threads manually
python -m dialogue_frame_finder_optimized \
    --source ./movie.mp4 --query "..." --outdir ./output --cpu-threads 6
```

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | *(required)* | URL (yt-dlp supported) or local file path |
| `--query`  | *(required)* | Dialogue line to search for |
| `--outdir` | `./output`  | Output directory (created if absent) |
| `--model`  | `tiny.en` (CPU) / `large-v3` (GPU) | Whisper model for short/medium tier |
| `--coarse-model` | `tiny.en` | Model for the long-tier coarse pass |
| `--fine-model`   | `tiny.en` | Model for the long-tier fine pass |
| `--device` | auto | `cpu` or `cuda` |
| `--cpu-threads` | `min(8, logical cores)` | CPU threads for faster-whisper |
| `--match-threshold` | `80.0` | Fuzzy match acceptance score (0–100) |
| `--coarse-threshold`| `50.0` | Fuzzy match score for the coarse pass |
| `--window-buffer`   | `45.0` | Seconds of padding around the coarse timestamp |
| `--cookies-from-browser` | None | Browser cookies for age/login-gated videos (`chrome`, `firefox`, `edge`) |
| `--cookies`              | None | Netscape-format cookies.txt file |

## Output

Both files land in `--outdir`:

```
output/
├── matched_frame.jpg   ← the exact video frame
└── result.json         ← full structured result
```

**`result.json` example** (real run, 54.4-min video):

```json
{
  "status": "partial_match",
  "query": "My mind rebels at stagnation. Give me problems. Give me work.",
  "matched_text": "My mind rebels, it's stagnation, give me problems, give me work, give me the most",
  "similarity_score": 78.87,
  "timestamp_sec": 324.77,
  "frame_number": 7787,
  "video_metadata": {
    "fps": 23.976,
    "duration_sec": 3261.74,
    "is_vfr": true
  },
  "tier_info": {
    "tier": "long",
    "model_size": "tiny.en",
    "total_speech_sec": 1209.69
  },
  "timings": {
    "acquire_video": 0.0,
    "get_video_metadata": 0.124,
    "extract_audio": 14.695,
    "run_vad": 105.132,
    "transcribe_coarse": 198.523,
    "match_dialogue_coarse": 0.005,
    "extract_audio_slice": 0.311,
    "transcribe_fine": 4.905,
    "match_dialogue_fine": 0.0,
    "extract_frame": 0.195,
    "total": 323.89
  }
}
```

Statuses: `success` (exact match) · `partial_match` (fuzzy ≥ threshold) ·
`no_match` (line not found — `closest_text` included for debugging).

## Variants in this repo

| Variant | Type | Models | Best for |
|---|---|---|---|
| `dialogue_frame_finder_optimized/` | **Modular package** | `tiny.en` end-to-end, beam 5 | **Default.** Fastest, clean architecture, curl fallback for TLS issues |
| `dialogue_frame_finder_optimized.py` | Single file | `tiny.en` end-to-end, beam 5 | Same logic as package, single-file convenience |
| `dialogue_frame_finder_base_en.py` | Single file | `base.en` coarse + `small.en` fine | Extra coarse-pass accuracy headroom for noisy audio |
| `dialogue_frame_finder.py` | Single file | `small` (CPU) / `large-v3` (GPU) | Original multilingual reference implementation |

All variants share the identical pipeline and matching logic — they differ only
in model selection and inference configuration.

## Troubleshooting

- **ok.ru / SSL connection reset (Error 10054)**: Recent Windows security
  updates changed TLS behavior. The modular package handles this automatically
  via a 3-layer download fallback (TLS 1.2 wrapper → yt-dlp direct → curl
  `--tls-max 1.2`). If all methods fail, download manually with
  `yt-dlp --no-check-certificates -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best" -o video.mp4 <URL>`
  and run on the local file.
- **`SSL: self-signed certificate in certificate chain`**: your antivirus/proxy
  is intercepting HTTPS. Use the local file path as above.
- **`WinError 1314`** on first model download: Windows symlink permission
  issue in the HuggingFace cache. Delete the partial
  `~/.cache/huggingface/hub/models--Systran--faster-whisper-<model>` folder
  and re-run (or enable Windows Developer Mode).
- **Non-English content**: use the multilingual original
  (`python dialogue_frame_finder.py --model small ...`) — the `.en` models are
  English-only.
- Run-to-run times can vary ±30 % on laptop CPUs (thermal throttling /
  background load) — that's hardware, not the pipeline.

## Development journey

This tool went through six iterations — Kaggle GPU baseline → error
handling → tiering + coarse-to-fine optimization → CLI packaging → CPU
inference optimization → modular architecture, with full benchmarks, design
decisions, and bugs found along the way. See **[approach.md](approach.md)**.

## License

MIT
