
# Dialogue Frame Finder

Find the **exact video frame** where a spoken dialogue line occurs — from any
URL supported by **yt-dlp** (YouTube, Vimeo, ok.ru, …) or a local video file.
Runs fully offline on **CPU only** — no GPU required.

Give it a video and a quote:

```bash
python dialogue_frame_finder_optimized.py \
    --source "https://ok.ru/video/248244667877" \
    --query   "My mind rebels at stagnation" \
    --outdir  ./output
```

… and it returns the matched frame, the timestamp, and a full JSON report:

```json
{
  "status": "success",
  "query": "My mind rebels at stagnation",
  "matched_text": "My mind rebels at stagnation.",
  "similarity_score": 100.0,
  "timestamp_sec": 325.3,
  "frame_number": 7799,
  "frame_image_path": "output/matched_frame.jpg"
}
```

## How it works

```
Video / URL
    │
    ▼
yt-dlp / local path ──► MP4
    │
    ▼
ffprobe ──────────────► FPS · duration · VFR flag
    │
    ▼
ffmpeg ───────────────► mono 16 kHz WAV
    │
    ▼
Silero VAD ───────────► speech segments · total speech seconds
    │
    ▼
Tier classifier ──────► short / medium / long
    │                       │
    │  short/medium:        └─ long: coarse pass (tiny.en, full audio)
    │  single pass              → candidate window (±45 s)
    │                           → fine pass (tiny.en, window only)
    ▼
Dialogue matcher (exact substring → rapidfuzz fuzzy fallback)
    │
    ▼
OpenCV frame seek ────► matched_frame.jpg  +  result.json
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
| `base.en` variant | 15.8–20.9 min | fuzzy 61.2 → correct frame |
| **`tiny.en` optimized (this repo default)** | **6.2 min** | **`success` — score 100.0, exact match** |

- Final timestamp landed **0.62 s** from the GPU ground truth
  (`large-v3` on Kaggle: 324.68 s vs 325.30 s here).
- On a short clip (23.6 s of speech): **32.9 s end-to-end** — ~2.9× faster
  than the original configuration.

## Requirements

- **Python 3.9+**
- **ffmpeg + ffprobe** — system binaries, not pip packages:

| OS | Install |
|----|---------|
| Windows | `winget install ffmpeg` or [ffmpeg.org](https://ffmpeg.org/download.html) (add to PATH) |
| macOS   | `brew install ffmpeg` |
| Ubuntu  | `sudo apt install ffmpeg` |

- Python packages:

```bash
pip install -r requirements.txt
```

Whisper models download automatically on first use and are cached
(`tiny.en` ≈ 75 MB).

## Usage

### URL source

```bash
python dialogue_frame_finder_optimized.py \
    --source "https://www.youtube.com/watch?v=..." \
    --query  "On your left" \
    --outdir ./output
```

### Local file

```bash
python dialogue_frame_finder_optimized.py \
    --source /path/to/movie.mp4 \
    --query  "Elementary, my dear Watson" \
    --outdir ./output
```

### Tuning knobs

```bash
# more coarse-pass accuracy for noisy audio
python dialogue_frame_finder_optimized.py \
    --source ./movie.mp4 --query "..." --outdir ./output \
    --coarse-model base.en --fine-model small.en

# cap CPU threads manually
python dialogue_frame_finder_optimized.py \
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
  "status": "success",
  "query": "My mind rebels at stagnation",
  "matched_text": "My mind rebels at stagnation.",
  "similarity_score": 100.0,
  "timestamp_sec": 325.3,
  "frame_number": 7799,
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
    "extract_audio": 3.93,
    "run_vad": 29.84,
    "transcribe_coarse": 335.57,
    "transcribe_fine": 3.31,
    "extract_frame": 0.14,
    "total": 373.09
  }
}
```

Statuses: `success` (exact match) · `partial_match` (fuzzy ≥ threshold) ·
`no_match` (line not found — `closest_text` included for debugging).

## Variants in this repo

| File | Models | Best for |
|---|---|---|
| `dialogue_frame_finder_optimized.py` | `tiny.en` end-to-end, beam 5 | **Default.** Fastest (6 min / hour of video), proven 100-score accuracy |
| `dialogue_frame_finder_base_en.py` | `base.en` coarse + `small.en` fine | Extra coarse-pass accuracy headroom for noisy audio |
| `dialogue_frame_finder.py` | `small` (CPU) / `large-v3` (GPU) | Original multilingual reference implementation |

All three share the identical pipeline and matching logic — they differ only
in model selection and inference configuration.

## Troubleshooting

- **`SSL: self-signed certificate in certificate chain`** when downloading
  from some sites (ok.ru etc.): your antivirus/proxy is intercepting HTTPS.
  Download once with `python -m yt_dlp --no-check-certificates -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best" -o video.mp4 <URL>`
  and run the pipeline on the local file.
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

This tool went through five iterations — Kaggle GPU baseline → error
handling → tiering + coarse-to-fine optimization → CLI packaging → CPU
inference optimization, with full benchmarks, design decisions, and bugs
found along the way. See **[approach.md](approach.md)**.

## License

MIT
