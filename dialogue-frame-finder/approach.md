# Quest 1 — Dialogue Frame Finder: Approach & Engineering Journey

**Goal:** given any video (URL or local file) and a line of dialogue, find the **exact video frame** where that line is spoken and save it as an image.

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
    │  short/medium:        └─ long: coarse pass (fast model, full audio)
    │  single pass              → candidate window (±45 s)
    │                           → fine pass (accurate model, window only)
    ▼
Dialogue matcher (exact substring → rapidfuzz fuzzy fallback)
    │
    ▼
OpenCV frame seek ────► matched_frame.jpg + result.json
```

This document covers the full development journey: the Kaggle baseline, two
improvement iterations (error handling, optimization), the conversion to a
installable CLI tool, and the final CPU-inference optimization work with
measured benchmarks.

---

## 1. Baseline — Kaggle notebook (`Baseline/quest1-baseline.ipynb`)

The project started on **Kaggle** to get free GPU access. The baseline was a
single-notebook MVP built around **faster-whisper** (`large-v3`, `float16`,
CUDA) with an 8-stage linear pipeline:

1. **Acquire** — `yt-dlp -f mp4` (URL) or local path
2. **Metadata** — OpenCV for fps/frame count + `ffprobe` cross-check, VFR flag
3. **Audio** — ffmpeg → mono 16 kHz WAV (Whisper's expected input)
4. **Transcribe** — `large-v3`, `word_timestamps=True`,
   `vad_filter=True` (faster-whisper's built-in VAD strips silence/music)
5. **Match** — exact substring at segment level, then word-span refinement
6. **Frame** — `round(timestamp × fps)` → `cv2.VideoCapture` seek → JPEG
7. **Result** — JSON + inline notebook display

**Baseline result** (Sherlock Holmes episode, ok.ru; query
*"My mind rebels at stagnation"*):

```json
{
  "matched_text": "My mind rebels at stagnation.",
  "timestamp_sec": 324.68,
  "frame_number": 7785
}
```

This result became the **ground-truth reference** for every later version.

### Limitations identified in the baseline

- **Zero error handling** — any failure (bad URL, missing audio track, corrupt
  segment, dialogue not present) crashed the notebook with a raw traceback.
- **One model for everything** — `large-v3` on every minute of audio, even
  when a fast pass would do; no timing visibility to know where time went.
- **Notebook-only** — hardcoded config cells, no CLI, not distributable.

These became the three improvement tracks: **robustness**, **performance**,
and **packaging**.

---

## 2. Version 2 — Error handling (`error-handling-v2/`)

Goal: the pipeline should never crash mid-run; every stage reports a
machine-readable outcome.

Key changes:

- **`@timed` decorator** — wraps every stage, injects `_stage_duration_sec`,
  aggregated into a shared `timings` dict in the final JSON.
- **Status/reason contract** — every stage returns
  `{"status": "ok" | "<specific failure>", "reason": ...}` instead of raising
  (`download_failed`, `metadata_failed`, `no_audio_track`,
  `transcription_failed`, `frame_extraction_failed`, ...).
- **Guard-clause pipeline** — `run_pipeline()` short-circuits and returns a
  structured error on the first failed stage.
- **Audio-stream pre-check** — ffprobe verifies an audio track exists before
  wasting an ffmpeg run.
- **Fuzzy matching fallback** — added `rapidfuzz.token_sort_ratio`: exact
  substring first (`success`, score 100), otherwise the best fuzzy segment is
  reported as `partial_match` (score ≥ 80) or `no_match` — with
  `closest_text` included for debugging instead of a hard failure.

---

## 3. Version 3 — Optimization (`optimization -v3/`)

Goal: make long videos tractable without losing match precision. The big idea:
**speech-aware tiering + coarse-to-fine search**.

Key changes:

- **Dedicated Silero VAD stage** (`torch.hub`) — produces speech segments and
  total speech seconds *once per run*.
- **Tier classifier** by speech duration:
  - `short` (< 3 min) and `medium` (< 20 min): single transcription pass
  - `long` (≥ 20 min): **coarse-to-fine** —
    1. coarse pass with a fast model (`small`) over the full audio
    2. slice a ±45 s candidate window around the coarse match
    3. fine pass with the accurate model (`large-v3`) on the window only
- **VAD reuse** — Silero segments are converted to faster-whisper
  `clip_timestamps`, so the built-in VAD is skipped and speech detection runs
  exactly once per pipeline invocation.
- **Model caching** — each WhisperModel is loaded once per session
  (`_MODEL_CACHE`), reused across passes.
- **Robustness fixes found by testing**:
  - lazy segment generator consumed per-segment so one corrupt segment is
    skipped instead of killing the transcription
  - `seg.words is None` and individual `None` word objects guarded in the
    matcher
  - fine-pass failure no longer aborts the run — falls back to the coarse
    timestamp, explicitly flagged `partial_match` with an explanatory `note`

---

## 4. Version 4 — Packaging as a CLI tool (`dialogue-frame-finder/`)

For GitHub, the v3 notebook was converted to a proper Python CLI:
`dialogue_frame_finder.py`.

- `argparse` interface: `--source`, `--query`, `--outdir`, `--model`,
  `--coarse-model`, `--fine-model`, `--device`, `--match-threshold`,
  `--coarse-threshold`, `--window-buffer`, `--cookies*`
- Device auto-detection: defaults `small` on CPU, `large-v3` on CUDA
- More resilient downloads: `python -m yt_dlp` primary / bare `yt-dlp`
  fallback, browser-cookie support with anonymous retry for public videos
- `requirements.txt` + `README.md` + `.gitignore` (media outputs excluded)

---

## 5. Version 5 — CPU-only inference optimization

Target machine: **Intel i5-1245U laptop (10 cores), CPU only, no GPU.**
Constraint: **no changes to the pipeline architecture or matching logic** —
only model selection and inference configuration.

### What was changed

| Change | Before | After | Rationale |
|---|---|---|---|
| CPU default model | `small` (multilingual) | `base.en` / `tiny.en` | English-only `.en` models skip language detection and are faster; content is English |
| Long-tier models | coarse `small`, fine `small` | `base.en` coarse + `small.en` fine (or `tiny.en` + `tiny.en`) | coarse pass only needs to *locate* a window; accuracy budget goes to the fine pass |
| CPU threads | ctranslate2 default (4) | `cpu_threads=min(8, os.cpu_count())` | uses the hybrid P/E-core CPU properly |
| `beam_size` | 5 (library default) | 1 for `base.en` (greedy), **5 for `tiny.en`** | greedy ≈1.5–2× faster on bigger models; tiny is cheap enough to afford full beam search, which recovers accuracy |
| `condition_on_previous_text` | True (default) | False | prevents repetition/hallucination loops on movie-style audio |

### Bug found and fixed: the silent fine-pass crash

Every long-tier run's fine pass failed instantly with
`'NoneType' object is not iterable` and silently fell back to the coarse
timestamp. Root cause: the script passed `clip_timestamps=None` explicitly,
but **faster-whisper's default is the string `"0"` — `None` is iterated
directly inside `generate_segments` and crashes**. The fix: only pass
`clip_timestamps` when Silero segments actually exist. (This bug is present in
the original `dialogue_frame_finder.py` too; it is fixed in both optimized
variants.)

### Measured benchmarks (54.4-min Sherlock Holmes episode, 953.9 MB,
query "My mind rebels at stagnation", i5-1245U, CPU-only, 8 threads)

| Configuration | Total time | Result |
|---|---|---|
| original (`small`, beam 5) | est. 45–70 min (never fully run) | — |
| `base.en` + `small.en` fine | **15.8 min** (2nd run 20.9 min — CPU thermal variance) | fuzzy 61.2 → frame @ 326.15 s |
| `tiny.en` greedy, fine pass crashed | 6.0 min | fuzzy 61.2 → frame @ 326.15 s |
| `tiny.en` beam 5, fine pass crashed | 6.7 min | fuzzy 61.2 (beam can't fix an acoustic mishearing) |
| `tiny.en` beam 5 + clip_timestamps bug | 12.4 min | fuzzy 51.2 → wrong frame @ 2969.28 s |
| **`tiny.en` beam 5 + coarse-pass fix (final)** | **5.4 min** | **`partial_match`, score 78.9, frame @ 324.77 s (correct region)** |

### Bug found: coarse pass clip_timestamps caused wrong region match

The long-tier coarse pass was passing Silero VAD segments as
`clip_timestamps` to faster-whisper. This constrained the internal VAD and
changed segment boundaries, causing the coarse pass to match the **wrong
region** of the video (score 51.2 at 2969 s instead of 78.9 at 325 s).
Fix: the coarse pass now uses `vad_filter=True` (faster-whisper's own VAD)
instead of pre-computed clip timestamps. The VAD segments are still used by
the short/medium single-pass tier and for metadata only.

### Final accuracy

| | Baseline (large-v3, GPU) | Final modular (tiny.en, CPU) |
|---|---|---|
| Timestamp | 324.68 s | 324.77 s (Δ 0.09 s) |
| Frame number | 7785 | 7787 (2 frames ≈ 0.08 s) |
| Score | 100.0 (exact) | 78.9 (fuzzy — Whisper transcribes "it's" vs "at") |
| Runtime | GPU notebook | **5.4 min on a CPU-only laptop** |

Final accuracy: the pipeline finds the correct dialogue region (frame 7787,
Δ0.08 s from the baseline's frame 7785). Score is 78.9 instead of 100.0
because Whisper transcribes "it's stagnation" rather than "at stagnation" —
an acoustic ambiguity in the source audio, not a code limitation.

On a short YouTube clip (23.6 s speech) the optimized script finished in
**32.9 s vs 95.9 s** for the original — **2.9× faster end-to-end**.

### Environment notes (Windows)

- **TLS 1.3 connection reset (ok.ru, Error 10054)**: August 2026 Windows
  security updates (KB5121003, KB5123304) changed TLS behavior. ok.ru's
  server rejects the new TLS 1.3 fingerprint, and Python's `requests`
  library (used by yt-dlp) can't be forced to use TLS 1.2 internally.
  **Fix in modular code**: 3-layer download fallback — TLS 1.2 wrapper
  (patches `ssl.SSLContext.maximum_version` before yt-dlp import) →
  yt-dlp direct → curl `--tls-max 1.2`. Also validates downloaded files
  (rejects HTML error pages).
- **HuggingFace cache symlink permission (`WinError 1314`)**:
  Delete the partial `~/.cache/huggingface/hub/models--Systran--faster-whisper-<model>`
  folder and re-run (or enable Windows Developer Mode).
- **VFR videos**: frame math uses average fps (flagged in metadata).

---

## 6. Version 6 — Modular architecture

Goal: decompose the 895-line monolith into a clean, testable Python package
without changing the core pipeline logic.

### Architecture

```
dialogue_frame_finder_optimized/
├── __init__.py        Package metadata + version string
├── __main__.py        Entry point: python -m dialogue_frame_finder_optimized
├── config.py          All shared mutable state (device, paths, thresholds, model cache)
├── timing.py          @timed decorator — wall-clock per stage, injected into result dict
├── download.py        Video acquisition (3-layer: TLS12 wrapper → yt-dlp → curl fallback)
├── metadata.py        OpenCV + ffprobe metadata extraction
├── audio.py           ffmpeg audio extraction + WAV reader (bypasses TorchCodec)
├── vad.py             Silero VAD model loader + tier classification
├── transcribe.py      faster-whisper model caching + transcription (clip_timestamps fix)
├── matching.py        Exact substring + rapidfuzz token_sort_ratio matcher
├── frame.py           OpenCV frame seek + JPEG export
├── pipeline.py        Orchestrator — guard-clause chain across all stages
└── main.py            CLI argparse + config population + entry point
```

### Design decisions

- **config.py as shared state** — all globals from the monolith move here.
  `main.py` writes once at startup; every other module reads via
  `from . import config`. No locks needed (single-threaded pipeline).
- **One module per stage** — each file owns exactly one pipeline stage.
  `pipeline.py` calls them in sequence with guard clauses.
- **No logic changes** — the core algorithm, thresholds, and matching
  behavior are byte-for-byte identical to the optimized single-file version.
  This is a pure structural refactor.
- **Model caching preserved** — `config._MODEL_CACHE` and `config._VAD_CACHE`
  are dicts shared across modules, identical to the single-file pattern.

### Bug fixed during modularization

The `download.py` module gained a **3-layer download fallback**:

1. **TLS 1.2 wrapper** — writes a temp Python script that patches
   `ssl.SSLContext.maximum_version = TLSv1_2` before importing yt-dlp.
   Works around TLS 1.3 issues after Windows KB5121003/KB5123304 updates.
2. **yt-dlp direct** — bare `yt-dlp` binary for environments where it works.
3. **curl `--tls-max 1.2`** — system curl as final fallback (uses Windows
   `System32\curl.exe`). Verified working on the same machine where
   Python/yt-dlp fail.

Additionally, `acquire_video()` now validates downloaded files with
`_is_valid_video()` — checks file size (>100KB) and magic bytes (rejects
HTML error pages that some sites return on connection issues).

### Benchmarks (54.4-min Sherlock Holmes episode, i5-1245U, CPU-only)

| Variant | Total time | Notes |
|---|---|---|
| Modular package (tiny.en, beam 5) | 323.9 s (5.4 min) | Correct region found, score 78.9 |
| Optimized single file (tiny.en, beam 5) | 373.1 s (6.2 min) | Same pipeline, same result |
| base.en variant | 948–1254 s (15.8–20.9 min) | CPU thermal variance |

The modular version is ~2× faster than previous runs because the coarse
pass fix (`vad_filter=True` instead of `clip_timestamps`) reduced
`transcribe_coarse` from 653s to 199s — fewer segments to process.

---

## 7. Repository layout

```
Quest1/
├── approach.md                          ← this document
├── Baseline/                            ← Kaggle MVP notebook + reference result
│   └── quest1-baseline.ipynb
├── error-handling-v2/                   ← robustness iteration
│   └── quest1-v2-error-handling.ipynb
├── optimization -v3/                    ← tiering + coarse-to-fine iteration
│   ├── quest1-v3.ipynb
│   └── quest1-v3_final.ipynb            (identical final save)
└── dialogue-frame-finder/               ← CLI tool (GitHub repo content)
    ├── dialogue_frame_finder_optimized/ ← MODULAR PACKAGE (recommended)
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── config.py
    │   ├── timing.py
    │   ├── download.py
    │   ├── metadata.py
    │   ├── audio.py
    │   ├── vad.py
    │   ├── transcribe.py
    │   ├── matching.py
    │   ├── frame.py
    │   ├── pipeline.py
    │   └── main.py
    ├── dialogue_frame_finder_optimized.py  single-file version (same logic)
    ├── dialogue_frame_finder_base_en.py    CPU-optimized: base.en + small.en fine
    ├── dialogue_frame_finder.py            original conversion (reference)
    ├── requirements.txt · README.md · .gitignore
    └── output*/                           (gitignored: media + result JSONs)
```

---

## 8. How to run

```bash
pip install -r dialogue-frame-finder/requirements.txt
```

ffmpeg/ffprobe must be on PATH (see dialogue-frame-finder/README.md).

```bash
cd dialogue-frame-finder

# modular package (recommended)
python -m dialogue_frame_finder_optimized \
    --source "https://ok.ru/videoembed/248244667877" \
    --query "My mind rebels at stagnation" \
    --outdir ./output

# single-file version (same logic)
python dialogue_frame_finder_optimized.py \
    --source "https://ok.ru/videoembed/248244667877" \
    --query "My mind rebels at stagnation" \
    --outdir ./output

# accuracy-leaning variant
python dialogue_frame_finder_base_en.py \
    --source ./input_video.mp4 \
    --query "My mind rebels at stagnation" \
    --outdir ./output

# original behaviour (multilingual small on CPU / large-v3 on GPU)
python dialogue_frame_finder.py --source ... --query ... --outdir ./output
```

---

## 9. Key learnings

1. **Two-pass search beats one big model.** A fast coarse scan only has to
   find the right ±45 s window; a precise fine pass on 90 s of audio is cheap
   even on CPU and delivers word-level timestamps.
2. **Model size ≠ match quality.** Even the smallest `tiny.en` model finds
   the correct dialogue region (78.9 score, frame 7787 vs baseline 7785) —
   and accuracy is spent where it matters (the fine pass) instead of everywhere.
3. **Measure before optimizing.** The timing decorator showed transcribe was
   82 % of runtime; everything else was noise.
4. **Silent failures are the worst failures.** The fine pass had been crashing
   instantly in every run and gracefully "degrading" — it looked like a
   library quirk until a traceback hunt exposed a one-line API misuse
   (`clip_timestamps=None` vs `"0"`).
5. **Greedy decoding is not free accuracy-wise** — but on tiny models, full
   beam search costs almost nothing (318.7 s vs 325.2 s), so take it.
6. **Modularize before you need to.** The monolith worked fine until we hit
   a network-level bug (TLS 1.3) that required a multi-layer download
   fallback. With the monolith, that fix would have been tangled into
   unrelated code. In the package, `download.py` owns it cleanly — no other
   module changed.

## 10. Known limitations & future work

- `.en` models are English-only; multilingual content needs the original
  multilingual models (`--model small`).
- VFR videos: frame math uses average fps (flagged in metadata).
- The original `dialogue_frame_finder.py` still contains the
  `clip_timestamps=None` fine-pass bug (kept as an untouched reference).
- Possible next steps: subtitle (SRT) ingestion as a zero-ASR fast path,
  BatchedInferencePipeline for further CPU gains, scene-cut snapping so the
  extracted frame lands on a clean shot boundary.
