# Quest 1 —Find the Exact Frame Where a Dialogue Appears in a media URL


Find the **exact video frame** where a spoken dialogue line occurs in any video.

This project started as a Kaggle notebook (`quest1-v3_final.ipynb`) and evolved into a
CPU-optimized modular Python tool.

## Where to look

| What | Where |
|------|-------|
| **Code + modular package** | [`dialogue-frame-finder/`](dialogue-frame-finder/) |
| How it was built | [`dialogue-frame-finder/approach.md`](dialogue-frame-finder/approach.md) |
| Prompts used | [`dialogue-frame-finder/prompts.txt`](dialogue-frame-finder/prompts.txt) |
| Run it | [`dialogue-frame-finder/README.md`](dialogue-frame-finder/README.md) |
| Where it all started | [`quest1-v3_final.ipynb`](quest1-v3_final.ipynb) |

## Quick start

```bash
git clone https://github.com/JJ1210-spec/Quest1.git
cd Quest1/dialogue-frame-finder
pip install -r requirements.txt
python -m dialogue_frame_finder_optimized \
    --source "https://ok.ru/video/248244667877" \
    --query "My mind rebels at stagnation" \
    --outdir ./output
