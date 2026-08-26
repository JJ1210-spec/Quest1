"""
Stage 1 — Video Acquisition
============================
Download via yt-dlp (URL) or verify a local path exists.

Strategy:
  1. yt-dlp simple (proven Kaggle approach — works on most systems)
  2. curl --tls-max 1.2 fallback (works around TLS 1.3 issues on some
     Windows machines after August 2026 security updates)
  3. If all fail, report the error
"""

import os
import sys
import shutil
import subprocess

from . import config
from .timing import timed


def _is_valid_video(path: str) -> bool:
    """Check if a file looks like a real video (not an HTML error page)."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 10_000:
            return False
        with open(path, "rb") as f:
            header = f.read(16)
        if header[:1] in (b"<", b"!", b"\xef", b"\xff"):
            return False
        if b"ftyp" in header or header[:3] == b"\x1a\x45\xdf":
            return True
        return os.path.getsize(path) > 100_000
    except Exception:
        return False


def _run_ytdlp(args: list) -> subprocess.CompletedProcess:
    """Run yt-dlp via Python module."""
    return subprocess.run(
        [sys.executable, "-m", "yt_dlp"] + args,
        check=True, capture_output=True, text=True,
    )


def _find_curl() -> str | None:
    """Locate a usable curl binary on Windows."""
    system_curl = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "curl.exe")
    if os.path.isfile(system_curl):
        return system_curl
    return shutil.which("curl")


def _download_with_curl(url: str, dest: str) -> dict:
    """Fallback downloader using curl --tls-max 1.2."""
    curl = _find_curl()
    if not curl:
        return {"status": "download_failed", "reason": "curl not found on system"}

    cmd = [curl, "--tls-max", "1.2", "-k", "-L", "-o", dest, "-f", url]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return {"status": "ok", "path": dest}
        reason = result.stderr.strip() if result.stderr else "curl produced no output"
        return {"status": "download_failed", "reason": reason}
    except subprocess.TimeoutExpired:
        return {"status": "download_failed", "reason": "curl download timed out (600s)"}
    except Exception as e:
        return {"status": "download_failed", "reason": str(e)}


def _download_with_ytdlp(source: str, dest: str) -> dict:
    """Download using yt-dlp — simple approach matching proven Kaggle code."""
    fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    attempts = [{"label": "simple", "args": ["-f", fmt, "-o", dest, source]}]

    if config.COOKIES_FROM_BROWSER:
        attempts.append({
            "label": "with browser cookies",
            "args": ["--cookies-from-browser", config.COOKIES_FROM_BROWSER,
                     "-f", fmt, "-o", dest, source],
        })
    elif config.COOKIES_FILE:
        attempts.append({
            "label": "with cookies file",
            "args": ["--cookies", config.COOKIES_FILE,
                     "-f", fmt, "-o", dest, source],
        })

    errors = []
    for attempt in attempts:
        try:
            _run_ytdlp(attempt["args"])
            if os.path.exists(dest):
                return {"status": "ok", "path": dest}
            out_dir = os.path.dirname(dest)
            base = os.path.splitext(os.path.basename(dest))[0]
            for fname in os.listdir(out_dir):
                if fname.startswith(base):
                    return {"status": "ok", "path": os.path.join(out_dir, fname)}
            return {"status": "ok", "path": dest}
        except FileNotFoundError:
            errors.append("yt-dlp not found on PATH")
            break
        except subprocess.CalledProcessError as e:
            errors.append(f"yt-dlp ({attempt['label']}): {e.stderr.strip() or str(e)}")

    return {"status": "download_failed", "reason": "\n".join(errors) if errors else "yt-dlp not found"}


@timed
def acquire_video(source: str, dest: str) -> dict:
    """Download via yt-dlp (URL) or verify a local path exists."""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            print(f"[INFO] Downloading video from {source} ...")

            if os.path.isfile(dest) and not _is_valid_video(dest):
                print(f"[INFO] Removing stale file: {dest}")
                try:
                    os.remove(dest)
                except OSError:
                    pass

            # Attempt 1: yt-dlp (simple, proven Kaggle approach)
            result = _download_with_ytdlp(source, dest)
            if result["status"] == "ok" and _is_valid_video(result.get("path", dest)):
                return result

            # Attempt 2: curl --tls-max 1.2 (TLS 1.3 workaround)
            print("[INFO] yt-dlp failed, trying curl with TLS 1.2 ...")
            result = _download_with_curl(source, dest)
            if result["status"] == "ok" and _is_valid_video(result.get("path", dest)):
                return result

            return {"status": "download_failed", "reason": result.get("reason", "All download methods failed")}
        else:
            if not os.path.exists(source):
                return {"status": "download_failed", "reason": f"Local file not found: {source}"}
            return {"status": "ok", "path": source}
    except Exception as e:
        return {"status": "download_failed", "reason": str(e)}
