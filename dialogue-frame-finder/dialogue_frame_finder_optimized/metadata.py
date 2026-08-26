"""
Stage 2 — Metadata Extraction
===============================
Extract fps, frame count, duration, and VFR flag via OpenCV + ffprobe.
"""

import json
import subprocess
import cv2

from .timing import timed


@timed
def get_video_metadata(video_path: str) -> dict:
    """Extract fps, frame count, duration, and VFR flag."""
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
