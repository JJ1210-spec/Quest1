"""
Stage 6+7 — Frame Extraction
==============================
Seek to a timestamp, read the frame, write it as JPEG.
"""

import cv2

from .timing import timed


@timed
def extract_frame(video_path: str, timestamp_sec: float, fps: float, out_path: str) -> dict:
    """Seek to *timestamp_sec*, read the frame, write it as JPEG."""
    try:
        frame_number = round(timestamp_sec * fps)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "frame_extraction_failed", "reason": "Could not reopen video file"}

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            return {
                "status": "frame_extraction_failed",
                "reason": f"Could not read frame {frame_number}",
            }

        cv2.imwrite(out_path, frame)
        return {"status": "ok", "frame_number": frame_number, "path": out_path}
    except Exception as e:
        return {"status": "frame_extraction_failed", "reason": str(e)}
