"""
src/mediapipe_extract.py — Hand landmark extraction from raw video (MediaPipe).

This is Group B's hand detection front-end.
It runs ONLY on video frames — no pre-extracted JSON is used.

Usage:
    from src.mediapipe_extract import extract_landmarks_from_video

    landmarks_df, video_info = extract_landmarks_from_video(
        video_path="path/to/video.mp4",
        clip_duration=20.0,
        frame_step=10,
    )
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# MediaPipe 0.10.31+ removed the "solutions" API. Use mediapipe-numpy2 for compatibility.
if not hasattr(mp, "solutions"):
    raise ImportError(
        "This code requires MediaPipe's 'solutions' API (e.g. mp.solutions.hands). "
        "Newer MediaPipe (0.10.31+) removed it. Install the compatible package:\n"
        "  pip install mediapipe-numpy2\n"
        "Then re-run your code. If you are in Colab, run the Setup cell first."
    )


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# MediaPipe landmark indices for fingertips
FINGERTIP_IDS: Dict[int, Tuple[int, str]] = {
    0: (4, "thumb"),
    1: (8, "index"),
    2: (12, "middle"),
    3: (16, "ring"),
    4: (20, "pinky"),
}

# All 21 MediaPipe hand landmark names (for reference)
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


# ═══════════════════════════════════════════════════════════════════
# Main extraction function
# ═══════════════════════════════════════════════════════════════════

def extract_landmarks_from_video(
    video_path: str,
    clip_duration: float = 20.0,
    frame_step: int = 10,
    fps: float = 60.0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.4,
    max_num_hands: int = 2,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run MediaPipe Hands on raw video and extract fingertip landmarks.

    This function opens the video, reads frames at *frame_step* intervals
    up to *clip_duration* seconds, and detects hand landmarks with MediaPipe.
    Only fingertip positions (thumb, index, middle, ring, pinky) are returned.

    Args:
        video_path:  Path to the video file.
        clip_duration:  Process only the first N seconds.
        frame_step:  Extract every Nth frame.
        fps:  Fallback FPS if not readable from the video header.
        min_detection_confidence:  MediaPipe detection threshold.
        min_tracking_confidence:  MediaPipe tracking threshold.
        max_num_hands:  Maximum simultaneous hands to detect.

    Returns:
        (landmarks_df, video_info)

        landmarks_df columns:
            frame_idx, time_sec, hand, finger_id, finger_name,
            x_norm, y_norm, x_pixel, y_pixel, confidence

        video_info dict:
            fps, width, height, max_frame, n_processed, n_with_hands
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(int(clip_duration * actual_fps), total_frames)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_indices = list(range(0, max_frame, frame_step))
    rows: List[Dict] = []
    n_with_hands = 0

    for fidx in tqdm(frame_indices, desc="MediaPipe", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            n_with_hands += 1
            for hlm, hinfo in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                hand_label = hinfo.classification[0].label.lower()
                hand_conf = float(hinfo.classification[0].score)

                for fid, (lm_idx, fname) in FINGERTIP_IDS.items():
                    lm = hlm.landmark[lm_idx]
                    rows.append(
                        {
                            "frame_idx": fidx,
                            "time_sec": round(fidx / actual_fps, 4),
                            "hand": hand_label,
                            "finger_id": fid,
                            "finger_name": fname,
                            "x_norm": float(lm.x),
                            "y_norm": float(lm.y),
                            "x_pixel": float(lm.x * width),
                            "y_pixel": float(lm.y * height),
                            "confidence": hand_conf,
                        }
                    )

    hands.close()
    cap.release()

    landmarks_df = pd.DataFrame(rows)
    video_info = {
        "fps": actual_fps,
        "width": width,
        "height": height,
        "max_frame": max_frame,
        "n_processed": len(frame_indices),
        "n_with_hands": n_with_hands,
    }
    return landmarks_df, video_info


def get_frame_from_video(
    video_path: str, frame_idx: int
) -> Optional[np.ndarray]:
    """Read a single BGR frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
