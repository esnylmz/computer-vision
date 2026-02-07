"""
src/viz.py — Visualization utilities for sanity checks.

Saves annotated frames with:
  - Fingertip overlays (coloured by finger, labelled L/R)
  - Keyboard corner polygon
  - Rectified keyboard ROI with warped fingertip positions

Usage:
    from src.viz import save_sanity_check_frames
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Colour palette (BGR for OpenCV)
# ═══════════════════════════════════════════════════════════════════

FINGER_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "thumb": (0, 0, 255),       # red
    "index": (0, 255, 0),       # green
    "middle": (255, 0, 0),      # blue
    "ring": (0, 255, 255),      # yellow
    "pinky": (255, 0, 255),     # magenta
}


# ═══════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════

def draw_fingertip_overlay(
    frame_bgr: np.ndarray,
    frame_landmarks: pd.DataFrame,
    corners: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Draw fingertip circles + hand/finger labels on a **BGR** frame.

    Optionally draws the keyboard corner polygon if *corners* is given.
    """
    vis = frame_bgr.copy()

    # Draw keyboard quadrilateral
    if corners is not None:
        pts = np.array(
            [corners["LT"], corners["RT"], corners["RB"], corners["LB"]],
            dtype=np.int32,
        )
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        for name, pt in corners.items():
            cv2.putText(
                vis, name, (int(pt[0]) + 5, int(pt[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

    # Draw fingertip detections
    for _, row in frame_landmarks.iterrows():
        x = int(row["x_pixel"])
        y = int(row["y_pixel"])
        color = FINGER_COLORS_BGR.get(row["finger_name"], (255, 255, 255))
        hand_char = row["hand"][0].upper()  # "L" or "R"
        label = f'{hand_char}-{row["finger_name"]}'

        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 10, (255, 255, 255), 1)
        cv2.putText(
            vis, label, (x + 12, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )

    return vis


def draw_rectified_with_fingertips(
    frame_bgr: np.ndarray,
    frame_landmarks: pd.DataFrame,
    H: np.ndarray,
    dst_size: Tuple[int, int],
) -> np.ndarray:
    """
    Warp the frame into rectified keyboard space and draw warped
    fingertip positions on top.
    """
    rectified = cv2.warpPerspective(frame_bgr, H, dst_size)

    if frame_landmarks.empty:
        return rectified

    # Warp fingertip pixel coords
    from src.homography import warp_points  # local import to avoid circular

    pts = frame_landmarks[["x_pixel", "y_pixel"]].values
    warped = warp_points(pts, H)

    for i, (_, row) in enumerate(frame_landmarks.iterrows()):
        wx, wy = int(warped[i, 0]), int(warped[i, 1])
        color = FINGER_COLORS_BGR.get(row["finger_name"], (255, 255, 255))
        hand_char = row["hand"][0].upper()

        cv2.circle(rectified, (wx, wy), 6, color, -1)
        cv2.circle(rectified, (wx, wy), 8, (255, 255, 255), 1)
        cv2.putText(
            rectified,
            f'{hand_char}-{row["finger_name"]}',
            (wx + 8, wy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )

    return rectified


# ═══════════════════════════════════════════════════════════════════
# Batch sanity-check frame saver
# ═══════════════════════════════════════════════════════════════════

def save_sanity_check_frames(
    video_path: str,
    landmarks_df: pd.DataFrame,
    H: np.ndarray,
    dst_size: Tuple[int, int],
    corners: Dict[str, Tuple[int, int]],
    output_dir: str,
    video_id: str,
    n_frames: int = 10,
) -> int:
    """
    Save *n_frames* annotated frames per video:

    For each selected frame:
      1. **overlay/** — original frame with fingertip circles + keyboard polygon.
      2. **rectified/** — warped keyboard ROI with fingertip positions.

    Returns the number of frames actually saved.
    """
    output_dir = Path(output_dir)
    overlay_dir = output_dir / video_id / "overlay"
    rectified_dir = output_dir / video_id / "rectified"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    if landmarks_df.empty:
        print(f"  {video_id}: no landmarks — skipping visualisation")
        return 0

    # Pick n_frames evenly spaced from available frames
    unique_frames = sorted(landmarks_df["frame_idx"].unique())
    if len(unique_frames) <= n_frames:
        selected = unique_frames
    else:
        idxs = np.linspace(0, len(unique_frames) - 1, n_frames, dtype=int)
        selected = [unique_frames[i] for i in idxs]

    cap = cv2.VideoCapture(str(video_path))
    saved = 0

    for fidx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        frame_lms = landmarks_df[landmarks_df["frame_idx"] == fidx]

        # 1) overlay
        overlay = draw_fingertip_overlay(frame_bgr, frame_lms, corners=corners)
        cv2.imwrite(
            str(overlay_dir / f"frame_{fidx:06d}.jpg"), overlay,
        )

        # 2) rectified keyboard with fingertip overlay
        rect = draw_rectified_with_fingertips(
            frame_bgr, frame_lms, H, dst_size,
        )
        cv2.imwrite(
            str(rectified_dir / f"frame_{fidx:06d}_rect.jpg"), rect,
        )

        saved += 1

    cap.release()
    print(
        f"  {video_id}: saved {saved} overlay + rectified frames "
        f"-> {output_dir / video_id}"
    )
    return saved
