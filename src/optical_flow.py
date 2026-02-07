"""
src/optical_flow.py — Optical flow features for motion analysis.

Adds motion features to complement static image features.
Helps distinguish "finger moving toward key" from "finger hovering."

Usage:
    from src.optical_flow import compute_optical_flow_features
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════
# Dense optical flow
# ═══════════════════════════════════════════════════════════════════

def compute_optical_flow(
    frame1_gray: np.ndarray,
    frame2_gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow between two frames.
    
    Args:
        frame1_gray: First frame (H, W) uint8
        frame2_gray: Second frame (H, W) uint8
    
    Returns:
        (flow_x, flow_y) each (H, W) float32
    """
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow[..., 0], flow[..., 1]


# ═══════════════════════════════════════════════════════════════════
# Extract flow features per fingertip
# ═══════════════════════════════════════════════════════════════════

def add_optical_flow_features(
    landmarks_df: pd.DataFrame,
    video_path: str,
    crop_size: int = 64,
) -> pd.DataFrame:
    """
    Add optical flow features to landmarks DataFrame.
    
    For each fingertip detection, computes flow in the local crop region
    and adds columns: flow_mag, flow_angle, flow_y (vertical motion).
    
    Args:
        landmarks_df: DataFrame with fingertip detections
        video_path: Path to video file
        crop_size: Size of crop around fingertip
    
    Returns:
        DataFrame with added flow columns
    """
    if landmarks_df.empty:
        df = landmarks_df.copy()
        df["flow_mag"] = pd.Series(dtype=float)
        df["flow_angle"] = pd.Series(dtype=float)
        df["flow_y"] = pd.Series(dtype=float)
        return df
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Initialize flow columns
    flow_mags = np.full(len(landmarks_df), np.nan)
    flow_angles = np.full(len(landmarks_df), np.nan)
    flow_ys = np.full(len(landmarks_df), np.nan)
    
    # Group by frame
    grouped = landmarks_df.groupby("frame_idx")
    prev_frame_gray = None
    prev_fidx = None
    
    for fidx in sorted(grouped.groups.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute flow (need two consecutive frames)
        if prev_frame_gray is not None and prev_fidx == fidx - 1:
            flow_x, flow_y = compute_optical_flow(prev_frame_gray, curr_frame_gray)
            
            # Extract flow at each fingertip
            for loc, row in grouped.get_group(fidx).iterrows():
                cx, cy = int(row["x_pixel"]), int(row["y_pixel"])
                
                # Sample flow in a small window around fingertip
                half = crop_size // 4
                x1, x2 = max(0, cx - half), min(frame.shape[1], cx + half)
                y1, y2 = max(0, cy - half), min(frame.shape[0], cy + half)
                
                if x2 > x1 and y2 > y1:
                    fx = flow_x[y1:y2, x1:x2].mean()
                    fy = flow_y[y1:y2, x1:x2].mean()
                    mag = np.sqrt(fx**2 + fy**2)
                    angle = np.arctan2(fy, fx)
                    
                    flow_mags[loc] = mag
                    flow_angles[loc] = angle
                    flow_ys[loc] = fy  # Positive = downward motion
        
        prev_frame_gray = curr_frame_gray
        prev_fidx = fidx
    
    cap.release()
    
    df = landmarks_df.copy()
    df["flow_mag"] = flow_mags
    df["flow_angle"] = flow_angles
    df["flow_y"] = flow_ys
    
    return df


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_optical_flow(
    frame: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    step: int = 16,
) -> np.ndarray:
    """
    Draw optical flow vectors on frame.
    
    Args:
        frame: BGR image (H, W, 3)
        flow_x, flow_y: Flow fields (H, W)
        step: Spacing between arrows
    
    Returns:
        Annotated frame (H, W, 3)
    """
    vis = frame.copy()
    h, w = frame.shape[:2]
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow_x[y, x], flow_y[y, x]
            mag = np.sqrt(fx**2 + fy**2)
            
            if mag > 0.5:  # Only draw significant motion
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx * 3), int(y + fy * 3)),
                    (0, 255, 0),
                    1,
                    tipLength=0.3,
                )
    
    return vis
