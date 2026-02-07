"""
src/teacher_labels.py — Group A teacher label generation.

Group A uses annotations (TSV / MIDI, hand skeleton JSON) with
REFINED temporal filtering to produce high-quality frame-level
press / no-press labels. These labels are used **only** for training
Group B's CNN — never at inference time.

Group A has BETTER landmarks than Group B because:
  - Uses pre-extracted JSON skeletons (higher quality than raw MediaPipe)
  - Applies sophisticated temporal filtering (Hampel + SavGol)
  - This creates cleaner teacher signals for CNN training

Usage:
    from src.teacher_labels import generate_teacher_labels_groupA

    labeled_df = generate_teacher_labels_groupA(
        video_id, dataset, tsv_df, fps=60, clip_duration=20,
    )
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# ═══════════════════════════════════════════════════════════════════
# MIDI / keyboard helpers
# ═══════════════════════════════════════════════════════════════════

MIDI_LOW = 21    # A0
MIDI_HIGH = 108  # C8


def midi_pitch_to_kbd_x(pitch: int) -> float:
    """Map a MIDI pitch to normalised keyboard x ∈ [0, 1]."""
    return (pitch - MIDI_LOW) / (MIDI_HIGH - MIDI_LOW)


def parse_tsv_events(
    tsv_df: pd.DataFrame, clip_duration: float
) -> List[Dict]:
    """
    Convert TSV annotation rows into a list of note event dicts.

    Each dict: {onset, offset, pitch, velocity, x_kbd}.
    Notes starting after *clip_duration* are dropped.
    """
    events: List[Dict] = []
    for _, row in tsv_df.iterrows():
        onset = float(row["onset"])
        if onset > clip_duration:
            continue
        offset = float(row["key_offset"]) if pd.notna(row.get("key_offset")) else onset + 0.2
        offset = min(offset, clip_duration)
        pitch = int(row["note"])
        events.append(
            {
                "onset": onset,
                "offset": offset,
                "pitch": pitch,
                "velocity": int(row.get("velocity", 64)),
                "x_kbd": midi_pitch_to_kbd_x(pitch),
            }
        )
    return events


# ═══════════════════════════════════════════════════════════════════
# Raw press-label generation
# ═══════════════════════════════════════════════════════════════════

def _active_notes_at(events: List[Dict], t: float) -> List[Dict]:
    """Return events whose [onset, offset] covers time *t*."""
    return [e for e in events if e["onset"] <= t <= e["offset"]]


def generate_raw_labels(
    landmarks_df: pd.DataFrame,
    events: List[Dict],
    fps: float,
    proximity_thresh: float = 0.06,
) -> pd.Series:
    """
    For every row in *landmarks_df* decide press = 1 / 0.

    A fingertip is labelled *press* when:
      1. At least one MIDI note is active at the row's time, **and**
      2. |fingertip.x_kbd − note.x_kbd| < *proximity_thresh*.

    Returns a Series of int (0 / 1) aligned with *landmarks_df* index.
    """
    labels = np.zeros(len(landmarks_df), dtype=np.float32)

    if "x_kbd" not in landmarks_df.columns:
        return pd.Series(labels, index=landmarks_df.index, name="press_raw")

    for i, (_, row) in enumerate(landmarks_df.iterrows()):
        t = row["frame_idx"] / fps
        active = _active_notes_at(events, t)
        if not active:
            continue
        fx = row["x_kbd"]
        for ev in active:
            if abs(fx - ev["x_kbd"]) < proximity_thresh:
                labels[i] = 1.0
                break

    return pd.Series(labels, index=landmarks_df.index, name="press_raw")


# ═══════════════════════════════════════════════════════════════════
# Temporal smoothing
# ═══════════════════════════════════════════════════════════════════

def smooth_labels(
    landmarks_df: pd.DataFrame,
    raw_col: str = "press_raw",
    sigma: float = 1.5,
) -> pd.Series:
    """
    Gaussian-smooth raw binary labels **per fingertip track**
    (same hand + finger across frames).

    Returns a Series 'press_smooth' aligned with the DataFrame index.
    """
    smoothed = landmarks_df[raw_col].copy().astype(float)

    for (hand, fid), grp in landmarks_df.groupby(["hand", "finger_id"]):
        if len(grp) < 3:
            continue
        vals = grp[raw_col].values.astype(float)
        sm = gaussian_filter1d(vals, sigma=sigma)
        smoothed.loc[grp.index] = sm

    smoothed.name = "press_smooth"
    return smoothed


# ═══════════════════════════════════════════════════════════════════
# Hampel filter for outlier removal
# ═══════════════════════════════════════════════════════════════════

def hampel_filter(
    data: np.ndarray,
    window_size: int = 20,
    n_sigmas: float = 3.0,
) -> np.ndarray:
    """
    Hampel filter for outlier detection and removal.
    
    Replaces outliers (> n_sigmas * MAD from median) with the median.
    Applied per-dimension independently.
    """
    n = len(data)
    filtered = data.copy()
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = data[start:end]
        
        # Handle NaN
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < 3:
            continue
        
        valid_window = window[valid_mask]
        median = np.median(valid_window)
        mad = np.median(np.abs(valid_window - median))
        
        if mad < 1e-8:
            continue
        
        threshold = n_sigmas * 1.4826 * mad  # 1.4826 converts MAD to stdev
        if np.abs(data[i] - median) > threshold:
            filtered[i] = median
    
    return filtered


def temporal_filter_landmarks(
    landmarks: np.ndarray,
    hampel_window: int = 20,
    hampel_threshold: float = 3.0,
    savgol_window: int = 11,
    savgol_order: int = 3,
    max_interpolation_gap: int = 30,
) -> np.ndarray:
    """
    Apply Group A's refined temporal filtering pipeline.
    
    Args:
        landmarks: (T, 21, 3) array (may contain NaN for missing frames)
        hampel_window: Window size for Hampel filter
        hampel_threshold: Number of MADs for outlier detection
        savgol_window: Window for Savitzky-Golay smoothing
        savgol_order: Polynomial order for SavGol
        max_interpolation_gap: Maximum frames to interpolate
    
    Returns:
        Filtered (T, 21, 3) array
    """
    T, num_landmarks, dim = landmarks.shape
    filtered = landmarks.copy()
    
    # Process each landmark independently
    for lm_idx in range(num_landmarks):
        for d in range(dim):
            series = landmarks[:, lm_idx, d]
            
            # 1. Hampel filter (outlier removal)
            series = hampel_filter(series, hampel_window, hampel_threshold)
            
            # 2. Interpolate small gaps
            valid_mask = ~np.isnan(series)
            if valid_mask.sum() < 2:
                continue
            
            valid_indices = np.where(valid_mask)[0]
            for i in range(len(valid_indices) - 1):
                start = int(valid_indices[i])
                end = int(valid_indices[i + 1])
                gap = end - start - 1  # number of missing positions between start and end
                
                if gap > 0 and gap <= max_interpolation_gap:
                    # Fill only indices start+1 .. end-1 (length = gap); do NOT use series[start:end] (length = gap+1)
                    interp = np.linspace(float(series[start]), float(series[end]), num=gap + 2)[1:-1]
                    series[start + 1:end] = interp
            
            # 3. Savitzky-Golay smoothing (only on valid data)
            valid_mask = ~np.isnan(series)
            if valid_mask.sum() >= savgol_window:
                try:
                    smoothed = savgol_filter(
                        series[valid_mask], savgol_window, savgol_order,
                    )
                    series[valid_mask] = smoothed
                except:
                    pass  # Keep original if SavGol fails
            
            filtered[:, lm_idx, d] = series
    
    return filtered


# ═══════════════════════════════════════════════════════════════════
# Group A landmark loading with refinement
# ═══════════════════════════════════════════════════════════════════

def load_and_refine_skeleton(
    dataset,
    video_id: str,
    clip_duration: float,
    fps: float,
    frame_step: int = 1,
) -> pd.DataFrame:
    """
    Load hand skeleton JSON and apply Group A's refined filtering.
    
    Returns a landmarks DataFrame in the same format as Group B
    (for compatibility), but with higher-quality filtered coordinates.
    """
    from src.hand.skeleton_loader import SkeletonLoader
    
    sample = dataset.get_sample_by_id(video_id)
    if sample is None:
        return pd.DataFrame()
    
    # Load skeleton JSON
    skeleton_json = dataset.load_skeleton(sample)
    loader = SkeletonLoader()
    hands = loader._parse_json(skeleton_json)
    
    # Convert to arrays
    max_frames = int(clip_duration * fps)
    left_arr = loader.to_array(hands['left'], fill_missing=True, total_frames=max_frames)
    right_arr = loader.to_array(hands['right'], fill_missing=True, total_frames=max_frames)
    
    # Apply refined temporal filtering
    if left_arr.size > 0 and len(left_arr) > 0:
        left_arr = temporal_filter_landmarks(left_arr)
    if right_arr.size > 0 and len(right_arr) > 0:
        right_arr = temporal_filter_landmarks(right_arr)
    
    # Convert to DataFrame (matching Group B format)
    from src.mediapipe_extract import FINGERTIP_IDS
    
    rows = []
    for fidx in range(0, min(max_frames, max(len(left_arr), len(right_arr))), frame_step):
        for hand_arr, hand_label in [(left_arr, "left"), (right_arr, "right")]:
            if fidx >= len(hand_arr):
                continue
            
            for fid, (lm_idx, fname) in FINGERTIP_IDS.items():
                lm = hand_arr[fidx, lm_idx]
                if np.isnan(lm).any():
                    continue
                
                rows.append({
                    "frame_idx": fidx,
                    "time_sec": round(fidx / fps, 4),
                    "hand": hand_label,
                    "finger_id": fid,
                    "finger_name": fname,
                    "x_norm": float(lm[0]),
                    "y_norm": float(lm[1]),
                    "x_pixel": float(lm[0] * 1920),  # PianoVAM resolution
                    "y_pixel": float(lm[1] * 1080),
                    "confidence": 1.0,  # Skeleton JSON assumed high quality
                })
    
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Public entry points
# ═══════════════════════════════════════════════════════════════════

def generate_teacher_labels_for_video(
    landmarks_df: pd.DataFrame,
    tsv_df: pd.DataFrame,
    fps: float = 60.0,
    clip_duration: float = 20.0,
    proximity_thresh: float = 0.06,
    smooth_sigma: float = 1.5,
) -> pd.DataFrame:
    """
    Generate teacher press labels for one video's landmarks.
    
    (Original function for Group B landmarks)

    Adds columns ``press_raw`` (binary) and ``press_smooth`` (continuous)
    to a **copy** of *landmarks_df*.
    """
    events = parse_tsv_events(tsv_df, clip_duration)

    df = landmarks_df.copy()
    df["press_raw"] = generate_raw_labels(df, events, fps, proximity_thresh)
    df["press_smooth"] = smooth_labels(df, "press_raw", sigma=smooth_sigma)

    n_pos = int(df["press_raw"].sum())
    n_tot = len(df)
    print(
        f"    Teacher labels: {n_pos}/{n_tot} press "
        f"({100 * n_pos / max(n_tot, 1):.1f}%)"
    )
    return df


def generate_teacher_labels_groupA(
    video_id: str,
    dataset,
    tsv_df: pd.DataFrame,
    fps: float = 60.0,
    clip_duration: float = 20.0,
    proximity_thresh: float = 0.06,
    smooth_sigma: float = 1.5,
    frame_step: int = 1,
) -> pd.DataFrame:
    """
    Generate teacher labels using Group A's REFINED annotations.
    
    This is the high-quality path:
      1. Load hand skeleton JSON
      2. Apply Hampel + SavGol filtering
      3. Extract fingertips
      4. Generate press labels via MIDI proximity
      5. Smooth labels
    
    Returns labeled DataFrame with higher quality than Group B.
    """
    # Load refined skeleton
    landmarks_df = load_and_refine_skeleton(
        dataset, video_id, clip_duration, fps, frame_step,
    )
    
    if landmarks_df.empty:
        print(f"    No skeleton data for {video_id}")
        return landmarks_df
    
    # Generate labels (same as Group B path)
    return generate_teacher_labels_for_video(
        landmarks_df, tsv_df, fps, clip_duration,
        proximity_thresh, smooth_sigma,
    )


# ═══════════════════════════════════════════════════════════════════
# Timeline plotting helper
# ═══════════════════════════════════════════════════════════════════

def plot_teacher_timeline(
    labeled_df: pd.DataFrame,
    video_id: str = "",
    hand: str = "right",
    finger_name: str = "index",
    ax=None,
):
    """
    Plot raw vs smoothed teacher labels over time for one fingertip.

    If *ax* is None a new matplotlib figure is created.
    """
    import matplotlib.pyplot as plt

    sub = labeled_df[
        (labeled_df["hand"] == hand)
        & (labeled_df["finger_name"] == finger_name)
    ].sort_values("frame_idx")

    if sub.empty:
        print(f"  No data for {hand}/{finger_name} — skipping plot")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.fill_between(
        sub["time_sec"], 0, sub["press_raw"],
        alpha=0.25, color="red", label="raw (binary)",
    )
    ax.plot(
        sub["time_sec"], sub["press_smooth"],
        linewidth=1.5, color="blue", label="smoothed",
    )
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("press label")
    ax.set_title(
        f"Teacher labels — {video_id}  {hand} {finger_name}"
    )
    ax.legend(fontsize=8)
    return ax
