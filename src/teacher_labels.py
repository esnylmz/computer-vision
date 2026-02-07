"""
src/teacher_labels.py — Group A teacher label generation.

Group A uses annotations (TSV / MIDI, optionally skeleton JSON)
to produce frame-level press / no-press labels.  These labels are
used **only** for training Group B's CNN — never at inference time.

Usage:
    from src.teacher_labels import generate_teacher_labels_for_video

    labeled_df = generate_teacher_labels_for_video(
        landmarks_df, tsv_df, fps=60, clip_duration=20,
    )
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d


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
# Public entry point
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
