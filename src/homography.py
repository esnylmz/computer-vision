"""
src/homography.py — Keyboard rectification via homography.

Computes a perspective transform from metadata keyboard corner points
to a normalized rectangular keyboard space, and warps fingertip
coordinates accordingly.

Using metadata corner points is intentional and acceptable for Group B
(they are camera calibration data, not hand/music annotations).

Usage:
    from src.homography import (
        compute_keyboard_homography,
        warp_points,
        add_keyboard_coords_to_landmarks,
    )

    H, dst_size = compute_keyboard_homography(corners)
    warped = warp_points(pixel_pts, H)
"""

import cv2
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# Standard rectified keyboard dimensions (pixels).
# ~10 px per key × 88 keys, depth proportional.
RECT_KB_WIDTH = 880
RECT_KB_HEIGHT = 110


# ═══════════════════════════════════════════════════════════════════
# Core homography
# ═══════════════════════════════════════════════════════════════════

def parse_corners(raw: Union[str, Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Parse keyboard corners from manifest (may be a JSON string or dict).

    Returns dict with keys LT, RT, RB, LB → (x, y) tuples.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    corners = {}
    for key in ["LT", "RT", "RB", "LB"]:
        val = raw.get(key, [0, 0])
        if isinstance(val, (list, tuple)) and len(val) == 2:
            corners[key] = (int(val[0]), int(val[1]))
        else:
            corners[key] = (0, 0)
    return corners


def corners_valid(corners: Dict[str, Tuple[int, int]]) -> bool:
    """Return True if all four corners are non-zero (i.e. metadata exists)."""
    for key in ["LT", "RT", "RB", "LB"]:
        x, y = corners.get(key, (0, 0))
        if x == 0 and y == 0:
            return False
    return True


def compute_keyboard_homography(
    corners: Dict[str, Tuple[int, int]],
    dst_width: int = RECT_KB_WIDTH,
    dst_height: int = RECT_KB_HEIGHT,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute a 3×3 homography that maps the four metadata keyboard corners
    to a rectified rectangle of size (dst_width, dst_height).

    Corner ordering:
        LT ── RT
        │      │
        LB ── RB

    Args:
        corners:  dict with keys LT, RT, RB, LB → (x, y).
        dst_width:  Width of rectified image (pixels).
        dst_height: Height of rectified image (pixels).

    Returns:
        (H, dst_size)  where H is a 3×3 float64 matrix and
        dst_size = (dst_width, dst_height).
    """
    src_pts = np.array(
        [corners["LT"], corners["RT"], corners["RB"], corners["LB"]],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height],
        ],
        dtype=np.float32,
    )

    H, _status = cv2.findHomography(src_pts, dst_pts)
    return H, (dst_width, dst_height)


# ═══════════════════════════════════════════════════════════════════
# Point / frame warping
# ═══════════════════════════════════════════════════════════════════

def warp_points(
    points_xy: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """
    Apply homography *H* to an (N, 2) array of (x, y) points.

    Returns an (N, 2) array of warped coordinates.
    """
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)


def warp_frame(
    frame: np.ndarray,
    H: np.ndarray,
    dst_size: Tuple[int, int],
) -> np.ndarray:
    """Warp a full image using homography *H*. Returns the rectified image."""
    return cv2.warpPerspective(frame, H, dst_size)


# ═══════════════════════════════════════════════════════════════════
# DataFrame integration
# ═══════════════════════════════════════════════════════════════════

def add_keyboard_coords_to_landmarks(
    landmarks_df: pd.DataFrame,
    corners: Dict[str, Tuple[int, int]],
    dst_width: int = RECT_KB_WIDTH,
    dst_height: int = RECT_KB_HEIGHT,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute homography, warp every fingertip (x_pixel, y_pixel) into
    normalized keyboard space, and add columns **x_kbd**, **y_kbd** to
    the DataFrame (values in [0, 1] if the point is inside the keyboard).

    Returns:
        (landmarks_df_with_kbd, H)
    """
    H, (w, h) = compute_keyboard_homography(corners, dst_width, dst_height)

    if landmarks_df.empty:
        landmarks_df = landmarks_df.copy()
        landmarks_df["x_kbd"] = pd.Series(dtype=float)
        landmarks_df["y_kbd"] = pd.Series(dtype=float)
        return landmarks_df, H

    pixel_pts = landmarks_df[["x_pixel", "y_pixel"]].values
    warped = warp_points(pixel_pts, H)

    landmarks_df = landmarks_df.copy()
    landmarks_df["x_kbd"] = warped[:, 0] / dst_width
    landmarks_df["y_kbd"] = warped[:, 1] / dst_height

    return landmarks_df, H
