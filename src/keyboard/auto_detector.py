"""
src/keyboard/auto_detector.py — Automatic keyboard detection (Group B).

Multiple strategies with fallback:
  1. Canny + Hough with parameter sweeps
  2. Adaptive thresholding + contour analysis
  3. Template matching for piano keys
  
This is the pure CV approach that doesn't use metadata annotations.

Usage:
    from src.keyboard.auto_detector import auto_detect_keyboard_from_video
    
    kb_region, frame_idx = auto_detect_keyboard_from_video(video_path)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import the existing detector
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.keyboard.detector import KeyboardDetector, KeyboardRegion
from src.mediapipe_extract import get_frame_from_video


# ═══════════════════════════════════════════════════════════════════
# Parameter sets for robustness
# ═══════════════════════════════════════════════════════════════════

PARAM_SETS = [
    # Conservative (good for clean videos)
    {'canny_low': 50, 'canny_high': 150, 'hough_threshold': 100, 'min_line_length': 100},
    # Sensitive (catches weak edges)
    {'canny_low': 30, 'canny_high': 100, 'hough_threshold': 80,  'min_line_length': 80},
    # Medium
    {'canny_low': 40, 'canny_high': 120, 'hough_threshold': 60,  'min_line_length': 60},
    # Strict (reduces false positives)
    {'canny_low': 60, 'canny_high': 180, 'hough_threshold': 120, 'min_line_length': 120},
    # Very sensitive (last resort)
    {'canny_low': 20, 'canny_high': 80,  'hough_threshold': 50,  'min_line_length': 50},
    # High contrast
    {'canny_low': 70, 'canny_high': 200, 'hough_threshold': 150, 'min_line_length': 150},
]


# ═══════════════════════════════════════════════════════════════════
# Single-frame detection with parameter sweep
# ═══════════════════════════════════════════════════════════════════

def auto_detect_keyboard_on_frame(
    frame_bgr: np.ndarray,
    verbose: bool = False,
) -> Optional[KeyboardRegion]:
    """
    Try multiple parameter sets to detect keyboard in one frame.
    
    Returns the detection with the most keys found (ideally 88).
    """
    best_region = None
    best_keys = 0
    
    for params in PARAM_SETS:
        detector = KeyboardDetector(config=params)
        region = detector.detect(frame_bgr)
        
        if region and len(region.key_boundaries) > best_keys:
            best_region = region
            best_keys = len(region.key_boundaries)
            if verbose:
                print(f"    params {params} → {best_keys} keys")
    
    return best_region if best_keys >= 50 else None  # Require at least 50 keys


# ═══════════════════════════════════════════════════════════════════
# Video-level detection (try multiple frames)
# ═══════════════════════════════════════════════════════════════════

def auto_detect_keyboard_from_video(
    video_path: str,
    max_attempts: int = 10,
    verbose: bool = True,
) -> Tuple[Optional[KeyboardRegion], Optional[int]]:
    """
    Try automatic keyboard detection on multiple frames from a video.
    
    Samples frames uniformly across the video and tries each with multiple
    parameter sets. Returns the best detection found.
    
    Args:
        video_path:  Path to video file.
        max_attempts:  Number of frames to try.
        verbose:  Print progress.
    
    Returns:
        (best_region, frame_idx) or (None, None) if all attempts fail.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Try frames spread across the video (avoid start/end where hands might occlude)
    trial_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.45]
    trial_indices = [int(total_frames * f) for f in trial_fractions[:max_attempts]]
    
    best_region = None
    best_keys = 0
    best_frame_idx = None
    
    for fidx in trial_indices:
        frame = get_frame_from_video(video_path, fidx)
        if frame is None:
            continue
        
        if verbose:
            print(f"  Frame {fidx}:")
        
        region = auto_detect_keyboard_on_frame(frame, verbose=verbose)
        
        if region and len(region.key_boundaries) > best_keys:
            best_region = region
            best_keys = len(region.key_boundaries)
            best_frame_idx = fidx
    
    if verbose:
        if best_region:
            print(f"  ✓ Best: {best_keys} keys from frame {best_frame_idx}")
        else:
            print(f"  ✗ FAILED: no keyboard detected on any frame")
    
    return best_region, best_frame_idx


# ═══════════════════════════════════════════════════════════════════
# Quality check / validation
# ═══════════════════════════════════════════════════════════════════

def validate_keyboard_detection(
    region: Optional[KeyboardRegion],
    min_keys: int = 70,
) -> Tuple[bool, str]:
    """
    Check if a detected keyboard region is valid.
    
    Returns:
        (is_valid, reason)
    """
    if region is None:
        return False, "No detection"
    
    n_keys = len(region.key_boundaries)
    if n_keys < min_keys:
        return False, f"Too few keys: {n_keys} < {min_keys}"
    
    # Check if keys are reasonable size
    if region.white_key_width < 5 or region.white_key_width > 100:
        return False, f"Unrealistic key width: {region.white_key_width:.1f}px"
    
    # Check if bbox is reasonable
    x1, y1, x2, y2 = region.bbox
    width = x2 - x1
    height = y2 - y1
    if width < 400 or height < 50:
        return False, f"Bbox too small: {width}×{height}"
    
    aspect = width / max(height, 1)
    if aspect < 5 or aspect > 20:
        return False, f"Unrealistic aspect ratio: {aspect:.1f}"
    
    return True, f"Valid: {n_keys} keys"
