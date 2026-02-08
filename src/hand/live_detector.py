"""
Live Hand Detection via MediaPipe

Runs MediaPipe Hands directly on video frames to extract 21-keypoint hand
skeletons, eliminating the dependency on pre-extracted skeleton JSON files.

Key improvements over basic MediaPipe usage:
    - Video mode (static_image_mode=False) for temporal tracking
    - Lower detection confidence for occluded / fast-moving hands
    - Dense frame sampling with configurable stride
    - Automatic left/right hand labelling
    - Output format compatible with the existing SkeletonLoader arrays

References:
    - Zhang et al. (2020) — MediaPipe Hands: On-device Real-time Hand Tracking

Usage:
    from src.hand.live_detector import LiveHandDetector

    detector = LiveHandDetector()
    left_arr, right_arr = detector.detect_from_video(video_path)
    # left_arr:  (T, 21, 3) float32 — NaN where no hand detected
    # right_arr: (T, 21, 3) float32
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class LiveDetectionConfig:
    """Configuration for live hand detection."""
    # MediaPipe model parameters
    model_complexity: int = 1           # 0 = lite, 1 = full (more accurate)
    max_num_hands: int = 2
    min_detection_confidence: float = 0.3   # lower → detect more (even blurry)
    min_tracking_confidence: float = 0.3    # lower → keep tracking longer
    # Frame sampling
    frame_stride: int = 1                   # process every N-th frame
    # Video mode gives temporal tracking (much better than static_image_mode)
    static_image_mode: bool = False


class LiveHandDetector:
    """
    Detect hands on every frame of a video using MediaPipe Hands.

    Returns two arrays ``(left, right)`` shaped ``(T, 21, 3)`` where
    ``T`` is the number of video frames, 21 is the landmark count,
    and the 3 channels are ``(x, y, z)`` in **normalised [0, 1]** coords
    (same convention as SkeletonLoader).  Frames with no detection are
    filled with ``NaN``.
    """

    def __init__(self, config: Optional[LiveDetectionConfig] = None):
        if mp is None:
            raise ImportError(
                "mediapipe is required for live hand detection. "
                "Install with: pip install mediapipe"
            )
        self.config = config or LiveDetectionConfig()

    def detect_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a video file and return per-frame hand landmarks.

        Args:
            video_path: path to the video file
            max_frames: stop after this many frames (None = all)
            progress_callback: optional callable(current_frame, total_frames)

        Returns:
            (left_landmarks, right_landmarks) — each (T, 21, 3) float32
            Coordinates are normalised to [0, 1].  NaN = not detected.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        # Pre-allocate output arrays (NaN = missing)
        left_all = np.full((total, 21, 3), np.nan, dtype=np.float32)
        right_all = np.full((total, 21, 3), np.nan, dtype=np.float32)

        cfg = self.config
        hands = mp.solutions.hands.Hands(
            static_image_mode=cfg.static_image_mode,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        frame_idx = 0
        while frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames according to stride (but still advance the index)
            if frame_idx % cfg.frame_stride != 0:
                frame_idx += 1
                continue

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_info.classification[0].label  # 'Left' or 'Right'
                    # NOTE: MediaPipe labels are MIRRORED (as if looking in a
                    # mirror).  For a top-down piano view we keep them as-is
                    # because the camera already provides the correct
                    # perspective.  Adjust if your setup differs.
                    arr = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                        dtype=np.float32,
                    )  # (21, 3)

                    if label == "Left":
                        left_all[frame_idx] = arr
                    else:
                        right_all[frame_idx] = arr

            if progress_callback and frame_idx % 100 == 0:
                progress_callback(frame_idx, total)

            frame_idx += 1

        hands.close()
        cap.release()

        return left_all, right_all

    def detect_from_frames(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a list of BGR frames (useful when frames are already loaded).

        Args:
            frames: list of BGR numpy arrays

        Returns:
            (left_landmarks, right_landmarks) — each (N, 21, 3) float32
        """
        n = len(frames)
        left_all = np.full((n, 21, 3), np.nan, dtype=np.float32)
        right_all = np.full((n, 21, 3), np.nan, dtype=np.float32)

        cfg = self.config
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,  # no temporal tracking across arbitrary frames
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_info.classification[0].label
                    arr = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                        dtype=np.float32,
                    )
                    if label == "Left":
                        left_all[i] = arr
                    else:
                        right_all[i] = arr

        hands.close()
        return left_all, right_all

    @staticmethod
    def detection_rate(landmarks: np.ndarray) -> float:
        """
        Fraction of frames where this hand was detected.

        Args:
            landmarks: (T, 21, 3) array — NaN where not detected

        Returns:
            float in [0, 1]
        """
        if landmarks.size == 0:
            return 0.0
        valid = ~np.isnan(landmarks[:, 0, 0])
        return float(np.mean(valid))
