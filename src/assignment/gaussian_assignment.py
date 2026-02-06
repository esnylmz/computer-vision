"""
Gaussian Probability-Based Finger Assignment

Assigns fingers to pressed piano keys using Gaussian probability distributions
centred on fingertip positions.  Only the **x-distance** (along the keyboard)
is used because, in a top-down piano view, the y-axis measures depth into the
keyboard – a dimension that varies systematically with finger length and
introduces a strong bias towards shorter fingers (thumb).

Based on Moryossef et al. (2023) "At Your Fingertips" methodology.

Usage:
    from src.assignment.gaussian_assignment import GaussianFingerAssigner

    assigner = GaussianFingerAssigner(key_boundaries)
    assignment = assigner.assign_from_landmarks(landmarks, key_idx, 'right')
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FingerAssignment:
    """Single finger assignment result."""
    note_onset: float          # Onset time in seconds
    frame_idx: int             # Frame index
    midi_pitch: int            # MIDI note number (21-108)
    key_idx: int               # Piano key index (0-87)
    assigned_finger: int       # 1-5 (thumb to pinky)
    hand: str                  # 'left' or 'right'
    confidence: float          # Assignment probability
    fingertip_position: Tuple[float, float]  # (x, y) of assigned fingertip
    all_probabilities: Optional[Dict[int, float]] = None

    @property
    def finger_name(self) -> str:
        names = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'}
        return names.get(self.assigned_finger, 'unknown')

    @property
    def label(self) -> str:
        hand_prefix = 'R' if self.hand == 'right' else 'L'
        return f"{hand_prefix}{self.assigned_finger}"


class GaussianFingerAssigner:
    """
    Assigns fingers to piano keys using a Gaussian probability model.

    For each pressed key the class computes a probability over all five
    fingertips based on **horizontal (x) distance only**, then selects the
    maximum-likelihood finger.

    Why x-only?
    -----------
    In the keyboard-homography space the key centres sit at the vertical
    midpoint of the key bounding box, while the fingertips are at the front
    edge (close to the player).  The resulting y-gap is ~50-100 px – far
    larger than any x-difference – so 2-D distance is dominated by y and
    the winner is determined by sub-pixel y-differences between fingers
    (thumb, being anatomically shorter, always wins).

    Sigma auto-scales to the average white-key width when not set explicitly,
    ensuring resolution-independent behaviour.
    """

    # Standard MediaPipe fingertip landmark indices
    FINGERTIP_LANDMARKS = {
        1: 4,    # Thumb tip
        2: 8,    # Index tip
        3: 12,   # Middle tip
        4: 16,   # Ring tip
        5: 20    # Pinky tip
    }

    FINGER_NAMES = {
        1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'
    }

    def __init__(
        self,
        key_boundaries: Dict[int, Tuple[int, int, int, int]],
        sigma: Optional[float] = None,
        candidate_range: int = 2,
        min_confidence: float = 0.01
    ):
        """
        Args:
            key_boundaries: Dict mapping key index to (x1, y1, x2, y2) bbox
            sigma: Gaussian spread in pixels (x-axis).  If *None* it is
                   automatically set to the mean white-key width.
            candidate_range: Number of adjacent keys to consider (±N)
            min_confidence: Minimum normalised probability for a valid
                            assignment
        """
        self.key_boundaries = key_boundaries
        self.candidate_range = candidate_range
        self.min_confidence = min_confidence

        # Pre-compute key centres
        self.key_centers: Dict[int, Tuple[float, float]] = {}
        for key_idx, (x1, y1, x2, y2) in key_boundaries.items():
            self.key_centers[key_idx] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # Auto-compute sigma from average key width when not given
        if sigma is not None:
            self.sigma = float(sigma)
        else:
            widths = [float(x2 - x1) for x1, y1, x2, y2 in key_boundaries.values()]
            self.sigma = float(np.mean(widths)) if widths else 15.0

    # ------------------------------------------------------------------
    # Core probability computation
    # ------------------------------------------------------------------

    def compute_probability(
        self,
        fingertip: Tuple[float, float],
        key_center: Tuple[float, float]
    ) -> float:
        """
        Compute Gaussian probability of a finger pressing a key.

        Uses **x-distance only** to avoid the systematic y-bias caused by
        differing finger lengths in the top-down camera view.
        """
        dx = abs(fingertip[0] - key_center[0])
        return float(np.exp(-dx ** 2 / (2.0 * self.sigma ** 2)))

    # ------------------------------------------------------------------
    # Single-note assignment
    # ------------------------------------------------------------------

    def assign_finger(
        self,
        fingertips: Dict[int, Tuple[float, float]],
        pressed_key_idx: int,
        hand: str,
        frame_idx: int = 0,
        onset_time: float = 0.0
    ) -> Optional[FingerAssignment]:
        """
        Assign a finger to a pressed key.

        Args:
            fingertips: Dict mapping finger number (1-5) to (x, y)
            pressed_key_idx: Key index (0-87)
            hand: 'left' or 'right'
        """
        if pressed_key_idx not in self.key_centers or not fingertips:
            return None

        key_center = self.key_centers[pressed_key_idx]

        # Compute raw probabilities
        probabilities: Dict[int, float] = {}
        for finger_num, pos in fingertips.items():
            if pos is not None:
                probabilities[finger_num] = self.compute_probability(pos, key_center)

        if not probabilities:
            return None

        # Normalise
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        best_finger = max(probabilities, key=probabilities.get)
        best_prob = probabilities[best_finger]

        if best_prob < self.min_confidence:
            return None

        return FingerAssignment(
            note_onset=onset_time,
            frame_idx=frame_idx,
            midi_pitch=pressed_key_idx + 21,
            key_idx=pressed_key_idx,
            assigned_finger=best_finger,
            hand=hand,
            confidence=best_prob,
            fingertip_position=fingertips[best_finger],
            all_probabilities=probabilities
        )

    # ------------------------------------------------------------------
    # Assignment from raw MediaPipe landmarks
    # ------------------------------------------------------------------

    def assign_from_landmarks(
        self,
        landmarks: np.ndarray,
        pressed_key_idx: int,
        hand: str,
        frame_idx: int = 0,
        onset_time: float = 0.0
    ) -> Optional[FingerAssignment]:
        """
        Assign a finger from a (21, 3) landmark array.
        """
        fingertips: Dict[int, Tuple[float, float]] = {}
        for finger_num, lm_idx in self.FINGERTIP_LANDMARKS.items():
            if lm_idx < len(landmarks) and not np.any(np.isnan(landmarks[lm_idx])):
                fingertips[finger_num] = (
                    float(landmarks[lm_idx, 0]),
                    float(landmarks[lm_idx, 1])
                )
        return self.assign_finger(
            fingertips, pressed_key_idx, hand, frame_idx, onset_time
        )

    # ------------------------------------------------------------------
    # Sequence assignment
    # ------------------------------------------------------------------

    def assign_sequence(
        self,
        midi_events: List[Dict],
        left_landmarks: np.ndarray,
        right_landmarks: np.ndarray,
        fps: float = 60.0,
        hand_separation_x: Optional[float] = None
    ) -> List[FingerAssignment]:
        """
        Assign fingers to a sequence of MIDI events.

        Args:
            midi_events: List of {'onset': float, 'pitch': int, …}
            left_landmarks:  (T, 21, 3)
            right_landmarks: (T, 21, 3)
            fps: Video frame rate
            hand_separation_x: x-coord for left/right split (auto → key 44)
        """
        assignments: List[FingerAssignment] = []

        for event in midi_events:
            onset_time = event.get('onset', 0)
            pitch = event.get('pitch', 60)
            key_idx = pitch - 21
            frame_idx = int(onset_time * fps)

            # Decide which hand
            if hand_separation_x is not None:
                key_x = self.key_centers.get(key_idx, (0, 0))[0]
                use_left = key_x < hand_separation_x
            else:
                use_left = key_idx < 44

            if use_left:
                if frame_idx >= len(left_landmarks):
                    continue
                landmarks = left_landmarks[frame_idx]
                hand = 'left'
            else:
                if frame_idx >= len(right_landmarks):
                    continue
                landmarks = right_landmarks[frame_idx]
                hand = 'right'

            if np.any(np.isnan(landmarks)):
                continue

            assignment = self.assign_from_landmarks(
                landmarks, key_idx, hand, frame_idx, onset_time
            )
            if assignment is not None:
                assignments.append(assignment)

        return assignments

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_candidate_keys(self, key_idx: int) -> List[int]:
        """Keys within ±candidate_range of *key_idx*."""
        return [
            k for k in range(key_idx - self.candidate_range,
                             key_idx + self.candidate_range + 1)
            if k in self.key_centers
        ]

    def compute_all_probabilities(
        self,
        fingertips: Dict[int, Tuple[float, float]],
        key_indices: List[int]
    ) -> Dict[int, Dict[int, float]]:
        """Probability matrix for all fingers × keys."""
        result: Dict[int, Dict[int, float]] = {}
        for key_idx in key_indices:
            if key_idx not in self.key_centers:
                continue
            kc = self.key_centers[key_idx]
            result[key_idx] = {
                fn: self.compute_probability(pos, kc)
                for fn, pos in fingertips.items() if pos is not None
            }
        return result
