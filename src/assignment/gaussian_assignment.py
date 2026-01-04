"""
Gaussian Probability-Based Finger Assignment

Assigns fingers to pressed piano keys using Gaussian probability distributions
centered on fingertip positions.

Based on Moryossef et al. (2023) "At Your Fingertips" methodology.

Usage:
    from src.assignment.gaussian_assignment import GaussianFingerAssigner
    
    assigner = GaussianFingerAssigner(key_regions, sigma=15.0)
    finger = assigner.assign(fingertip_positions, pressed_key)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FingerAssignment:
    """Single finger assignment result."""
    note_onset: float  # Onset time in seconds
    frame_idx: int  # Frame index
    midi_pitch: int  # MIDI note number (21-108)
    key_idx: int  # Piano key index (0-87)
    assigned_finger: int  # 1-5 (thumb to pinky)
    hand: str  # 'left' or 'right'
    confidence: float  # Assignment probability
    fingertip_position: Tuple[float, float]  # (x, y) of assigned fingertip
    all_probabilities: Optional[Dict[int, float]] = None  # Probabilities for all fingers
    
    @property
    def finger_name(self) -> str:
        """Get finger name."""
        names = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'}
        return names.get(self.assigned_finger, 'unknown')
    
    @property
    def label(self) -> str:
        """Get label in format 'R3' or 'L1'."""
        hand_prefix = 'R' if self.hand == 'right' else 'L'
        return f"{hand_prefix}{self.assigned_finger}"


class GaussianFingerAssigner:
    """
    Assigns fingers to piano keys using Gaussian probability model.
    
    For each pressed key, computes probability distribution over all
    fingertips based on distance, then selects maximum likelihood finger.
    """
    
    # Finger indices in MediaPipe landmark order
    FINGERTIP_INDICES = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'pinky': 20
    }
    
    # Finger numbers (standard piano notation)
    FINGER_NUMBERS = {
        4: 1,   # Thumb
        8: 2,   # Index
        12: 3,  # Middle
        16: 4,  # Ring
        20: 5   # Pinky
    }
    
    # Inverse mapping
    LANDMARK_FROM_FINGER = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
    
    def __init__(
        self,
        key_boundaries: Dict[int, Tuple[int, int, int, int]],
        sigma: float = 15.0,
        candidate_range: int = 2,
        min_confidence: float = 0.01
    ):
        """
        Args:
            key_boundaries: Dict mapping key index to (x1, y1, x2, y2) bbox
            sigma: Standard deviation for Gaussian (pixels)
            candidate_range: Number of adjacent keys to consider (±N)
            min_confidence: Minimum confidence for valid assignment
        """
        self.key_boundaries = key_boundaries
        self.sigma = sigma
        self.candidate_range = candidate_range
        self.min_confidence = min_confidence
        
        # Precompute key centers
        self.key_centers = {}
        for key_idx, (x1, y1, x2, y2) in key_boundaries.items():
            self.key_centers[key_idx] = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def compute_probability(
        self, 
        fingertip: Tuple[float, float],
        key_center: Tuple[float, float]
    ) -> float:
        """
        Compute Gaussian probability of finger pressing key.
        
        Args:
            fingertip: (x, y) fingertip position
            key_center: (x, y) key center position
            
        Returns:
            Probability value
        """
        distance = np.sqrt(
            (fingertip[0] - key_center[0])**2 + 
            (fingertip[1] - key_center[1])**2
        )
        return np.exp(-distance**2 / (2 * self.sigma**2))
    
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
            fingertips: Dict mapping finger number (1-5) to (x, y) position
            pressed_key_idx: Key index (0-87) of pressed key
            hand: 'left' or 'right'
            frame_idx: Frame index
            onset_time: Note onset time
            
        Returns:
            FingerAssignment or None if assignment fails
        """
        if pressed_key_idx not in self.key_centers:
            return None
        
        if not fingertips:
            return None
        
        key_center = self.key_centers[pressed_key_idx]
        
        # Compute probability for each fingertip
        probabilities = {}
        for finger_num, pos in fingertips.items():
            if pos is not None:
                prob = self.compute_probability(pos, key_center)
                probabilities[finger_num] = prob
        
        if not probabilities:
            return None
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        # Select finger with highest probability
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
    
    def assign_from_landmarks(
        self,
        landmarks: np.ndarray,
        pressed_key_idx: int,
        hand: str,
        frame_idx: int = 0,
        onset_time: float = 0.0
    ) -> Optional[FingerAssignment]:
        """
        Assign finger from raw MediaPipe landmarks.
        
        Args:
            landmarks: Shape (21, 3) landmark array
            pressed_key_idx: Key index of pressed key
            hand: 'left' or 'right'
            frame_idx: Frame index
            onset_time: Note onset time
            
        Returns:
            FingerAssignment or None
        """
        # Extract fingertips from landmarks
        fingertips = {}
        for finger_num, lm_idx in self.LANDMARK_FROM_FINGER.items():
            if lm_idx < len(landmarks) and not np.any(np.isnan(landmarks[lm_idx])):
                fingertips[finger_num] = (float(landmarks[lm_idx, 0]), 
                                         float(landmarks[lm_idx, 1]))
        
        return self.assign_finger(
            fingertips, pressed_key_idx, hand, frame_idx, onset_time
        )
    
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
            midi_events: List of {'onset': float, 'pitch': int, 'velocity': int}
            left_landmarks: Shape (T, 21, 3) left hand landmarks
            right_landmarks: Shape (T, 21, 3) right hand landmarks
            fps: Video frame rate
            hand_separation_x: X-coordinate for hand separation (auto if None)
            
        Returns:
            List of FingerAssignment for each MIDI event
        """
        assignments = []
        
        for event in midi_events:
            onset_time = event.get('onset', 0)
            pitch = event.get('pitch', 60)
            key_idx = pitch - 21
            
            # Convert time to frame index
            frame_idx = int(onset_time * fps)
            
            # Determine which hand to use
            if hand_separation_x is not None:
                key_x = self.key_centers.get(key_idx, (0, 0))[0]
                use_left = key_x < hand_separation_x
            else:
                # Use left hand for lower half of keyboard
                use_left = key_idx < 44
            
            # Get landmarks for appropriate hand
            if use_left:
                if frame_idx < len(left_landmarks):
                    landmarks = left_landmarks[frame_idx]
                    hand = 'left'
                else:
                    continue
            else:
                if frame_idx < len(right_landmarks):
                    landmarks = right_landmarks[frame_idx]
                    hand = 'right'
                else:
                    continue
            
            # Skip if landmarks are invalid
            if np.any(np.isnan(landmarks)):
                continue
            
            # Assign finger
            assignment = self.assign_from_landmarks(
                landmarks, key_idx, hand, frame_idx, onset_time
            )
            
            if assignment is not None:
                assignments.append(assignment)
        
        return assignments
    
    def get_candidate_keys(self, key_idx: int) -> List[int]:
        """Get list of candidate keys around the target key."""
        candidates = []
        for offset in range(-self.candidate_range, self.candidate_range + 1):
            candidate = key_idx + offset
            if candidate in self.key_centers:
                candidates.append(candidate)
        return candidates
    
    def compute_all_probabilities(
        self,
        fingertips: Dict[int, Tuple[float, float]],
        key_indices: List[int]
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute probability matrix for all fingers and keys.
        
        Args:
            fingertips: Dict mapping finger number to position
            key_indices: List of key indices to consider
            
        Returns:
            Dict mapping key_idx to {finger_num: probability}
        """
        result = {}
        
        for key_idx in key_indices:
            if key_idx not in self.key_centers:
                continue
            
            key_center = self.key_centers[key_idx]
            probs = {}
            
            for finger_num, pos in fingertips.items():
                if pos is not None:
                    probs[finger_num] = self.compute_probability(pos, key_center)
            
            result[key_idx] = probs
        
        return result
    
    def adjust_sigma(
        self, 
        hand_span: float,
        reference_span: float = 150.0
    ):
        """
        Adjust sigma based on observed hand span.
        
        Args:
            hand_span: Observed hand span in pixels
            reference_span: Reference span for default sigma
        """
        scale = hand_span / reference_span
        self.sigma = self.sigma * scale

