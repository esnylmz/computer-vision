"""
Hand Separation Module

Separates left and right hand based on position and playing patterns.

Usage:
    from src.assignment.hand_separation import HandSeparator
    
    separator = HandSeparator()
    hand = separator.determine_hand(fingertip_position, key_position)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HandInfo:
    """Information about a detected hand."""
    hand_type: str  # 'left' or 'right'
    center_x: float
    center_y: float
    span: float  # Distance from thumb to pinky
    confidence: float


class HandSeparator:
    """
    Separates left and right hands in piano performance.
    
    Uses multiple cues:
    - Position relative to keyboard center
    - Hand orientation (thumb position relative to pinky)
    - Temporal consistency
    """
    
    def __init__(
        self,
        keyboard_center_x: Optional[float] = None,
        separation_margin: float = 50.0
    ):
        """
        Args:
            keyboard_center_x: X-coordinate of keyboard center
            separation_margin: Margin for ambiguous region
        """
        self.keyboard_center_x = keyboard_center_x
        self.separation_margin = separation_margin
        self._history: List[Tuple[str, float]] = []  # (hand, x_position)
    
    def set_keyboard_center(self, center_x: float):
        """Set keyboard center for hand separation."""
        self.keyboard_center_x = center_x
    
    def determine_hand_by_position(
        self,
        hand_center_x: float
    ) -> Tuple[str, float]:
        """
        Determine hand type by position.
        
        Args:
            hand_center_x: X-coordinate of hand center
            
        Returns:
            (hand_type, confidence) tuple
        """
        if self.keyboard_center_x is None:
            return ('unknown', 0.0)
        
        distance_from_center = hand_center_x - self.keyboard_center_x
        
        if abs(distance_from_center) < self.separation_margin:
            # Ambiguous region
            confidence = 0.5 + 0.5 * abs(distance_from_center) / self.separation_margin
            hand = 'right' if distance_from_center > 0 else 'left'
        else:
            confidence = 1.0
            hand = 'right' if distance_from_center > 0 else 'left'
        
        return (hand, confidence)
    
    def determine_hand_by_orientation(
        self,
        thumb_x: float,
        pinky_x: float
    ) -> Tuple[str, float]:
        """
        Determine hand type by thumb-pinky orientation.
        
        For right hand, thumb is typically to the left of pinky.
        For left hand, thumb is typically to the right of pinky.
        
        Args:
            thumb_x: X-coordinate of thumb
            pinky_x: X-coordinate of pinky
            
        Returns:
            (hand_type, confidence) tuple
        """
        thumb_pinky_diff = thumb_x - pinky_x
        
        # If thumb is to the left of pinky -> right hand
        # If thumb is to the right of pinky -> left hand
        if abs(thumb_pinky_diff) < 10:  # Very close, low confidence
            return ('unknown', 0.5)
        
        if thumb_pinky_diff < 0:
            return ('right', min(1.0, abs(thumb_pinky_diff) / 50))
        else:
            return ('left', min(1.0, abs(thumb_pinky_diff) / 50))
    
    def determine_hand(
        self,
        fingertips: Dict[int, Tuple[float, float]],
        use_history: bool = True
    ) -> HandInfo:
        """
        Determine hand type using multiple cues.
        
        Args:
            fingertips: Dict mapping finger number (1-5) to (x, y) position
            use_history: Whether to use temporal consistency
            
        Returns:
            HandInfo object
        """
        if not fingertips:
            return HandInfo('unknown', 0, 0, 0, 0)
        
        # Compute hand center
        positions = list(fingertips.values())
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        # Compute span
        if 1 in fingertips and 5 in fingertips:
            span = np.sqrt(
                (fingertips[1][0] - fingertips[5][0])**2 +
                (fingertips[1][1] - fingertips[5][1])**2
            )
        else:
            span = 0
        
        # Position-based determination
        pos_hand, pos_conf = self.determine_hand_by_position(center_x)
        
        # Orientation-based determination
        if 1 in fingertips and 5 in fingertips:
            orient_hand, orient_conf = self.determine_hand_by_orientation(
                fingertips[1][0], fingertips[5][0]
            )
        else:
            orient_hand, orient_conf = 'unknown', 0
        
        # Combine cues
        if pos_hand == orient_hand and pos_hand != 'unknown':
            hand = pos_hand
            confidence = (pos_conf + orient_conf) / 2
        elif pos_conf > orient_conf:
            hand = pos_hand
            confidence = pos_conf
        else:
            hand = orient_hand
            confidence = orient_conf
        
        # Use history for temporal consistency
        if use_history and self._history:
            recent = self._history[-5:]
            same_hand_count = sum(1 for h, x in recent 
                                 if h == hand and abs(x - center_x) < 100)
            if same_hand_count >= 3:
                confidence = min(1.0, confidence + 0.2)
        
        # Update history
        self._history.append((hand, center_x))
        if len(self._history) > 30:
            self._history = self._history[-30:]
        
        return HandInfo(
            hand_type=hand,
            center_x=center_x,
            center_y=center_y,
            span=span,
            confidence=confidence
        )
    
    def separate_hands(
        self,
        left_landmarks: np.ndarray,
        right_landmarks: np.ndarray,
        frame_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Ensure hands are correctly labeled as left/right.
        
        Some skeleton extractors may mislabel hands. This function
        checks and swaps if necessary.
        
        Args:
            left_landmarks: Landmarks labeled as left hand
            right_landmarks: Landmarks labeled as right hand
            frame_idx: Frame index
            
        Returns:
            (corrected_left, corrected_right) landmarks
        """
        left_valid = left_landmarks is not None and not np.all(np.isnan(left_landmarks))
        right_valid = right_landmarks is not None and not np.all(np.isnan(right_landmarks))
        
        if not left_valid and not right_valid:
            return None, None
        
        if left_valid and not right_valid:
            # Only one hand - determine which it is
            hand_info = self._determine_single_hand(left_landmarks)
            if hand_info.hand_type == 'right':
                return None, left_landmarks
            return left_landmarks, None
        
        if right_valid and not left_valid:
            hand_info = self._determine_single_hand(right_landmarks)
            if hand_info.hand_type == 'left':
                return right_landmarks, None
            return None, right_landmarks
        
        # Both hands present - check if they need swapping
        left_center_x = np.nanmean(left_landmarks[:, 0])
        right_center_x = np.nanmean(right_landmarks[:, 0])
        
        if left_center_x > right_center_x:
            # Hands are swapped
            return right_landmarks, left_landmarks
        
        return left_landmarks, right_landmarks
    
    def _determine_single_hand(self, landmarks: np.ndarray) -> HandInfo:
        """Determine hand type from single hand landmarks."""
        fingertips = {}
        tip_indices = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
        
        for finger, idx in tip_indices.items():
            if not np.any(np.isnan(landmarks[idx])):
                fingertips[finger] = (landmarks[idx, 0], landmarks[idx, 1])
        
        return self.determine_hand(fingertips, use_history=False)
    
    def determine_hand_for_key(
        self,
        key_idx: int,
        keyboard_width: float,
        split_key: int = 44
    ) -> str:
        """
        Determine which hand typically plays a key.
        
        This is a heuristic based on keyboard position.
        Real hand assignment depends on the specific piece.
        
        Args:
            key_idx: Key index (0-87)
            keyboard_width: Total keyboard width in pixels
            split_key: Key index for hand split point
            
        Returns:
            'left' or 'right'
        """
        return 'left' if key_idx < split_key else 'right'
    
    def clear_history(self):
        """Clear temporal history."""
        self._history = []

