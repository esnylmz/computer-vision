"""
Fingertip Position Extractor

Extracts and processes fingertip positions from hand landmarks.

Usage:
    from src.hand.fingertip_extractor import FingertipExtractor
    
    extractor = FingertipExtractor()
    fingertips = extractor.extract(landmarks)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FingertipData:
    """Fingertip data for a single frame."""
    positions: Dict[int, Tuple[float, float, float]]  # finger_num -> (x, y, z)
    hand_type: str  # 'left' or 'right'
    frame_idx: int
    
    def get_position_2d(self, finger: int) -> Optional[Tuple[float, float]]:
        """Get 2D position (x, y) for a finger."""
        if finger in self.positions:
            x, y, z = self.positions[finger]
            return (x, y)
        return None
    
    def to_array(self) -> np.ndarray:
        """Convert to array of shape (5, 3)."""
        arr = np.zeros((5, 3))
        for finger in range(1, 6):
            if finger in self.positions:
                arr[finger - 1] = self.positions[finger]
        return arr


class FingertipExtractor:
    """
    Extracts fingertip positions from MediaPipe hand landmarks.
    
    MediaPipe fingertip indices:
        4: Thumb tip
        8: Index tip
        12: Middle tip
        16: Ring tip
        20: Pinky tip
    """
    
    # Mapping from finger number to MediaPipe landmark index
    FINGERTIP_INDICES = {
        1: 4,   # Thumb
        2: 8,   # Index
        3: 12,  # Middle
        4: 16,  # Ring
        5: 20   # Pinky
    }
    
    # Finger names
    FINGER_NAMES = {
        1: 'thumb',
        2: 'index',
        3: 'middle',
        4: 'ring',
        5: 'pinky'
    }
    
    def __init__(
        self, 
        use_z: bool = False,
        transform_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            use_z: Whether to use z-coordinate (depth)
            transform_matrix: Optional 3x3 transformation matrix
        """
        self.use_z = use_z
        self.transform_matrix = transform_matrix
    
    def extract(
        self, 
        landmarks: np.ndarray,
        frame_idx: int = 0,
        hand_type: str = 'right'
    ) -> FingertipData:
        """
        Extract fingertip positions from landmarks.
        
        Args:
            landmarks: Shape (21, 3) landmark array
            frame_idx: Frame index
            hand_type: 'left' or 'right'
            
        Returns:
            FingertipData object
        """
        positions = {}
        
        for finger_num, lm_idx in self.FINGERTIP_INDICES.items():
            if lm_idx < len(landmarks):
                pos = landmarks[lm_idx]
                
                if self.transform_matrix is not None:
                    pos = self._transform_point(pos)
                
                if not self.use_z:
                    positions[finger_num] = (float(pos[0]), float(pos[1]), 0.0)
                else:
                    positions[finger_num] = tuple(float(p) for p in pos[:3])
        
        return FingertipData(
            positions=positions,
            hand_type=hand_type,
            frame_idx=frame_idx
        )
    
    def extract_sequence(
        self, 
        landmarks_sequence: np.ndarray,
        hand_type: str = 'right'
    ) -> List[FingertipData]:
        """
        Extract fingertips from a sequence of landmarks.
        
        Args:
            landmarks_sequence: Shape (T, 21, 3)
            hand_type: 'left' or 'right'
            
        Returns:
            List of FingertipData, one per frame
        """
        results = []
        
        for frame_idx in range(len(landmarks_sequence)):
            if not np.any(np.isnan(landmarks_sequence[frame_idx])):
                results.append(self.extract(
                    landmarks_sequence[frame_idx],
                    frame_idx=frame_idx,
                    hand_type=hand_type
                ))
        
        return results
    
    def _transform_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to point."""
        if len(point) == 2:
            point = np.array([point[0], point[1], 1])
        elif len(point) == 3:
            point = np.array([point[0], point[1], 1])
        
        transformed = self.transform_matrix @ point
        return transformed[:2] / transformed[2] if transformed[2] != 0 else transformed[:2]
    
    def get_fingertip_at_frame(
        self, 
        landmarks_sequence: np.ndarray,
        frame_idx: int,
        finger: int
    ) -> Optional[Tuple[float, float]]:
        """
        Get single fingertip position at a specific frame.
        
        Args:
            landmarks_sequence: Shape (T, 21, 3)
            frame_idx: Frame index
            finger: Finger number (1-5)
            
        Returns:
            (x, y) tuple or None
        """
        if frame_idx >= len(landmarks_sequence):
            return None
        
        lm_idx = self.FINGERTIP_INDICES.get(finger)
        if lm_idx is None:
            return None
        
        pos = landmarks_sequence[frame_idx, lm_idx, :2]
        
        if np.any(np.isnan(pos)):
            return None
        
        return (float(pos[0]), float(pos[1]))
    
    def get_all_fingertips_at_frame(
        self, 
        landmarks_sequence: np.ndarray,
        frame_idx: int
    ) -> Dict[int, Tuple[float, float]]:
        """
        Get all fingertip positions at a specific frame.
        
        Args:
            landmarks_sequence: Shape (T, 21, 3)
            frame_idx: Frame index
            
        Returns:
            Dict mapping finger number to (x, y) position
        """
        result = {}
        
        for finger in range(1, 6):
            pos = self.get_fingertip_at_frame(landmarks_sequence, frame_idx, finger)
            if pos is not None:
                result[finger] = pos
        
        return result
    
    def compute_finger_distances(
        self, 
        fingertips: FingertipData
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute distances between all pairs of fingertips.
        
        Returns:
            Dict mapping (finger1, finger2) to distance
        """
        distances = {}
        positions = fingertips.to_array()[:, :2]  # Use only x, y
        
        for i in range(5):
            for j in range(i + 1, 5):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[(i + 1, j + 1)] = float(dist)
        
        return distances
    
    def compute_hand_span(self, fingertips: FingertipData) -> float:
        """
        Compute hand span (distance from thumb to pinky).
        
        Args:
            fingertips: FingertipData object
            
        Returns:
            Distance in pixels
        """
        thumb_pos = fingertips.get_position_2d(1)
        pinky_pos = fingertips.get_position_2d(5)
        
        if thumb_pos is None or pinky_pos is None:
            return 0.0
        
        return float(np.linalg.norm(
            np.array(thumb_pos) - np.array(pinky_pos)
        ))
    
    def estimate_hand_center(self, fingertips: FingertipData) -> Tuple[float, float]:
        """
        Estimate hand center from fingertip positions.
        
        Returns:
            (x, y) center position
        """
        positions = []
        for finger in range(1, 6):
            pos = fingertips.get_position_2d(finger)
            if pos is not None:
                positions.append(pos)
        
        if not positions:
            return (0.0, 0.0)
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        
        return (float(center[0]), float(center[1]))
    
    def is_finger_pressing(
        self, 
        fingertips: FingertipData,
        finger: int,
        keyboard_y_threshold: float
    ) -> bool:
        """
        Check if a finger is in pressing position.
        
        Args:
            fingertips: FingertipData object
            finger: Finger number (1-5)
            keyboard_y_threshold: Y-coordinate threshold for pressing
            
        Returns:
            True if finger appears to be pressing
        """
        pos = fingertips.get_position_2d(finger)
        if pos is None:
            return False
        
        return pos[1] >= keyboard_y_threshold
    
    def get_pressing_fingers(
        self, 
        fingertips: FingertipData,
        keyboard_y_threshold: float
    ) -> List[int]:
        """
        Get list of fingers that appear to be pressing.
        
        Args:
            fingertips: FingertipData object
            keyboard_y_threshold: Y-coordinate threshold
            
        Returns:
            List of finger numbers (1-5) that are pressing
        """
        pressing = []
        for finger in range(1, 6):
            if self.is_finger_pressing(fingertips, finger, keyboard_y_threshold):
                pressing.append(finger)
        return pressing

