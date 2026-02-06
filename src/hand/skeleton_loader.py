"""
Hand Skeleton Loader

Loads and parses MediaPipe hand landmark data from JSON files.

MediaPipe 21-Keypoint Structure:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)

Usage:
    from src.hand.skeleton_loader import SkeletonLoader
    
    loader = SkeletonLoader()
    landmarks = loader.load("path/to/skeleton.json")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class HandLandmarks:
    """Hand landmark data for a single frame."""
    landmarks: np.ndarray  # Shape (21, 3) - 21 landmarks, (x, y, z)
    hand_type: str  # 'left' or 'right'
    confidence: float  # Detection confidence
    frame_idx: int  # Frame index
    
    @property
    def wrist(self) -> np.ndarray:
        """Get wrist position."""
        return self.landmarks[0]
    
    @property
    def thumb_tip(self) -> np.ndarray:
        """Get thumb tip position."""
        return self.landmarks[4]
    
    @property
    def index_tip(self) -> np.ndarray:
        """Get index finger tip position."""
        return self.landmarks[8]
    
    @property
    def middle_tip(self) -> np.ndarray:
        """Get middle finger tip position."""
        return self.landmarks[12]
    
    @property
    def ring_tip(self) -> np.ndarray:
        """Get ring finger tip position."""
        return self.landmarks[16]
    
    @property
    def pinky_tip(self) -> np.ndarray:
        """Get pinky tip position."""
        return self.landmarks[20]
    
    @property
    def fingertips(self) -> np.ndarray:
        """Get all fingertip positions. Shape (5, 3)."""
        return self.landmarks[[4, 8, 12, 16, 20]]
    
    def get_finger_tip(self, finger: int) -> np.ndarray:
        """
        Get fingertip by finger number (1-5).
        
        Args:
            finger: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
            
        Returns:
            (x, y, z) coordinates
        """
        tip_indices = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}
        return self.landmarks[tip_indices[finger]]


class SkeletonLoader:
    """
    Loads hand skeleton data from various formats.
    
    Supports:
    - PianoVAM JSON format
    - MediaPipe JSON format
    - Custom formats
    """
    
    # MediaPipe landmark indices
    LANDMARK_NAMES = [
        'wrist',
        'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
        'index_mcp', 'index_pip', 'index_dip', 'index_tip',
        'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
        'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
        'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
    ]
    
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]
    
    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: Whether to normalize coordinates to [0, 1]
        """
        self.normalize = normalize
    
    def load(
        self, 
        path: Union[str, Path]
    ) -> Dict[str, List[HandLandmarks]]:
        """
        Load skeleton data from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Dict with 'left' and 'right' keys, each containing
            list of HandLandmarks for each frame
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return self._parse_json(data)
    
    def _parse_json(self, data: Union[Dict, List]) -> Dict[str, List[HandLandmarks]]:
        """Parse JSON data into HandLandmarks objects."""
        result = {'left': [], 'right': []}
        
        # Handle different JSON formats
        if isinstance(data, list):
            # List of frames
            for frame_idx, frame_data in enumerate(data):
                self._parse_frame(frame_data, frame_idx, result)
        elif isinstance(data, dict):
            # Check for different key formats
            if 'frames' in data:
                for frame_idx, frame_data in enumerate(data['frames']):
                    self._parse_frame(frame_data, frame_idx, result)
            elif 'left_hand' in data or 'right_hand' in data:
                # Single frame or aggregated data
                self._parse_frame(data, 0, result)
            else:
                # Try parsing as frame-indexed dict
                for frame_idx_str, frame_data in data.items():
                    try:
                        frame_idx = int(frame_idx_str)
                        self._parse_frame(frame_data, frame_idx, result)
                    except ValueError:
                        continue
        
        return result
    
    def _parse_frame(
        self, 
        frame_data: Dict, 
        frame_idx: int,
        result: Dict[str, List[HandLandmarks]]
    ):
        """Parse a single frame's hand data."""
        for hand_type in ['left', 'right']:
            hand_key = f'{hand_type}_hand'          # e.g. 'left_hand'
            alt_key = hand_type                      # e.g. 'left'
            cap_key = hand_type.capitalize()         # e.g. 'Left'  (PianoVAM format)
            
            hand_data = (frame_data.get(hand_key)
                         or frame_data.get(alt_key)
                         or frame_data.get(cap_key))
            
            if hand_data is None:
                continue
            
            landmarks = self._parse_landmarks(hand_data)
            
            if landmarks is not None:
                confidence = hand_data.get('confidence', 1.0) if isinstance(hand_data, dict) else 1.0
                
                result[hand_type].append(HandLandmarks(
                    landmarks=landmarks,
                    hand_type=hand_type,
                    confidence=confidence,
                    frame_idx=frame_idx
                ))
    
    def _parse_landmarks(self, hand_data: Union[Dict, List]) -> Optional[np.ndarray]:
        """Parse landmarks from hand data."""
        landmarks = np.zeros((21, 3))
        
        if isinstance(hand_data, dict):
            # Dict with 'landmarks' or 'keypoints' key
            lm_data = hand_data.get('landmarks', hand_data.get('keypoints', hand_data))
            
            if isinstance(lm_data, list):
                for i, lm in enumerate(lm_data[:21]):
                    if isinstance(lm, dict):
                        landmarks[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                    elif isinstance(lm, (list, tuple)):
                        landmarks[i] = lm[:3] if len(lm) >= 3 else list(lm) + [0] * (3 - len(lm))
            elif isinstance(lm_data, dict):
                for i, name in enumerate(self.LANDMARK_NAMES):
                    if name in lm_data:
                        lm = lm_data[name]
                        if isinstance(lm, dict):
                            landmarks[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                        elif isinstance(lm, (list, tuple)):
                            landmarks[i] = lm[:3]
        
        elif isinstance(hand_data, list):
            for i, lm in enumerate(hand_data[:21]):
                if isinstance(lm, dict):
                    landmarks[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                elif isinstance(lm, (list, tuple)):
                    landmarks[i] = lm[:3] if len(lm) >= 3 else list(lm) + [0] * (3 - len(lm))
        
        # Check if landmarks are valid (not all zeros)
        if np.allclose(landmarks, 0):
            return None
        
        if self.normalize:
            landmarks = self._normalize_landmarks(landmarks)
        
        return landmarks
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to [0, 1] range."""
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        return (landmarks - min_vals) / range_vals
    
    def to_array(
        self, 
        landmarks_list: List[HandLandmarks],
        fill_missing: bool = True,
        total_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert list of HandLandmarks to numpy array.
        
        Args:
            landmarks_list: List of HandLandmarks
            fill_missing: Fill missing frames with NaN
            total_frames: Total number of frames (for filling)
            
        Returns:
            Array of shape (T, 21, 3) where T is number of frames
        """
        if not landmarks_list:
            if total_frames:
                return np.full((total_frames, 21, 3), np.nan)
            return np.zeros((0, 21, 3))
        
        if total_frames is None:
            total_frames = max(lm.frame_idx for lm in landmarks_list) + 1
        
        result = np.full((total_frames, 21, 3), np.nan) if fill_missing else None
        
        if not fill_missing:
            result = np.zeros((len(landmarks_list), 21, 3))
            for i, lm in enumerate(landmarks_list):
                result[i] = lm.landmarks
        else:
            for lm in landmarks_list:
                if lm.frame_idx < total_frames:
                    result[lm.frame_idx] = lm.landmarks
        
        return result
    
    def get_frame_indices(self, landmarks_list: List[HandLandmarks]) -> List[int]:
        """Get list of frame indices with valid landmarks."""
        return sorted(set(lm.frame_idx for lm in landmarks_list))

