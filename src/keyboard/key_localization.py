"""
Key Localization Module

Maps pixel coordinates to piano key indices using the black key pattern.

Usage:
    from src.keyboard.key_localization import KeyLocalizer
    
    localizer = KeyLocalizer(keyboard_region)
    key_idx = localizer.point_to_key(x, y)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class KeyInfo:
    """Information about a single piano key."""
    key_idx: int  # 0-87
    midi_pitch: int  # 21-108
    note_name: str  # e.g., "C4", "A#3"
    is_black: bool
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)


class KeyLocalizer:
    """
    Maps coordinates to piano keys and provides key information.
    
    Uses the keyboard region's key boundaries to determine
    which key corresponds to a given pixel position.
    """
    
    # Note names for each position in an octave
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Black key indices within octave
    BLACK_KEY_POSITIONS = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#
    
    NUM_KEYS = 88
    FIRST_MIDI_PITCH = 21  # A0
    
    def __init__(self, key_boundaries: Dict[int, Tuple[int, int, int, int]]):
        """
        Args:
            key_boundaries: Dictionary mapping key index to bounding box
        """
        self.key_boundaries = key_boundaries
        self._key_info_cache: Dict[int, KeyInfo] = {}
        self._build_key_info()
    
    def _build_key_info(self):
        """Pre-compute key information for all keys."""
        for key_idx, bbox in self.key_boundaries.items():
            midi_pitch = key_idx + self.FIRST_MIDI_PITCH
            note_in_octave = midi_pitch % 12
            octave = (midi_pitch // 12) - 1
            
            note_name = f"{self.NOTE_NAMES[note_in_octave]}{octave}"
            is_black = note_in_octave in self.BLACK_KEY_POSITIONS
            
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            self._key_info_cache[key_idx] = KeyInfo(
                key_idx=key_idx,
                midi_pitch=midi_pitch,
                note_name=note_name,
                is_black=is_black,
                bbox=bbox,
                center=center
            )
    
    def point_to_key(
        self, 
        x: float, 
        y: float,
        prefer_black: bool = True
    ) -> Optional[int]:
        """
        Find the key at a given point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            prefer_black: If True, prioritize black keys when point is in overlap region
            
        Returns:
            Key index (0-87) or None if no key at position
        """
        candidates = []
        
        for key_idx, (x1, y1, x2, y2) in self.key_boundaries.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                candidates.append(key_idx)
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Multiple candidates (overlap region between black and white keys)
        if prefer_black:
            # Return black key if present
            for key_idx in candidates:
                if self._key_info_cache[key_idx].is_black:
                    return key_idx
        
        # Return key with center closest to point
        min_dist = float('inf')
        best_key = candidates[0]
        
        for key_idx in candidates:
            cx, cy = self._key_info_cache[key_idx].center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_key = key_idx
        
        return best_key
    
    def get_key_info(self, key_idx: int) -> Optional[KeyInfo]:
        """Get information about a specific key."""
        return self._key_info_cache.get(key_idx)
    
    def get_key_by_midi(self, midi_pitch: int) -> Optional[KeyInfo]:
        """Get key information by MIDI pitch number."""
        key_idx = midi_pitch - self.FIRST_MIDI_PITCH
        return self.get_key_info(key_idx)
    
    def get_key_by_name(self, note_name: str) -> Optional[KeyInfo]:
        """Get key information by note name (e.g., 'C4', 'A#3')."""
        for key_info in self._key_info_cache.values():
            if key_info.note_name == note_name:
                return key_info
        return None
    
    def get_nearby_keys(
        self, 
        x: float, 
        y: float, 
        radius: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Get keys near a point with distances.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Number of keys to consider in each direction
            
        Returns:
            List of (key_idx, distance) tuples, sorted by distance
        """
        # First find the closest key
        closest_key = self.point_to_key(x, y)
        
        if closest_key is None:
            # Find key with x-coordinate closest to query point
            min_x_dist = float('inf')
            for key_idx, info in self._key_info_cache.items():
                x_dist = abs(info.center[0] - x)
                if x_dist < min_x_dist:
                    min_x_dist = x_dist
                    closest_key = key_idx
        
        if closest_key is None:
            return []
        
        # Get keys within radius
        nearby = []
        for key_idx in range(max(0, closest_key - radius), 
                            min(self.NUM_KEYS, closest_key + radius + 1)):
            if key_idx in self._key_info_cache:
                cx, cy = self._key_info_cache[key_idx].center
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                nearby.append((key_idx, dist))
        
        nearby.sort(key=lambda x: x[1])
        return nearby
    
    def get_white_keys(self) -> List[KeyInfo]:
        """Get all white keys."""
        return [info for info in self._key_info_cache.values() if not info.is_black]
    
    def get_black_keys(self) -> List[KeyInfo]:
        """Get all black keys."""
        return [info for info in self._key_info_cache.values() if info.is_black]
    
    def midi_to_key_center(self, midi_pitch: int) -> Optional[Tuple[float, float]]:
        """Get the center coordinates for a MIDI pitch."""
        key_info = self.get_key_by_midi(midi_pitch)
        return key_info.center if key_info else None
    
    def compute_key_distances(
        self, 
        point: Tuple[float, float]
    ) -> Dict[int, float]:
        """
        Compute distance from point to all key centers.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            Dictionary mapping key index to distance
        """
        x, y = point
        distances = {}
        
        for key_idx, info in self._key_info_cache.items():
            cx, cy = info.center
            distances[key_idx] = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        return distances

