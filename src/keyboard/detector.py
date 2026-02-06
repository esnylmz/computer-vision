"""
Keyboard Detection Module

Detects piano keyboard boundaries and localizes individual keys
using computer vision techniques.

Based on:
- Akbari & Cheng (2015) - Hough Transform approach
- Moryossef et al. (2023) - Black key pattern identification

Usage:
    from src.keyboard.detector import KeyboardDetector
    
    detector = KeyboardDetector()
    key_regions = detector.detect(frame)
    # key_regions: Dict[int, Tuple[x1, y1, x2, y2]] for keys 0-87
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class KeyboardRegion:
    """Detected keyboard region with key boundaries."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    homography: np.ndarray  # 3x3 transformation matrix
    key_boundaries: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    white_key_width: float = 0.0
    black_key_width: float = 0.0
    corners: Optional[Dict[str, Tuple[int, int]]] = None
    
    def get_key_center(self, key_idx: int) -> Optional[Tuple[float, float]]:
        """Get center point of a key."""
        if key_idx in self.key_boundaries:
            x1, y1, x2, y2 = self.key_boundaries[key_idx]
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None
    
    def midi_to_key_idx(self, midi_pitch: int) -> int:
        """Convert MIDI pitch (21-108) to key index (0-87)."""
        return midi_pitch - 21
    
    def key_idx_to_midi(self, key_idx: int) -> int:
        """Convert key index (0-87) to MIDI pitch (21-108)."""
        return key_idx + 21


class KeyboardDetector:
    """
    Detects piano keyboard and localizes 88 keys.
    
    Pipeline:
    1. Edge detection (Canny)
    2. Line detection (Hough Transform)
    3. Keyboard boundary extraction
    4. Black/white key separation
    5. Note identification via 2-3 black key pattern
    6. Homography computation for normalization
    """
    
    # Piano key layout: 88 keys, starting from A0
    # Pattern of black keys in one octave (C to B): 1=black, 0=white
    BLACK_KEY_PATTERN = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # C C# D D# E F F# G G# A A# B
    NUM_KEYS = 88
    NUM_WHITE_KEYS = 52
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize keyboard detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.hough_threshold = self.config.get('hough_threshold', 100)
        self.min_line_length = self.config.get('min_line_length', 100)
        
    def detect(self, frame: np.ndarray) -> Optional[KeyboardRegion]:
        """
        Detect keyboard in a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            KeyboardRegion with key boundaries, or None if detection fails
        """
        # Step 1: Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 2: Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Step 3: Find keyboard boundaries using Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=10
        )
        
        if lines is None:
            return None
            
        # Step 4: Extract horizontal lines (keyboard top/bottom)
        horizontal_lines = self._filter_horizontal_lines(lines)
        
        if len(horizontal_lines) < 2:
            return None
        
        # Step 5: Find keyboard bounding box
        bbox = self._find_keyboard_bbox(horizontal_lines, frame.shape)
        
        if bbox is None:
            return None
            
        # Step 6: Compute homography for normalization
        homography = self._compute_homography_from_bbox(bbox, frame.shape)
        
        # Step 7: Localize individual keys
        key_boundaries = self._localize_keys(frame, bbox)
        
        return KeyboardRegion(
            bbox=bbox,
            homography=homography,
            key_boundaries=key_boundaries,
            white_key_width=self._estimate_white_key_width(bbox),
            black_key_width=self._estimate_black_key_width(bbox)
        )
    
    def detect_from_corners(
        self, 
        corners: Dict[str, str],
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> KeyboardRegion:
        """
        Create keyboard region from PianoVAM corner annotations.
        
        This is the preferred method when corner annotations are available,
        as it's more reliable than automatic detection.
        
        Args:
            corners: Dict with 'LT', 'RT', 'RB', 'LB' corner coordinates
                     Format: "x, y" strings or (x, y) tuples
            frame_shape: Optional (height, width) for validation
        
        Returns:
            KeyboardRegion computed from given corners
        """
        # Parse corner coordinates
        def parse_corner(value):
            if isinstance(value, str):
                x, y = value.split(', ')
                return (int(float(x)), int(float(y)))
            elif isinstance(value, (list, tuple)):
                return (int(value[0]), int(value[1]))
            return value
        
        lt = parse_corner(corners.get('LT', corners.get('Point_LT', '0, 0')))
        rt = parse_corner(corners.get('RT', corners.get('Point_RT', '1920, 0')))
        rb = parse_corner(corners.get('RB', corners.get('Point_RB', '1920, 200')))
        lb = parse_corner(corners.get('LB', corners.get('Point_LB', '0, 200')))
        
        # Compute homography from corners to normalized rectangle
        src_pts = np.float32([lt, rt, rb, lb])
        
        # Target: normalized rectangle based on keyboard aspect ratio
        width = int(np.linalg.norm(np.array(rt) - np.array(lt)))
        height = int(np.linalg.norm(np.array(lb) - np.array(lt)))
        
        # Ensure minimum dimensions
        width = max(width, 800)
        height = max(height, 100)
        
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Compute key boundaries in normalized space
        key_boundaries = self._compute_key_boundaries_from_width(width, height)
        
        # Bounding box from corners
        all_x = [lt[0], rt[0], rb[0], lb[0]]
        all_y = [lt[1], rt[1], rb[1], lb[1]]
        bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        
        return KeyboardRegion(
            bbox=bbox,
            homography=homography,
            key_boundaries=key_boundaries,
            white_key_width=width / self.NUM_WHITE_KEYS,
            black_key_width=(width / self.NUM_WHITE_KEYS) * 0.6,
            corners={'LT': lt, 'RT': rt, 'RB': rb, 'LB': lb}
        )
    
    def _filter_horizontal_lines(self, lines: np.ndarray) -> List[np.ndarray]:
        """Filter lines to keep only horizontal ones (keyboard edges)."""
        horizontal = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < np.pi / 18:  # Within 10 degrees of horizontal
                horizontal.append(line[0])
        return horizontal
    
    def _find_keyboard_bbox(
        self, 
        lines: List[np.ndarray], 
        shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find keyboard bounding box from horizontal lines."""
        if not lines:
            return None
        
        # Get y-coordinates of all horizontal lines
        y_coords = []
        x_min, x_max = shape[1], 0
        
        for x1, y1, x2, y2 in lines:
            y_coords.extend([y1, y2])
            x_min = min(x_min, x1, x2)
            x_max = max(x_max, x1, x2)
        
        if not y_coords:
            return None
        
        # Find top and bottom edges
        y_coords = sorted(y_coords)
        y_top = y_coords[0]
        y_bottom = y_coords[-1]
        
        # Ensure reasonable aspect ratio for piano keyboard
        height = y_bottom - y_top
        width = x_max - x_min
        
        if height < 20 or width < 100:
            return None
        
        return (x_min, y_top, x_max, y_bottom)
    
    def _compute_homography_from_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Compute homography matrix from bounding box."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        src_pts = np.float32([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ])
        dst_pts = np.float32([
            [0, 0], [width, 0],
            [width, height], [0, height]
        ])
        
        return cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    def _localize_keys(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """Localize all 88 keys within detected keyboard region."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        return self._compute_key_boundaries_from_width(width, height, offset=(x1, y1))
    
    def _compute_key_boundaries_from_width(
        self, 
        width: int, 
        height: int,
        offset: Tuple[int, int] = (0, 0)
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Compute key boundaries assuming evenly spaced white keys.
        
        The 88-key piano starts at A0 (MIDI 21) and ends at C8 (MIDI 108).
        Key index 0 = A0, Key index 87 = C8
        """
        key_boundaries = {}
        white_key_width = width / self.NUM_WHITE_KEYS
        black_key_width = white_key_width * 0.6
        black_key_height = height * 0.6
        
        ox, oy = offset
        
        # Map from key index (0-87) to white key position
        # A0 is key 0, first note is A (no black key before it in that partial octave)
        white_key_idx = 0
        
        for key_idx in range(self.NUM_KEYS):
            # Convert key index to note within octave
            # Key 0 = A0, so we offset by 9 (A is 9 semitones from C)
            midi_pitch = key_idx + 21  # A0 = MIDI 21
            note_in_octave = midi_pitch % 12
            
            # Check if this is a black or white key
            # Black keys: C#, D#, F#, G#, A# (indices 1, 3, 6, 8, 10)
            is_black = note_in_octave in [1, 3, 6, 8, 10]
            
            if is_black:
                # Black key positioned between white keys
                # It sits on top of the previous white key's right side
                x_center = white_key_idx * white_key_width
                bx1 = int(ox + x_center - black_key_width / 2)
                bx2 = int(ox + x_center + black_key_width / 2)
                by1 = int(oy)
                by2 = int(oy + black_key_height)
                key_boundaries[key_idx] = (bx1, by1, bx2, by2)
            else:
                # White key
                wx1 = int(ox + white_key_idx * white_key_width)
                wx2 = int(ox + (white_key_idx + 1) * white_key_width)
                wy1 = int(oy)
                wy2 = int(oy + height)
                key_boundaries[key_idx] = (wx1, wy1, wx2, wy2)
                white_key_idx += 1
        
        return key_boundaries
    
    def _estimate_white_key_width(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate white key width from bounding box."""
        x1, _, x2, _ = bbox
        return (x2 - x1) / self.NUM_WHITE_KEYS
    
    def _estimate_black_key_width(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate black key width (typically 60% of white key width)."""
        return self._estimate_white_key_width(bbox) * 0.6
    
    def visualize(
        self, 
        frame: np.ndarray, 
        region: KeyboardRegion,
        show_key_indices: bool = False
    ) -> np.ndarray:
        """
        Visualize detected keyboard region on frame.
        
        Args:
            frame: Input frame
            region: Detected keyboard region
            show_key_indices: Whether to show key index numbers
            
        Returns:
            Frame with visualization overlay
        """
        vis = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw key boundaries
        for key_idx, (kx1, ky1, kx2, ky2) in region.key_boundaries.items():
            midi_pitch = key_idx + 21
            note_in_octave = midi_pitch % 12
            is_black = note_in_octave in [1, 3, 6, 8, 10]
            
            color = (100, 100, 100) if is_black else (200, 200, 200)
            cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), color, 1)
            
            if show_key_indices and key_idx % 12 == 0:  # Show every C
                cv2.putText(
                    vis, str(key_idx), (kx1, ky2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
                )
        
        return vis

