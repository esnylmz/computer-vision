"""
Homography Computation for Keyboard Normalization

Computes and applies perspective transformations to normalize
keyboard view from different camera angles.

Usage:
    from src.keyboard.homography import HomographyComputer
    
    computer = HomographyComputer()
    H = computer.compute_from_corners(corners)
    normalized = computer.warp_frame(frame, H)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


class HomographyComputer:
    """
    Computes homography matrices for keyboard normalization.
    
    Transforms perspective view of piano keyboard to normalized
    top-down view for consistent key localization.
    """
    
    # Standard keyboard dimensions (in relative units)
    STANDARD_WIDTH = 1220  # mm for 88-key piano
    STANDARD_HEIGHT = 150  # mm typical key depth
    
    def __init__(
        self, 
        target_width: int = 1280,
        target_height: int = 160
    ):
        """
        Args:
            target_width: Width of normalized output
            target_height: Height of normalized output
        """
        self.target_width = target_width
        self.target_height = target_height
    
    def compute_from_corners(
        self, 
        corners: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute homography from corner points.
        
        Args:
            corners: Dict with 'LT', 'RT', 'RB', 'LB' corner points
                    (Left-Top, Right-Top, Right-Bottom, Left-Bottom)
        
        Returns:
            3x3 homography matrix
        """
        src_pts = np.float32([
            corners['LT'],
            corners['RT'],
            corners['RB'],
            corners['LB']
        ])
        
        dst_pts = np.float32([
            [0, 0],
            [self.target_width, 0],
            [self.target_width, self.target_height],
            [0, self.target_height]
        ])
        
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return H
    
    def compute_from_corner_strings(
        self, 
        corners: Dict[str, str]
    ) -> np.ndarray:
        """
        Compute homography from corner strings (PianoVAM format).
        
        Args:
            corners: Dict with "x, y" format strings
            
        Returns:
            3x3 homography matrix
        """
        parsed = {}
        for key, value in corners.items():
            if isinstance(value, str):
                x, y = value.split(', ')
                parsed[key] = (int(float(x)), int(float(y)))
            else:
                parsed[key] = value
        
        return self.compute_from_corners(parsed)
    
    def compute_from_lines(
        self, 
        top_line: Tuple[Tuple[int, int], Tuple[int, int]],
        bottom_line: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute homography from detected top and bottom lines.
        
        Args:
            top_line: ((x1, y1), (x2, y2)) for keyboard top edge
            bottom_line: ((x1, y1), (x2, y2)) for keyboard bottom edge
            
        Returns:
            3x3 homography matrix
        """
        (tx1, ty1), (tx2, ty2) = top_line
        (bx1, by1), (bx2, by2) = bottom_line
        
        corners = {
            'LT': (tx1, ty1),
            'RT': (tx2, ty2),
            'RB': (bx2, by2),
            'LB': (bx1, by1)
        }
        
        return self.compute_from_corners(corners)
    
    def warp_frame(
        self, 
        frame: np.ndarray, 
        H: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply homography to warp frame.
        
        Args:
            frame: Input frame
            H: 3x3 homography matrix
            output_size: (width, height) or None to use target size
            
        Returns:
            Warped frame
        """
        if output_size is None:
            output_size = (self.target_width, self.target_height)
        
        return cv2.warpPerspective(frame, H, output_size)
    
    def warp_point(
        self, 
        point: Tuple[float, float], 
        H: np.ndarray
    ) -> Tuple[float, float]:
        """
        Transform a single point using homography.
        
        Args:
            point: (x, y) coordinates
            H: 3x3 homography matrix
            
        Returns:
            Transformed (x, y) coordinates
        """
        x, y = point
        src = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))
    
    def warp_points(
        self, 
        points: List[Tuple[float, float]], 
        H: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Transform multiple points using homography.
        
        Args:
            points: List of (x, y) coordinates
            H: 3x3 homography matrix
            
        Returns:
            List of transformed (x, y) coordinates
        """
        if not points:
            return []
        
        src = np.array([[p] for p in points], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        
        return [(float(p[0, 0]), float(p[0, 1])) for p in dst]
    
    def inverse(self, H: np.ndarray) -> np.ndarray:
        """Compute inverse homography matrix."""
        return np.linalg.inv(H)
    
    def refine_with_features(
        self, 
        frame: np.ndarray,
        initial_corners: Dict[str, Tuple[int, int]],
        search_radius: int = 20
    ) -> Dict[str, Tuple[int, int]]:
        """
        Refine corner positions using feature detection.
        
        Args:
            frame: Input frame
            initial_corners: Initial corner estimates
            search_radius: Radius to search around initial corners
            
        Returns:
            Refined corner positions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        refined = {}
        for key, (x, y) in initial_corners.items():
            # Extract search region
            x1 = max(0, x - search_radius)
            y1 = max(0, y - search_radius)
            x2 = min(frame.shape[1], x + search_radius)
            y2 = min(frame.shape[0], y + search_radius)
            
            roi = gray[y1:y2, x1:x2]
            
            # Find corners in ROI
            corners = cv2.goodFeaturesToTrack(
                roi, maxCorners=1, qualityLevel=0.01, minDistance=5
            )
            
            if corners is not None and len(corners) > 0:
                cx, cy = corners[0][0]
                refined[key] = (int(x1 + cx), int(y1 + cy))
            else:
                refined[key] = (x, y)
        
        return refined
    
    def estimate_from_black_keys(
        self, 
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate homography by detecting black key pattern.
        
        Uses the characteristic 2-3 black key grouping pattern
        to identify keyboard orientation.
        
        Args:
            frame: Input frame
            
        Returns:
            Estimated homography matrix or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black keys
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (potential black keys)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by aspect ratio (black keys are tall and narrow)
        black_key_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            if 2.0 < aspect_ratio < 6.0 and h > 30:
                black_key_candidates.append((x, y, w, h))
        
        if len(black_key_candidates) < 5:
            return None
        
        # Sort by x-coordinate
        black_key_candidates.sort(key=lambda k: k[0])
        
        # Find keyboard extent
        x_min = min(k[0] for k in black_key_candidates)
        x_max = max(k[0] + k[2] for k in black_key_candidates)
        y_min = min(k[1] for k in black_key_candidates)
        y_max = max(k[1] + k[3] for k in black_key_candidates)
        
        # Estimate keyboard height (black keys are ~60% of total)
        keyboard_height = int((y_max - y_min) / 0.6)
        
        corners = {
            'LT': (x_min, y_min),
            'RT': (x_max, y_min),
            'RB': (x_max, y_min + keyboard_height),
            'LB': (x_min, y_min + keyboard_height)
        }
        
        return self.compute_from_corners(corners)

