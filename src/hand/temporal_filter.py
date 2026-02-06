"""
Temporal Filtering for Hand Landmarks

Applies noise reduction and smoothing to MediaPipe hand landmark sequences.

Based on PianoMotion10M (Gan et al. 2024) recommendations:
1. Hampel filter for outlier detection
2. Linear interpolation for missing frames
3. Savitzky-Golay filter for smoothing

Usage:
    from src.hand.temporal_filter import TemporalFilter
    
    filter = TemporalFilter()
    smoothed = filter.process(raw_landmarks)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Optional, Tuple


class TemporalFilter:
    """
    Temporal filtering pipeline for hand landmark sequences.
    
    Handles common issues in MediaPipe output:
    - Sudden jumps (detection failures)
    - Missing frames (no hand detected)
    - High-frequency noise
    """
    
    def __init__(
        self,
        hampel_window: int = 20,
        hampel_threshold: float = 3.0,
        max_interpolation_gap: int = 30,
        savgol_window: int = 11,
        savgol_order: int = 3
    ):
        """
        Args:
            hampel_window: Window size for Hampel filter
            hampel_threshold: Number of MADs for outlier detection
            max_interpolation_gap: Maximum gap to interpolate (frames)
            savgol_window: Window size for Savitzky-Golay filter
            savgol_order: Polynomial order for Savitzky-Golay
        """
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold
        self.max_interpolation_gap = max_interpolation_gap
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
    
    def process(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply full filtering pipeline to landmark sequence.
        
        Args:
            landmarks: Shape (T, 21, 3) - T frames, 21 landmarks, (x, y, z)
            
        Returns:
            Filtered landmarks with same shape
        """
        if landmarks.size == 0:
            return landmarks
        
        T, num_landmarks, coords = landmarks.shape
        result = landmarks.copy()
        
        # Process each landmark and coordinate independently
        for l in range(num_landmarks):
            for c in range(coords):
                signal = result[:, l, c].copy()
                
                # Check if signal has valid data
                if np.all(np.isnan(signal)):
                    continue
                
                # Step 1: Hampel filter for outlier detection
                signal, outlier_mask = self._hampel_filter(signal)
                
                # Step 2: Interpolate outliers and missing values
                signal = self._interpolate_gaps(signal, outlier_mask)
                
                # Step 3: Savitzky-Golay smoothing
                if not np.all(np.isnan(signal)):
                    signal = self._savgol_smooth(signal)
                
                result[:, l, c] = signal
        
        return result
    
    def process_with_confidence(
        self, 
        landmarks: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Apply filtering with confidence weighting.
        
        Args:
            landmarks: Shape (T, 21, 3)
            confidences: Shape (T,) confidence scores per frame
            
        Returns:
            Filtered landmarks
        """
        result = landmarks.copy()
        
        # Mark low-confidence frames as outliers
        low_conf_mask = confidences < 0.5
        result[low_conf_mask] = np.nan
        
        return self.process(result)
    
    def _hampel_filter(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hampel filter to identify and replace outliers.
        
        Uses median absolute deviation (MAD) for robust outlier detection.
        
        Returns:
            (filtered_signal, outlier_mask)
        """
        n = len(signal)
        result = signal.copy()
        outlier_mask = np.zeros(n, dtype=bool)
        
        # Handle NaN values
        nan_mask = np.isnan(signal)
        outlier_mask |= nan_mask
        
        k = self.hampel_window // 2
        
        for i in range(k, n - k):
            if nan_mask[i]:
                continue
            
            window = signal[i - k:i + k + 1]
            valid_window = window[~np.isnan(window)]
            
            if len(valid_window) < 3:
                continue
            
            median = np.median(valid_window)
            mad = np.median(np.abs(valid_window - median))
            
            # MAD to std conversion factor
            sigma = 1.4826 * mad
            
            if sigma > 1e-6 and np.abs(signal[i] - median) > self.hampel_threshold * sigma:
                result[i] = median
                outlier_mask[i] = True
        
        return result, outlier_mask
    
    def _interpolate_gaps(
        self, 
        signal: np.ndarray,
        outlier_mask: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate gaps in signal (NaN values or detected outliers).
        
        Only interpolates gaps smaller than max_interpolation_gap.
        """
        result = signal.copy()
        
        # Mark outliers as NaN for interpolation
        result[outlier_mask] = np.nan
        
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(result)
        
        if not np.any(valid_mask):
            return signal  # Return original if all invalid
        
        if np.all(valid_mask):
            return result  # No gaps to fill
        
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]
        
        if len(valid_indices) < 2:
            return signal
        
        # Create interpolation function
        f = interp1d(
            valid_indices, 
            result[valid_mask],
            kind='linear', 
            fill_value='extrapolate',
            bounds_error=False
        )
        
        for idx in invalid_indices:
            # Find nearest valid neighbors
            left_neighbors = valid_indices[valid_indices < idx]
            right_neighbors = valid_indices[valid_indices > idx]
            
            if len(left_neighbors) > 0 and len(right_neighbors) > 0:
                gap_size = right_neighbors[0] - left_neighbors[-1]
                if gap_size <= self.max_interpolation_gap:
                    result[idx] = f(idx)
                else:
                    result[idx] = signal[idx]  # Keep original for large gaps
            elif len(left_neighbors) > 0:
                # Extrapolate from left
                if idx - left_neighbors[-1] <= self.max_interpolation_gap // 2:
                    result[idx] = result[left_neighbors[-1]]
            elif len(right_neighbors) > 0:
                # Extrapolate from right
                if right_neighbors[0] - idx <= self.max_interpolation_gap // 2:
                    result[idx] = result[right_neighbors[0]]
        
        return result
    
    def _savgol_smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter for smoothing."""
        # Replace remaining NaNs with interpolated values for smoothing
        nan_mask = np.isnan(signal)
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            if np.sum(valid_mask) < self.savgol_window:
                return signal
            
            valid_indices = np.where(valid_mask)[0]
            f = interp1d(
                valid_indices, 
                signal[valid_mask],
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            signal = f(np.arange(len(signal)))
        
        if len(signal) < self.savgol_window:
            return signal
        
        # Ensure window length is odd
        window = self.savgol_window
        if window % 2 == 0:
            window += 1
        
        # Ensure window <= signal length
        window = min(window, len(signal))
        if window % 2 == 0:
            window -= 1
        
        if window < 3:
            return signal
        
        # Ensure order < window
        order = min(self.savgol_order, window - 1)
        
        return savgol_filter(signal, window, order)
    
    def compute_velocity(self, landmarks: np.ndarray, fps: float = 60.0) -> np.ndarray:
        """
        Compute velocity of landmarks.
        
        Args:
            landmarks: Shape (T, 21, 3)
            fps: Frame rate
            
        Returns:
            Velocity array of shape (T-1, 21, 3)
        """
        dt = 1.0 / fps
        velocity = np.diff(landmarks, axis=0) / dt
        return velocity
    
    def compute_acceleration(self, landmarks: np.ndarray, fps: float = 60.0) -> np.ndarray:
        """
        Compute acceleration of landmarks.
        
        Args:
            landmarks: Shape (T, 21, 3)
            fps: Frame rate
            
        Returns:
            Acceleration array of shape (T-2, 21, 3)
        """
        velocity = self.compute_velocity(landmarks, fps)
        dt = 1.0 / fps
        acceleration = np.diff(velocity, axis=0) / dt
        return acceleration
    
    def detect_sudden_movements(
        self, 
        landmarks: np.ndarray,
        velocity_threshold: float = 1000.0,
        fps: float = 60.0
    ) -> np.ndarray:
        """
        Detect frames with sudden movements (potential detection errors).
        
        Args:
            landmarks: Shape (T, 21, 3)
            velocity_threshold: Maximum expected velocity
            fps: Frame rate
            
        Returns:
            Boolean mask of shape (T,) indicating suspicious frames
        """
        velocity = self.compute_velocity(landmarks, fps)
        velocity_magnitude = np.linalg.norm(velocity, axis=2)  # (T-1, 21)
        max_velocity = np.nanmax(velocity_magnitude, axis=1)  # (T-1,)
        
        suspicious = np.zeros(len(landmarks), dtype=bool)
        suspicious[1:] = max_velocity > velocity_threshold
        
        return suspicious

