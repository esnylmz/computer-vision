"""
Video Processing Utilities

Provides functions for loading video frames, extracting regions of interest,
and synchronizing video with MIDI events.

Usage:
    from src.data.video_utils import VideoProcessor
    
    processor = VideoProcessor("path/to/video.mp4")
    frames = processor.extract_frames(start=0, end=100)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Union
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


class VideoProcessor:
    """
    Video processor for piano performance analysis.
    
    Handles video loading, frame extraction, and region cropping.
    Optimized for processing long piano performance videos.
    """
    
    def __init__(self, video_path: Optional[Union[str, Path]] = None):
        """
        Args:
            video_path: Path to video file (optional, can be set later)
        """
        self.video_path = Path(video_path) if video_path else None
        self._cap = None
        self._info = None
    
    def open(self, video_path: Union[str, Path]) -> 'VideoProcessor':
        """
        Open a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Self for method chaining
        """
        self.video_path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self._info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self._cap.get(cv2.CAP_PROP_FPS)
        )
        
        return self
    
    def close(self):
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        if self.video_path and self._cap is None:
            self.open(self.video_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def info(self) -> VideoInfo:
        """Get video information."""
        if self._info is None:
            if self.video_path:
                self.open(self.video_path)
            else:
                raise ValueError("No video loaded")
        return self._info
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a single frame by index.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Frame as BGR numpy array, or None if frame not available
        """
        if self._cap is None:
            self.open(self.video_path)
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        
        return frame if ret else None
    
    def get_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """
        Get frame at specific time.
        
        Args:
            time_sec: Time in seconds
            
        Returns:
            Frame as BGR numpy array
        """
        frame_idx = int(time_sec * self.info.fps)
        return self.get_frame(frame_idx)
    
    def extract_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video.
        
        Args:
            start: Starting frame index
            end: Ending frame index (exclusive), None for all frames
            step: Frame step (1 = every frame, 2 = every other frame, etc.)
            
        Yields:
            (frame_index, frame) tuples
        """
        if self._cap is None:
            self.open(self.video_path)
        
        if end is None:
            end = self.info.frame_count
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for frame_idx in range(start, end, step):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            
            if not ret:
                break
                
            yield frame_idx, frame
    
    def extract_frames_at_indices(
        self, 
        indices: List[int]
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames at specific indices.
        
        Args:
            indices: List of frame indices to extract
            
        Yields:
            (frame_index, frame) tuples
        """
        if self._cap is None:
            self.open(self.video_path)
        
        for frame_idx in indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            
            if ret:
                yield frame_idx, frame
    
    def crop_region(
        self, 
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop a region from frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped region
        """
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()
    
    def apply_homography(
        self,
        frame: np.ndarray,
        homography: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply homography transformation to frame.
        
        Args:
            frame: Input frame
            homography: 3x3 homography matrix
            output_size: (width, height) of output
            
        Returns:
            Warped frame
        """
        return cv2.warpPerspective(frame, homography, output_size)
    
    def resize(
        self, 
        frame: np.ndarray, 
        size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize frame.
        
        Args:
            frame: Input frame
            size: (width, height)
            interpolation: OpenCV interpolation method
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, size, interpolation=interpolation)
    
    def to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def extract_frames_from_video(
    video_path: Union[str, Path],
    frame_indices: Optional[List[int]] = None,
    sample_rate: int = 1
) -> List[np.ndarray]:
    """
    Convenience function to extract frames from video.
    
    Args:
        video_path: Path to video file
        frame_indices: Specific frames to extract, or None for all
        sample_rate: Extract every Nth frame (if frame_indices is None)
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    
    with VideoProcessor(video_path) as processor:
        if frame_indices is not None:
            for _, frame in processor.extract_frames_at_indices(frame_indices):
                frames.append(frame)
        else:
            for _, frame in processor.extract_frames(step=sample_rate):
                frames.append(frame)
    
    return frames

