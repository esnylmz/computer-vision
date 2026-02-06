"""Tests for hand processing module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand.temporal_filter import TemporalFilter
from src.hand.fingertip_extractor import FingertipExtractor, FingertipData
from src.hand.skeleton_loader import SkeletonLoader, HandLandmarks


class TestTemporalFilter:
    """Tests for TemporalFilter class."""
    
    def test_init_default(self):
        """Test default initialization."""
        filter = TemporalFilter()
        assert filter.hampel_window == 20
        assert filter.hampel_threshold == 3.0
        assert filter.max_interpolation_gap == 30
        assert filter.savgol_window == 11
    
    def test_process_clean_signal(self):
        """Test processing a clean signal (no outliers)."""
        filter = TemporalFilter()
        
        # Create clean landmark sequence
        T = 100
        landmarks = np.zeros((T, 21, 3))
        for t in range(T):
            landmarks[t, :, 0] = t  # X increases linearly
            landmarks[t, :, 1] = 50  # Y constant
        
        result = filter.process(landmarks)
        
        assert result.shape == landmarks.shape
        # Should be close to original for clean data
        assert np.allclose(result[10:-10], landmarks[10:-10], atol=5)
    
    def test_process_with_outliers(self):
        """Test that outliers are detected and removed."""
        filter = TemporalFilter(hampel_window=10, hampel_threshold=2.0)
        
        # Create signal with outliers
        T = 50
        landmarks = np.zeros((T, 21, 3))
        landmarks[:, :, 0] = np.linspace(0, 100, T)[:, np.newaxis]
        
        # Add outlier
        landmarks[25, 0, 0] = 500  # Large spike
        
        result = filter.process(landmarks)
        
        # Outlier should be smoothed out
        assert result[25, 0, 0] < 200
    
    def test_process_with_missing(self):
        """Test interpolation of missing frames."""
        filter = TemporalFilter(max_interpolation_gap=10)
        
        T = 30
        landmarks = np.zeros((T, 21, 3))
        landmarks[:, :, 0] = np.linspace(0, 100, T)[:, np.newaxis]
        
        # Mark some frames as missing
        landmarks[10:15, :, :] = np.nan
        
        result = filter.process(landmarks)
        
        # Missing values should be interpolated
        assert not np.any(np.isnan(result[10:15]))
    
    def test_compute_velocity(self):
        """Test velocity computation."""
        filter = TemporalFilter()
        
        T = 10
        landmarks = np.zeros((T, 21, 3))
        landmarks[:, 0, 0] = np.arange(T) * 60  # 60 pixels per frame at 60fps = 1 pixel/sec
        
        velocity = filter.compute_velocity(landmarks, fps=60.0)
        
        assert velocity.shape == (T - 1, 21, 3)
        # Velocity should be constant
        assert np.allclose(velocity[:, 0, 0], 3600, atol=1)  # 60 * 60 = 3600 pixels/sec


class TestFingertipExtractor:
    """Tests for FingertipExtractor class."""
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks."""
        landmarks = np.zeros((21, 3))
        # Set fingertip positions
        landmarks[4] = [100, 200, 0]   # Thumb
        landmarks[8] = [150, 200, 0]   # Index
        landmarks[12] = [200, 200, 0]  # Middle
        landmarks[16] = [250, 200, 0]  # Ring
        landmarks[20] = [300, 200, 0]  # Pinky
        return landmarks
    
    def test_extract(self, sample_landmarks):
        """Test basic extraction."""
        extractor = FingertipExtractor()
        
        result = extractor.extract(sample_landmarks, frame_idx=0, hand_type='right')
        
        assert isinstance(result, FingertipData)
        assert len(result.positions) == 5
        assert result.hand_type == 'right'
        assert result.frame_idx == 0
    
    def test_finger_positions(self, sample_landmarks):
        """Test correct finger positions."""
        extractor = FingertipExtractor()
        
        result = extractor.extract(sample_landmarks)
        
        assert result.positions[1][0] == 100  # Thumb X
        assert result.positions[2][0] == 150  # Index X
        assert result.positions[3][0] == 200  # Middle X
        assert result.positions[4][0] == 250  # Ring X
        assert result.positions[5][0] == 300  # Pinky X
    
    def test_get_position_2d(self, sample_landmarks):
        """Test 2D position extraction."""
        extractor = FingertipExtractor()
        
        result = extractor.extract(sample_landmarks)
        pos = result.get_position_2d(1)
        
        assert pos == (100, 200)
    
    def test_compute_hand_span(self, sample_landmarks):
        """Test hand span computation."""
        extractor = FingertipExtractor()
        
        result = extractor.extract(sample_landmarks)
        span = extractor.compute_hand_span(result)
        
        # Span from thumb (100) to pinky (300)
        assert span == 200.0
    
    def test_extract_sequence(self):
        """Test sequence extraction."""
        extractor = FingertipExtractor()
        
        # Create sequence
        T = 5
        sequence = np.zeros((T, 21, 3))
        for t in range(T):
            sequence[t, 4] = [100 + t*10, 200, 0]
            sequence[t, 8] = [150 + t*10, 200, 0]
        
        results = extractor.extract_sequence(sequence, hand_type='left')
        
        assert len(results) == T
        assert all(r.hand_type == 'left' for r in results)


class TestSkeletonLoader:
    """Tests for SkeletonLoader class."""
    
    def test_init(self):
        """Test initialization."""
        loader = SkeletonLoader(normalize=False)
        assert loader.normalize is False
        
        loader = SkeletonLoader(normalize=True)
        assert loader.normalize is True
    
    def test_to_array(self):
        """Test conversion to array."""
        loader = SkeletonLoader()
        
        # Create sample HandLandmarks
        landmarks_list = [
            HandLandmarks(
                landmarks=np.random.rand(21, 3),
                hand_type='right',
                confidence=1.0,
                frame_idx=i
            )
            for i in range(5)
        ]
        
        array = loader.to_array(landmarks_list, total_frames=5)
        
        assert array.shape == (5, 21, 3)
    
    def test_to_array_with_gaps(self):
        """Test array conversion with missing frames."""
        loader = SkeletonLoader()
        
        landmarks_list = [
            HandLandmarks(
                landmarks=np.ones((21, 3)),
                hand_type='right',
                confidence=1.0,
                frame_idx=0
            ),
            HandLandmarks(
                landmarks=np.ones((21, 3)) * 2,
                hand_type='right',
                confidence=1.0,
                frame_idx=4
            )
        ]
        
        array = loader.to_array(landmarks_list, fill_missing=True, total_frames=5)
        
        assert array.shape == (5, 21, 3)
        # Frames 1-3 should be NaN
        assert np.all(np.isnan(array[1:4]))


class TestHandLandmarks:
    """Tests for HandLandmarks dataclass."""
    
    def test_fingertips_property(self):
        """Test fingertips property."""
        landmarks = np.zeros((21, 3))
        landmarks[4] = [1, 2, 3]
        landmarks[8] = [4, 5, 6]
        landmarks[12] = [7, 8, 9]
        landmarks[16] = [10, 11, 12]
        landmarks[20] = [13, 14, 15]
        
        hl = HandLandmarks(
            landmarks=landmarks,
            hand_type='right',
            confidence=1.0,
            frame_idx=0
        )
        
        fingertips = hl.fingertips
        
        assert fingertips.shape == (5, 3)
        assert np.array_equal(fingertips[0], [1, 2, 3])
        assert np.array_equal(fingertips[4], [13, 14, 15])
    
    def test_get_finger_tip(self):
        """Test get_finger_tip method."""
        landmarks = np.zeros((21, 3))
        landmarks[8] = [100, 200, 0]  # Index tip
        
        hl = HandLandmarks(
            landmarks=landmarks,
            hand_type='right',
            confidence=1.0,
            frame_idx=0
        )
        
        tip = hl.get_finger_tip(2)  # Index = finger 2
        
        assert np.array_equal(tip, [100, 200, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

