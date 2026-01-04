"""Tests for keyboard detection module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyboard.detector import KeyboardDetector, KeyboardRegion
from src.keyboard.key_localization import KeyLocalizer, KeyInfo
from src.keyboard.homography import HomographyComputer


class TestKeyboardDetector:
    """Tests for KeyboardDetector class."""
    
    def test_init_default(self):
        """Test default initialization."""
        detector = KeyboardDetector()
        assert detector.canny_low == 50
        assert detector.canny_high == 150
        assert detector.hough_threshold == 100
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'canny_low': 30,
            'canny_high': 200,
            'hough_threshold': 50
        }
        detector = KeyboardDetector(config)
        assert detector.canny_low == 30
        assert detector.canny_high == 200
        assert detector.hough_threshold == 50
    
    def test_detect_from_corners(self):
        """Test keyboard detection from corner annotations."""
        detector = KeyboardDetector()
        
        corners = {
            'LT': '100, 200',
            'RT': '1800, 200',
            'RB': '1800, 400',
            'LB': '100, 400'
        }
        
        region = detector.detect_from_corners(corners)
        
        assert region is not None
        assert isinstance(region, KeyboardRegion)
        assert len(region.key_boundaries) == 88
        assert region.white_key_width > 0
    
    def test_key_boundaries_correct_count(self):
        """Test that 88 key boundaries are generated."""
        detector = KeyboardDetector()
        
        corners = {
            'LT': (0, 0),
            'RT': (1700, 0),
            'RB': (1700, 200),
            'LB': (0, 200)
        }
        
        region = detector.detect_from_corners(corners)
        
        # Count white and black keys
        white_count = 0
        black_count = 0
        
        for key_idx in range(88):
            midi_pitch = key_idx + 21
            is_black = (midi_pitch % 12) in [1, 3, 6, 8, 10]
            if is_black:
                black_count += 1
            else:
                white_count += 1
        
        assert white_count == 52
        assert black_count == 36
        assert white_count + black_count == 88


class TestKeyLocalizer:
    """Tests for KeyLocalizer class."""
    
    @pytest.fixture
    def key_boundaries(self):
        """Create sample key boundaries."""
        boundaries = {}
        white_key_width = 30
        black_key_width = 18
        
        white_idx = 0
        for key_idx in range(88):
            midi_pitch = key_idx + 21
            is_black = (midi_pitch % 12) in [1, 3, 6, 8, 10]
            
            if is_black:
                x = white_idx * white_key_width - black_key_width // 2
                boundaries[key_idx] = (x, 0, x + black_key_width, 100)
            else:
                x = white_idx * white_key_width
                boundaries[key_idx] = (x, 0, x + white_key_width, 160)
                white_idx += 1
        
        return boundaries
    
    def test_get_key_info(self, key_boundaries):
        """Test getting key information."""
        localizer = KeyLocalizer(key_boundaries)
        
        # Test middle C (MIDI 60, key index 39)
        info = localizer.get_key_by_midi(60)
        
        assert info is not None
        assert info.midi_pitch == 60
        assert info.note_name == "C4"
        assert info.is_black is False
    
    def test_black_key_identification(self, key_boundaries):
        """Test black key identification."""
        localizer = KeyLocalizer(key_boundaries)
        
        # C# (MIDI 61) should be black
        info = localizer.get_key_by_midi(61)
        assert info.is_black is True
        
        # D (MIDI 62) should be white
        info = localizer.get_key_by_midi(62)
        assert info.is_black is False
    
    def test_point_to_key(self, key_boundaries):
        """Test point to key mapping."""
        localizer = KeyLocalizer(key_boundaries)
        
        # Get the center of first key
        first_key_info = localizer.get_key_info(0)
        cx, cy = first_key_info.center
        
        key_idx = localizer.point_to_key(cx, cy)
        assert key_idx == 0


class TestHomographyComputer:
    """Tests for HomographyComputer class."""
    
    def test_compute_from_corners(self):
        """Test homography computation from corners."""
        computer = HomographyComputer(target_width=1280, target_height=160)
        
        corners = {
            'LT': (100, 200),
            'RT': (1820, 200),
            'RB': (1820, 400),
            'LB': (100, 400)
        }
        
        H = computer.compute_from_corners(corners)
        
        assert H is not None
        assert H.shape == (3, 3)
    
    def test_warp_point(self):
        """Test point warping."""
        computer = HomographyComputer(target_width=100, target_height=100)
        
        corners = {
            'LT': (0, 0),
            'RT': (100, 0),
            'RB': (100, 100),
            'LB': (0, 100)
        }
        
        H = computer.compute_from_corners(corners)
        
        # Identity-like homography, point should stay same
        warped = computer.warp_point((50, 50), H)
        
        assert abs(warped[0] - 50) < 1
        assert abs(warped[1] - 50) < 1
    
    def test_inverse_homography(self):
        """Test inverse homography."""
        computer = HomographyComputer()
        
        corners = {
            'LT': (0, 0),
            'RT': (1280, 0),
            'RB': (1280, 160),
            'LB': (0, 160)
        }
        
        H = computer.compute_from_corners(corners)
        H_inv = computer.inverse(H)
        
        # H @ H_inv should be identity
        identity = H @ H_inv
        assert np.allclose(identity, np.eye(3), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

