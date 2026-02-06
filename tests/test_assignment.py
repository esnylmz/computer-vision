"""Tests for finger assignment module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.assignment.gaussian_assignment import GaussianFingerAssigner, FingerAssignment
from src.assignment.midi_sync import MidiVideoSync, SyncedEvent
from src.assignment.hand_separation import HandSeparator


class TestGaussianFingerAssigner:
    """Tests for GaussianFingerAssigner class."""
    
    @pytest.fixture
    def key_boundaries(self):
        """Create simple key boundaries for testing."""
        boundaries = {}
        key_width = 30
        
        for key_idx in range(88):
            x = key_idx * key_width
            boundaries[key_idx] = (x, 0, x + key_width, 100)
        
        return boundaries
    
    def test_init(self, key_boundaries):
        """Test initialization."""
        assigner = GaussianFingerAssigner(key_boundaries, sigma=15.0)
        
        assert assigner.sigma == 15.0
        assert len(assigner.key_centers) == 88
    
    def test_compute_probability(self, key_boundaries):
        """Test probability computation."""
        assigner = GaussianFingerAssigner(key_boundaries, sigma=15.0)
        
        # Fingertip exactly at key center
        key_center = assigner.key_centers[0]
        prob = assigner.compute_probability(key_center, key_center)
        
        assert prob == 1.0  # Maximum probability at center
    
    def test_probability_decreases_with_distance(self, key_boundaries):
        """Test that probability decreases with distance."""
        assigner = GaussianFingerAssigner(key_boundaries, sigma=15.0)
        
        key_center = assigner.key_centers[0]
        
        prob_at_center = assigner.compute_probability(key_center, key_center)
        prob_offset_10 = assigner.compute_probability(
            (key_center[0] + 10, key_center[1]), key_center
        )
        prob_offset_30 = assigner.compute_probability(
            (key_center[0] + 30, key_center[1]), key_center
        )
        
        assert prob_at_center > prob_offset_10 > prob_offset_30
    
    def test_assign_finger(self, key_boundaries):
        """Test finger assignment."""
        assigner = GaussianFingerAssigner(key_boundaries, sigma=15.0)
        
        # Place fingertips near key 10
        key_center = assigner.key_centers[10]
        fingertips = {
            1: (key_center[0] + 100, key_center[1]),  # Thumb far
            2: (key_center[0], key_center[1]),         # Index at center
            3: (key_center[0] + 50, key_center[1]),   # Middle nearby
            4: (key_center[0] + 80, key_center[1]),   # Ring far
            5: (key_center[0] + 100, key_center[1])   # Pinky far
        }
        
        assignment = assigner.assign_finger(
            fingertips, pressed_key_idx=10, hand='right'
        )
        
        assert assignment is not None
        assert assignment.assigned_finger == 2  # Index should be assigned
        assert assignment.confidence > 0.5
    
    def test_assign_returns_none_for_invalid_key(self, key_boundaries):
        """Test that invalid key returns None."""
        assigner = GaussianFingerAssigner(key_boundaries)
        
        fingertips = {1: (0, 0)}
        assignment = assigner.assign_finger(fingertips, pressed_key_idx=100, hand='right')
        
        assert assignment is None


class TestMidiVideoSync:
    """Tests for MidiVideoSync class."""
    
    def test_time_to_frame(self):
        """Test time to frame conversion."""
        sync = MidiVideoSync(fps=60.0)
        
        assert sync.time_to_frame(0.0) == 0
        assert sync.time_to_frame(1.0) == 60
        assert sync.time_to_frame(0.5) == 30
    
    def test_frame_to_time(self):
        """Test frame to time conversion."""
        sync = MidiVideoSync(fps=60.0)
        
        assert sync.frame_to_time(0) == 0.0
        assert sync.frame_to_time(60) == 1.0
        assert sync.frame_to_time(30) == 0.5
    
    def test_sync_events(self):
        """Test event synchronization."""
        sync = MidiVideoSync(fps=60.0)
        
        midi_events = [
            {'onset': 0.0, 'offset': 0.5, 'pitch': 60, 'velocity': 80},
            {'onset': 0.5, 'offset': 1.0, 'pitch': 62, 'velocity': 90},
        ]
        
        synced = sync.sync_events(midi_events)
        
        assert len(synced) == 2
        assert synced[0].frame_idx == 0
        assert synced[0].midi_pitch == 60
        assert synced[1].frame_idx == 30
        assert synced[1].midi_pitch == 62
    
    def test_get_events_at_frame(self):
        """Test getting events at specific frame."""
        sync = MidiVideoSync(fps=60.0, onset_tolerance_frames=2)
        
        synced = [
            SyncedEvent(0, 0.0, 0.5, 60, 80, 39),
            SyncedEvent(30, 0.5, 1.0, 62, 90, 41),
            SyncedEvent(60, 1.0, 1.5, 64, 85, 43),
        ]
        
        events = sync.get_events_at_frame(synced, 30)
        
        assert len(events) == 1
        assert events[0].midi_pitch == 62
    
    def test_group_by_frame(self):
        """Test grouping events by frame."""
        sync = MidiVideoSync(fps=60.0)
        
        synced = [
            SyncedEvent(0, 0.0, 0.5, 60, 80, 39),
            SyncedEvent(0, 0.0, 0.5, 64, 80, 43),  # Same frame, chord
            SyncedEvent(30, 0.5, 1.0, 62, 90, 41),
        ]
        
        grouped = sync.group_by_frame(synced)
        
        assert len(grouped) == 2
        assert len(grouped[0]) == 2  # Chord at frame 0
        assert len(grouped[30]) == 1


class TestHandSeparator:
    """Tests for HandSeparator class."""
    
    def test_determine_hand_by_position(self):
        """Test hand determination by position."""
        separator = HandSeparator(keyboard_center_x=640)
        
        hand, conf = separator.determine_hand_by_position(400)
        assert hand == 'left'
        
        hand, conf = separator.determine_hand_by_position(800)
        assert hand == 'right'
    
    def test_determine_hand_by_orientation(self):
        """Test hand determination by thumb-pinky orientation."""
        separator = HandSeparator()
        
        # Right hand: thumb left of pinky
        hand, conf = separator.determine_hand_by_orientation(100, 200)
        assert hand == 'right'
        
        # Left hand: thumb right of pinky
        hand, conf = separator.determine_hand_by_orientation(200, 100)
        assert hand == 'left'
    
    def test_determine_hand_combined(self):
        """Test combined hand determination."""
        separator = HandSeparator(keyboard_center_x=640)
        
        fingertips = {
            1: (100, 200),   # Thumb
            2: (150, 200),
            3: (200, 200),
            4: (250, 200),
            5: (300, 200)    # Pinky
        }
        
        info = separator.determine_hand(fingertips)
        
        # Both position (left of center) and orientation (thumb left of pinky)
        # would suggest different hands - should favor position
        assert info.hand_type in ['left', 'right']
        assert info.confidence > 0


class TestFingerAssignment:
    """Tests for FingerAssignment dataclass."""
    
    def test_finger_name(self):
        """Test finger name property."""
        assignment = FingerAssignment(
            note_onset=0.0,
            frame_idx=0,
            midi_pitch=60,
            key_idx=39,
            assigned_finger=2,
            hand='right',
            confidence=0.9,
            fingertip_position=(100, 200)
        )
        
        assert assignment.finger_name == 'index'
    
    def test_label(self):
        """Test label property."""
        assignment = FingerAssignment(
            note_onset=0.0,
            frame_idx=0,
            midi_pitch=60,
            key_idx=39,
            assigned_finger=3,
            hand='left',
            confidence=0.9,
            fingertip_position=(100, 200)
        )
        
        assert assignment.label == 'L3'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

