"""
MIDI-Video Synchronization

Synchronizes MIDI events with video frames for accurate
finger-to-key assignment.

Usage:
    from src.assignment.midi_sync import MidiVideoSync
    
    sync = MidiVideoSync(fps=60.0)
    frame_events = sync.get_events_at_frame(midi_events, frame_idx)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SyncedEvent:
    """MIDI event synchronized with video frame."""
    frame_idx: int
    onset_time: float
    offset_time: float
    midi_pitch: int
    velocity: int
    key_idx: int  # 0-87 piano key index
    
    @property
    def note_name(self) -> str:
        """Get note name from MIDI pitch."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.midi_pitch // 12) - 1
        note = note_names[self.midi_pitch % 12]
        return f"{note}{octave}"


class MidiVideoSync:
    """
    Handles synchronization between MIDI events and video frames.
    
    Takes into account:
    - Frame rate differences
    - Audio/video offset
    - Event onset tolerance window
    """
    
    MIDI_PITCH_OFFSET = 21  # MIDI pitch to piano key index
    
    def __init__(
        self, 
        fps: float = 60.0,
        onset_tolerance_frames: int = 2,
        audio_video_offset: float = 0.0
    ):
        """
        Args:
            fps: Video frame rate
            onset_tolerance_frames: Frames to consider around onset
            audio_video_offset: Audio-video offset in seconds
        """
        self.fps = fps
        self.onset_tolerance_frames = onset_tolerance_frames
        self.audio_video_offset = audio_video_offset
    
    def sync_events(
        self, 
        midi_events: List[Dict],
        total_frames: Optional[int] = None
    ) -> List[SyncedEvent]:
        """
        Synchronize all MIDI events with video frames.
        
        Args:
            midi_events: List of dicts with 'onset', 'offset', 'pitch', 'velocity'
            total_frames: Total number of video frames
            
        Returns:
            List of SyncedEvent objects
        """
        synced = []
        
        for event in midi_events:
            onset = event.get('onset', 0) + self.audio_video_offset
            offset = event.get('offset', onset + 0.1) + self.audio_video_offset
            pitch = event.get('pitch', 60)
            velocity = event.get('velocity', 64)
            
            frame_idx = self.time_to_frame(onset)
            
            if total_frames is not None and frame_idx >= total_frames:
                continue
            
            synced.append(SyncedEvent(
                frame_idx=frame_idx,
                onset_time=onset,
                offset_time=offset,
                midi_pitch=pitch,
                velocity=velocity,
                key_idx=pitch - self.MIDI_PITCH_OFFSET
            ))
        
        return synced
    
    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.fps)
    
    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx / self.fps
    
    def get_events_at_frame(
        self, 
        synced_events: List[SyncedEvent],
        frame_idx: int,
        onset_only: bool = True
    ) -> List[SyncedEvent]:
        """
        Get events at a specific frame.
        
        Args:
            synced_events: List of SyncedEvent objects
            frame_idx: Frame index to query
            onset_only: If True, only return events with onset at this frame
            
        Returns:
            List of events at or near the frame
        """
        result = []
        frame_time = self.frame_to_time(frame_idx)
        tolerance_time = self.onset_tolerance_frames / self.fps
        
        for event in synced_events:
            if onset_only:
                # Check if onset is within tolerance of frame
                if abs(event.frame_idx - frame_idx) <= self.onset_tolerance_frames:
                    result.append(event)
            else:
                # Check if note is sounding at this frame
                if event.onset_time <= frame_time + tolerance_time and \
                   event.offset_time >= frame_time - tolerance_time:
                    result.append(event)
        
        return result
    
    def get_onset_frames(
        self, 
        synced_events: List[SyncedEvent]
    ) -> List[int]:
        """Get sorted list of unique onset frames."""
        frames = sorted(set(e.frame_idx for e in synced_events))
        return frames
    
    def group_by_frame(
        self, 
        synced_events: List[SyncedEvent]
    ) -> Dict[int, List[SyncedEvent]]:
        """
        Group events by their onset frame.
        
        Returns:
            Dict mapping frame index to list of events
        """
        grouped = {}
        
        for event in synced_events:
            frame = event.frame_idx
            if frame not in grouped:
                grouped[frame] = []
            grouped[frame].append(event)
        
        return grouped
    
    def find_simultaneous_notes(
        self, 
        synced_events: List[SyncedEvent],
        window_frames: int = 2
    ) -> List[List[SyncedEvent]]:
        """
        Find groups of notes played simultaneously (chords).
        
        Args:
            synced_events: List of events
            window_frames: Frame window for simultaneous detection
            
        Returns:
            List of event groups (each group is a chord)
        """
        if not synced_events:
            return []
        
        # Sort by frame
        sorted_events = sorted(synced_events, key=lambda e: e.frame_idx)
        
        groups = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            if event.frame_idx - current_group[-1].frame_idx <= window_frames:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]
        
        groups.append(current_group)
        
        return groups
    
    def create_frame_event_matrix(
        self, 
        synced_events: List[SyncedEvent],
        total_frames: int,
        num_keys: int = 88
    ) -> np.ndarray:
        """
        Create a binary matrix of key states per frame.
        
        Args:
            synced_events: List of events
            total_frames: Total number of frames
            num_keys: Number of piano keys
            
        Returns:
            Binary array of shape (total_frames, num_keys)
        """
        matrix = np.zeros((total_frames, num_keys), dtype=np.float32)
        
        for event in synced_events:
            if 0 <= event.key_idx < num_keys:
                onset_frame = event.frame_idx
                offset_frame = self.time_to_frame(event.offset_time)
                
                onset_frame = max(0, onset_frame)
                offset_frame = min(total_frames - 1, offset_frame)
                
                matrix[onset_frame:offset_frame + 1, event.key_idx] = 1.0
        
        return matrix
    
    def interpolate_missing_frames(
        self, 
        synced_events: List[SyncedEvent],
        hand_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate hand landmarks for frames with events but missing landmarks.
        
        Args:
            synced_events: List of events
            hand_landmarks: Shape (T, 21, 3) with NaN for missing
            
        Returns:
            Interpolated landmarks array
        """
        # Get frames that need landmarks
        needed_frames = set(e.frame_idx for e in synced_events)
        
        # Find missing frames
        missing = []
        for frame in needed_frames:
            if frame < len(hand_landmarks) and np.any(np.isnan(hand_landmarks[frame])):
                missing.append(frame)
        
        if not missing:
            return hand_landmarks
        
        result = hand_landmarks.copy()
        
        # Find valid frames
        valid_frames = [i for i in range(len(hand_landmarks)) 
                       if not np.any(np.isnan(hand_landmarks[i]))]
        
        if len(valid_frames) < 2:
            return result
        
        # Interpolate each missing frame
        for frame in missing:
            # Find nearest valid frames
            left = [f for f in valid_frames if f < frame]
            right = [f for f in valid_frames if f > frame]
            
            if left and right:
                l_frame = left[-1]
                r_frame = right[0]
                alpha = (frame - l_frame) / (r_frame - l_frame)
                result[frame] = (1 - alpha) * hand_landmarks[l_frame] + \
                               alpha * hand_landmarks[r_frame]
            elif left:
                result[frame] = hand_landmarks[left[-1]]
            elif right:
                result[frame] = hand_landmarks[right[0]]
        
        return result

