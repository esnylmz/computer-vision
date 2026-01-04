"""
MIDI Processing Utilities

Provides functions for loading and processing MIDI files,
extracting note events, and synchronizing with video frames.

Usage:
    from src.data.midi_utils import MidiProcessor
    
    processor = MidiProcessor()
    events = processor.load_midi("path/to/file.mid")
"""

import mido
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class MidiEvent:
    """Single MIDI note event."""
    onset: float  # Onset time in seconds
    offset: float  # Offset time in seconds
    pitch: int  # MIDI note number (21-108 for piano)
    velocity: int  # Note velocity (0-127)
    channel: int  # MIDI channel
    
    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.offset - self.onset
    
    @property
    def note_name(self) -> str:
        """Get note name (e.g., 'C4', 'A#3')."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        note = note_names[self.pitch % 12]
        return f"{note}{octave}"
    
    def to_frame(self, fps: float = 60.0) -> int:
        """Convert onset time to frame index."""
        return int(self.onset * fps)


class MidiProcessor:
    """
    MIDI file processor for piano performance analysis.
    
    Handles loading MIDI files, extracting note events,
    and converting between time representations.
    """
    
    # Piano MIDI range
    PIANO_MIN_PITCH = 21  # A0
    PIANO_MAX_PITCH = 108  # C8
    
    def __init__(self, fps: float = 60.0):
        """
        Args:
            fps: Video frame rate for time-to-frame conversion
        """
        self.fps = fps
    
    def load_midi(self, midi_path: Union[str, Path]) -> List[MidiEvent]:
        """
        Load MIDI file and extract note events.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of MidiEvent objects sorted by onset time
        """
        midi_path = Path(midi_path)
        
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        mid = mido.MidiFile(midi_path)
        
        events = []
        active_notes = {}  # Track note_on events waiting for note_off
        current_time = 0.0
        
        for track in mid.tracks:
            current_time = 0.0
            
            for msg in track:
                # Convert delta time to absolute time
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, self._get_tempo(mid))
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note on event
                    key = (msg.note, msg.channel)
                    active_notes[key] = {
                        'onset': current_time,
                        'velocity': msg.velocity,
                        'channel': msg.channel
                    }
                    
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note off event
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        note_data = active_notes.pop(key)
                        events.append(MidiEvent(
                            onset=note_data['onset'],
                            offset=current_time,
                            pitch=msg.note,
                            velocity=note_data['velocity'],
                            channel=note_data['channel']
                        ))
        
        # Sort by onset time
        events.sort(key=lambda e: (e.onset, e.pitch))
        
        return events
    
    def _get_tempo(self, mid: mido.MidiFile) -> int:
        """Extract tempo from MIDI file (microseconds per beat)."""
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return msg.tempo
        return 500000  # Default: 120 BPM
    
    def events_to_pianoroll(
        self, 
        events: List[MidiEvent],
        duration: Optional[float] = None,
        time_resolution: float = 0.01
    ) -> np.ndarray:
        """
        Convert MIDI events to piano roll representation.
        
        Args:
            events: List of MidiEvent objects
            duration: Total duration in seconds (auto-detected if None)
            time_resolution: Time step in seconds
            
        Returns:
            Piano roll array of shape (num_timesteps, 88)
        """
        if not events:
            return np.zeros((1, 88))
        
        if duration is None:
            duration = max(e.offset for e in events)
        
        num_steps = int(duration / time_resolution) + 1
        piano_roll = np.zeros((num_steps, 88))
        
        for event in events:
            if self.PIANO_MIN_PITCH <= event.pitch <= self.PIANO_MAX_PITCH:
                key_idx = event.pitch - self.PIANO_MIN_PITCH
                start_step = int(event.onset / time_resolution)
                end_step = int(event.offset / time_resolution)
                piano_roll[start_step:end_step + 1, key_idx] = event.velocity / 127.0
        
        return piano_roll
    
    def get_events_at_frame(
        self, 
        events: List[MidiEvent], 
        frame: int,
        window: int = 1
    ) -> List[MidiEvent]:
        """
        Get MIDI events active at a specific frame.
        
        Args:
            events: List of MidiEvent objects
            frame: Frame index
            window: Frame window for onset detection (±window frames)
            
        Returns:
            List of events active at the frame
        """
        frame_time = frame / self.fps
        window_time = window / self.fps
        
        active_events = []
        for event in events:
            # Check if event onset is within window
            if abs(event.onset - frame_time) <= window_time:
                active_events.append(event)
            # Or if note is still sounding
            elif event.onset <= frame_time <= event.offset:
                active_events.append(event)
        
        return active_events
    
    def get_onset_frames(self, events: List[MidiEvent]) -> List[Tuple[int, MidiEvent]]:
        """
        Get list of (frame_index, event) for each note onset.
        
        Returns:
            List of (frame, event) tuples sorted by frame
        """
        onset_frames = [(event.to_frame(self.fps), event) for event in events]
        onset_frames.sort(key=lambda x: x[0])
        return onset_frames
    
    def filter_by_pitch_range(
        self, 
        events: List[MidiEvent],
        min_pitch: int = 21,
        max_pitch: int = 108
    ) -> List[MidiEvent]:
        """Filter events to only include notes in pitch range."""
        return [e for e in events if min_pitch <= e.pitch <= max_pitch]
    
    def split_by_hand(
        self, 
        events: List[MidiEvent],
        split_pitch: int = 60  # Middle C
    ) -> Tuple[List[MidiEvent], List[MidiEvent]]:
        """
        Split events into left and right hand based on pitch.
        
        Note: This is a simple heuristic. Real hand separation
        requires more sophisticated analysis.
        
        Args:
            events: List of MidiEvent objects
            split_pitch: MIDI pitch to split on (default: Middle C)
            
        Returns:
            (left_hand_events, right_hand_events)
        """
        left_hand = [e for e in events if e.pitch < split_pitch]
        right_hand = [e for e in events if e.pitch >= split_pitch]
        return left_hand, right_hand


def load_midi_events(midi_path: Union[str, Path], fps: float = 60.0) -> List[MidiEvent]:
    """
    Convenience function to load MIDI events.
    
    Args:
        midi_path: Path to MIDI file
        fps: Frame rate for time conversion
        
    Returns:
        List of MidiEvent objects
    """
    processor = MidiProcessor(fps=fps)
    return processor.load_midi(midi_path)

