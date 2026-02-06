"""Finger-to-key assignment module."""

from .midi_sync import MidiVideoSync
from .gaussian_assignment import GaussianFingerAssigner, FingerAssignment
from .hand_separation import HandSeparator

__all__ = [
    "MidiVideoSync",
    "GaussianFingerAssigner",
    "FingerAssignment",
    "HandSeparator",
]

