"""
Biomechanical Constraints for Piano Fingering

Implements constraints based on hand physiology to validate
and improve finger assignments.

Based on piano pedagogy literature and biomechanical studies.

Usage:
    from src.refinement.constraints import BiomechanicalConstraints
    
    constraints = BiomechanicalConstraints()
    is_valid = constraints.is_valid_transition(finger1, finger2, interval)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    note_idx: int
    constraint_type: str
    description: str
    severity: float  # 0-1, higher is worse


class BiomechanicalConstraints:
    """
    Validates finger assignments based on biomechanical constraints.
    
    Constraints include:
    - Maximum finger stretch (interval between fingers)
    - Finger crossing rules
    - Same-finger repetition
    - Hand position changes
    """
    
    # Maximum comfortable stretch in semitones between fingers
    MAX_STRETCH = {
        (1, 2): 10,  # Thumb to index
        (1, 3): 11,  # Thumb to middle
        (1, 4): 12,  # Thumb to ring
        (1, 5): 12,  # Thumb to pinky (max span)
        (2, 3): 5,   # Index to middle
        (2, 4): 7,   # Index to ring
        (2, 5): 9,   # Index to pinky
        (3, 4): 4,   # Middle to ring
        (3, 5): 6,   # Middle to pinky
        (4, 5): 5,   # Ring to pinky
    }
    
    # Finger ordering preference for ascending passages (right hand)
    # Lower finger number should be on lower pitch
    ASCENDING_ORDER = True
    
    def __init__(
        self,
        skill_level: str = 'intermediate',
        strict: bool = False
    ):
        """
        Args:
            skill_level: 'beginner', 'intermediate', or 'advanced'
            strict: If True, apply stricter constraints
        """
        self.skill_level = skill_level
        self.strict = strict
        
        # Adjust stretch limits based on skill level
        self.stretch_multiplier = {
            'beginner': 0.8,
            'intermediate': 1.0,
            'advanced': 1.2
        }.get(skill_level, 1.0)
    
    def get_max_stretch(self, finger1: int, finger2: int) -> int:
        """Get maximum stretch between two fingers."""
        key = tuple(sorted([finger1, finger2]))
        base_stretch = self.MAX_STRETCH.get(key, 12)
        return int(base_stretch * self.stretch_multiplier)
    
    def is_valid_stretch(
        self, 
        finger1: int, 
        finger2: int, 
        interval: int
    ) -> bool:
        """
        Check if stretch between fingers is valid.
        
        Args:
            finger1: First finger (1-5)
            finger2: Second finger (1-5)
            interval: Interval in semitones (absolute value)
            
        Returns:
            True if stretch is physically possible
        """
        max_stretch = self.get_max_stretch(finger1, finger2)
        return abs(interval) <= max_stretch
    
    def is_valid_transition(
        self,
        finger1: int,
        finger2: int,
        pitch1: int,
        pitch2: int,
        hand: str = 'right'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if finger transition is valid.
        
        Args:
            finger1: Current finger (1-5)
            finger2: Next finger (1-5)
            pitch1: Current MIDI pitch
            pitch2: Next MIDI pitch
            hand: 'left' or 'right'
            
        Returns:
            (is_valid, reason) tuple
        """
        interval = pitch2 - pitch1
        
        # Check stretch
        if not self.is_valid_stretch(finger1, finger2, interval):
            return False, f"Stretch too large: {abs(interval)} semitones"
        
        # Check same finger repetition (usually avoided)
        if finger1 == finger2 and interval != 0:
            if self.strict:
                return False, "Same finger on different notes"
            # Allow for repeated notes or very small intervals
            if abs(interval) > 2:
                return False, "Same finger with large interval"
        
        # Check finger ordering (for non-thumb notes)
        if self.strict and finger1 != 1 and finger2 != 1:
            if hand == 'right':
                # For ascending, higher finger should be on higher pitch
                if interval > 0 and finger2 < finger1:
                    return False, "Finger order violation (ascending)"
                if interval < 0 and finger2 > finger1:
                    return False, "Finger order violation (descending)"
            else:  # left hand (reversed)
                if interval > 0 and finger2 > finger1:
                    return False, "Finger order violation (ascending)"
                if interval < 0 and finger2 < finger1:
                    return False, "Finger order violation (descending)"
        
        return True, None
    
    def is_valid_thumb_crossing(
        self,
        finger_before: int,
        finger_after: int,
        pitch_before: int,
        pitch_after: int,
        hand: str = 'right'
    ) -> bool:
        """
        Check if thumb-under or finger-over thumb is valid.
        
        Thumb crossing is a special technique where thumb passes
        under other fingers or fingers cross over thumb.
        
        Returns:
            True if crossing is valid
        """
        # Thumb under (thumb goes under fingers 2-4)
        if finger_before in [2, 3, 4] and finger_after == 1:
            # For right hand ascending, thumb goes right
            if hand == 'right' and pitch_after > pitch_before:
                return True
            # For left hand descending, thumb goes right
            if hand == 'left' and pitch_after < pitch_before:
                return True
        
        # Finger over thumb (fingers 2-4 cross over thumb)
        if finger_before == 1 and finger_after in [2, 3, 4]:
            if hand == 'right' and pitch_after > pitch_before:
                return True
            if hand == 'left' and pitch_after < pitch_before:
                return True
        
        return False
    
    def validate_sequence(
        self,
        fingers: List[int],
        pitches: List[int],
        hands: List[str]
    ) -> List[ConstraintViolation]:
        """
        Validate a sequence of finger assignments.
        
        Args:
            fingers: List of finger assignments (1-5)
            pitches: List of MIDI pitches
            hands: List of hand labels
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        for i in range(len(fingers) - 1):
            is_valid, reason = self.is_valid_transition(
                fingers[i], fingers[i+1],
                pitches[i], pitches[i+1],
                hands[i]
            )
            
            if not is_valid:
                violations.append(ConstraintViolation(
                    note_idx=i,
                    constraint_type='transition',
                    description=reason,
                    severity=self._compute_severity(
                        fingers[i], fingers[i+1],
                        pitches[i], pitches[i+1]
                    )
                ))
        
        return violations
    
    def _compute_severity(
        self,
        finger1: int,
        finger2: int,
        pitch1: int,
        pitch2: int
    ) -> float:
        """Compute severity of a constraint violation."""
        interval = abs(pitch2 - pitch1)
        max_stretch = self.get_max_stretch(finger1, finger2)
        
        if interval <= max_stretch:
            return 0.0
        
        # Severity increases with excess stretch
        excess = interval - max_stretch
        return min(1.0, excess / 12)  # Cap at 1.0
    
    def compute_constraint_loss(
        self,
        fingers: List[int],
        pitches: List[int],
        hands: List[str]
    ) -> float:
        """
        Compute constraint violation loss for training.
        
        Returns:
            Loss value (0 = no violations)
        """
        violations = self.validate_sequence(fingers, pitches, hands)
        
        if not violations:
            return 0.0
        
        total_severity = sum(v.severity for v in violations)
        return total_severity / len(fingers)
    
    def get_valid_fingers(
        self,
        prev_finger: int,
        prev_pitch: int,
        current_pitch: int,
        hand: str = 'right'
    ) -> List[int]:
        """
        Get list of valid fingers for current note.
        
        Args:
            prev_finger: Previous finger used
            prev_pitch: Previous MIDI pitch
            current_pitch: Current MIDI pitch
            hand: 'left' or 'right'
            
        Returns:
            List of valid finger options (1-5)
        """
        valid = []
        interval = current_pitch - prev_pitch
        
        for finger in range(1, 6):
            is_valid, _ = self.is_valid_transition(
                prev_finger, finger,
                prev_pitch, current_pitch,
                hand
            )
            if is_valid:
                valid.append(finger)
        
        return valid if valid else list(range(1, 6))  # Return all if none valid
    
    def suggest_correction(
        self,
        fingers: List[int],
        pitches: List[int],
        hands: List[str],
        violation_idx: int
    ) -> Optional[int]:
        """
        Suggest a corrected finger for a violation.
        
        Args:
            fingers: Current finger sequence
            pitches: MIDI pitches
            hands: Hand labels
            violation_idx: Index of the violating note
            
        Returns:
            Suggested finger (1-5) or None
        """
        if violation_idx == 0:
            return None
        
        valid = self.get_valid_fingers(
            fingers[violation_idx - 1],
            pitches[violation_idx - 1],
            pitches[violation_idx],
            hands[violation_idx]
        )
        
        if valid:
            # Prefer finger closest to current assignment
            current = fingers[violation_idx]
            valid.sort(key=lambda f: abs(f - current))
            return valid[0]
        
        return None

