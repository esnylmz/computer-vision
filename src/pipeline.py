"""
Piano Fingering Detection Pipeline

Main entry point for running the full fingering detection pipeline.

Usage:
    python -m src.pipeline --config configs/default.yaml --input video.mp4
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from .data.dataset import PianoVAMDataset
from .data.midi_utils import MidiProcessor
from .data.video_utils import VideoProcessor
from .keyboard.detector import KeyboardDetector, KeyboardRegion
from .hand.skeleton_loader import SkeletonLoader
from .hand.temporal_filter import TemporalFilter
from .hand.fingertip_extractor import FingertipExtractor
from .assignment.gaussian_assignment import GaussianFingerAssigner, FingerAssignment
from .assignment.midi_sync import MidiVideoSync
from .assignment.hand_separation import HandSeparator
from .evaluation.metrics import FingeringMetrics
from .utils.config import load_config, Config
from .utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


class FingeringPipeline:
    """
    End-to-end pipeline for piano fingering detection.
    
    Stages:
    1. Keyboard Detection - Detect and localize piano keys
    2. Hand Processing - Filter and process hand landmarks
    3. Finger Assignment - Assign fingers to pressed keys
    4. (Optional) Neural Refinement - Refine assignments with BiLSTM
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.keyboard_detector = KeyboardDetector({
            'canny_low': config.keyboard.canny_low,
            'canny_high': config.keyboard.canny_high,
            'hough_threshold': config.keyboard.hough_threshold
        })
        
        self.skeleton_loader = SkeletonLoader()
        
        self.temporal_filter = TemporalFilter(
            hampel_window=config.hand.hampel_window,
            hampel_threshold=config.hand.hampel_threshold,
            max_interpolation_gap=config.hand.interpolation_max_gap,
            savgol_window=config.hand.savgol_window,
            savgol_order=config.hand.savgol_order
        )
        
        self.fingertip_extractor = FingertipExtractor()
        self.hand_separator = HandSeparator()
        self.midi_sync = MidiVideoSync(fps=config.video_fps)
        
        self.finger_assigner = None  # Initialized after keyboard detection
        self.metrics = FingeringMetrics()
        
        logger.info("Pipeline initialized")
    
    def process_sample(
        self,
        video_path: str,
        skeleton_data: Dict,
        midi_events: List[Dict],
        keyboard_corners: Optional[Dict] = None
    ) -> List[FingerAssignment]:
        """
        Process a single sample through the pipeline.
        
        Args:
            video_path: Path to video file
            skeleton_data: Hand skeleton data
            midi_events: MIDI note events
            keyboard_corners: Optional keyboard corner annotations
            
        Returns:
            List of finger assignments
        """
        # Stage 1: Keyboard Detection
        logger.info("Stage 1: Keyboard Detection")
        if keyboard_corners:
            keyboard_region = self.keyboard_detector.detect_from_corners(keyboard_corners)
        else:
            with VideoProcessor(video_path) as vp:
                frame = vp.get_frame(0)
            keyboard_region = self.keyboard_detector.detect(frame)
        
        if keyboard_region is None:
            logger.error("Keyboard detection failed")
            return []
        
        logger.info(f"Detected {len(keyboard_region.key_boundaries)} keys")
        
        # Initialize finger assigner with detected keyboard
        self.finger_assigner = GaussianFingerAssigner(
            keyboard_region.key_boundaries,
            sigma=self.config.assignment.sigma,
            candidate_range=self.config.assignment.candidate_keys
        )
        
        # Set keyboard center for hand separation
        x_min = min(kb[0] for kb in keyboard_region.key_boundaries.values())
        x_max = max(kb[2] for kb in keyboard_region.key_boundaries.values())
        self.hand_separator.set_keyboard_center((x_min + x_max) / 2)
        
        # Stage 2: Hand Processing
        logger.info("Stage 2: Hand Processing")
        hands = self.skeleton_loader._parse_json(skeleton_data)
        
        left_landmarks = self.skeleton_loader.to_array(hands['left'])
        right_landmarks = self.skeleton_loader.to_array(hands['right'])
        
        if left_landmarks.size > 0:
            left_landmarks = self.temporal_filter.process(left_landmarks)
            logger.info(f"Left hand: {len(left_landmarks)} frames")
        
        if right_landmarks.size > 0:
            right_landmarks = self.temporal_filter.process(right_landmarks)
            logger.info(f"Right hand: {len(right_landmarks)} frames")
        
        # Stage 3: Finger Assignment
        logger.info("Stage 3: Finger Assignment")
        synced_events = self.midi_sync.sync_events(midi_events)
        logger.info(f"Synced {len(synced_events)} MIDI events")
        
        assignments = []
        
        for event in synced_events:
            frame_idx = event.frame_idx
            key_idx = event.key_idx
            
            # Determine which hand to use
            key_center = self.finger_assigner.key_centers.get(key_idx)
            if key_center is None:
                continue
            
            # Try right hand first, then left
            assignment = None
            
            if frame_idx < len(right_landmarks):
                landmarks = right_landmarks[frame_idx]
                if not np.any(np.isnan(landmarks)):
                    assignment = self.finger_assigner.assign_from_landmarks(
                        landmarks, key_idx, 'right', frame_idx, event.onset_time
                    )
            
            if assignment is None and frame_idx < len(left_landmarks):
                landmarks = left_landmarks[frame_idx]
                if not np.any(np.isnan(landmarks)):
                    assignment = self.finger_assigner.assign_from_landmarks(
                        landmarks, key_idx, 'left', frame_idx, event.onset_time
                    )
            
            if assignment:
                assignments.append(assignment)
        
        logger.info(f"Assigned fingers to {len(assignments)} notes")
        
        return assignments
    
    def evaluate(
        self,
        predictions: List[FingerAssignment],
        ground_truth: List[int]
    ) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of FingerAssignment objects
            ground_truth: List of ground truth finger labels
            
        Returns:
            Evaluation metrics dictionary
        """
        pred_fingers = [p.assigned_finger for p in predictions]
        
        result = self.metrics.evaluate(pred_fingers, ground_truth)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {result.accuracy:.3f}")
        logger.info(f"  M_gen: {result.m_gen:.3f}")
        logger.info(f"  M_high: {result.m_high:.3f}")
        logger.info(f"  IFR: {result.ifr:.3f}")
        
        return {
            'accuracy': result.accuracy,
            'm_gen': result.m_gen,
            'm_high': result.m_high,
            'ifr': result.ifr,
            'num_notes': result.num_notes
        }


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Piano Fingering Detection Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input video file or sample ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    logger.info("Piano Fingering Detection Pipeline")
    logger.info(f"Config: {args.config}")
    
    config = load_config(args.config)
    pipeline = FingeringPipeline(config)
    
    # TODO: Load input and run pipeline
    logger.info("Pipeline ready. Use from Python API or notebooks.")


if __name__ == "__main__":
    main()

