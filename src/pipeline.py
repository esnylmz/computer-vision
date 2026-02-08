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
from .keyboard.auto_detector import AutoKeyboardDetector, AutoDetectionResult
from .hand.skeleton_loader import SkeletonLoader
from .hand.live_detector import LiveHandDetector, LiveDetectionConfig
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
        kbd_cfg = {
            'canny_low': config.keyboard.canny_low,
            'canny_high': config.keyboard.canny_high,
            'hough_threshold': config.keyboard.hough_threshold,
        }
        self.keyboard_detector = KeyboardDetector(kbd_cfg)
        self.auto_keyboard_detector = AutoKeyboardDetector(kbd_cfg)
        
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
        midi_events: List[Dict],
        keyboard_corners: Optional[Dict] = None
    ) -> List[FingerAssignment]:
        """
        Process a single sample through the full-CV pipeline.
        
        Both keyboard detection and hand pose estimation run directly
        on raw video — no dataset annotations or pre-extracted skeletons
        are used.  Corner annotations, when provided, are used
        **only for IoU evaluation** of the auto-detection quality.
        
        Args:
            video_path: Path to video file
            midi_events: MIDI note events
            keyboard_corners: Optional corner annotations (evaluation only)
            
        Returns:
            List of finger assignments
        """
        # Stage 1: Automatic Keyboard Detection (full CV — no annotations)
        logger.info("Stage 1: Automatic Keyboard Detection")

        auto_result = self.auto_keyboard_detector.detect_from_video(video_path)
        if auto_result.success:
            keyboard_region = auto_result.keyboard_region
            logger.info(f"  Auto-detection succeeded (bbox={auto_result.consensus_bbox})")
        else:
            logger.error("Keyboard auto-detection failed")
            return []

        # Evaluate against corner annotations if available (evaluation only)
        if keyboard_corners:
            iou = self.auto_keyboard_detector.evaluate_against_corners(
                auto_result, keyboard_corners
            )
            logger.info(f"  IoU vs corner annotations (eval): {iou:.3f}")
        
        logger.info(f"Detected {len(keyboard_region.key_boundaries)} keys")
        
        # Auto-detected key boundaries are already in pixel space
        kb_px = keyboard_region.key_boundaries

        self.finger_assigner = GaussianFingerAssigner(
            kb_px,
            sigma=self.config.assignment.sigma,
            candidate_range=self.config.assignment.candidate_keys
        )
        
        # Set keyboard center for hand separation (pixel space)
        key_cx = sorted(self.finger_assigner.key_centers.values(), key=lambda c: c[0])
        self.hand_separator.set_keyboard_center(key_cx[len(key_cx) // 2][0])
        
        # Stage 2: Live Hand Detection (MediaPipe on raw video)
        logger.info("Stage 2: Live Hand Detection (MediaPipe)")
        live_cfg = LiveDetectionConfig(
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            frame_stride=2,
            static_image_mode=False,
        )
        live_det = LiveHandDetector(config=live_cfg)
        left_landmarks, right_landmarks = live_det.detect_from_video(video_path)
        
        if left_landmarks.size > 0:
            left_landmarks = self.temporal_filter.process(left_landmarks)
            logger.info(f"Left hand: {len(left_landmarks)} frames, "
                        f"detection rate: {LiveHandDetector.detection_rate(left_landmarks):.1%}")
        
        if right_landmarks.size > 0:
            right_landmarks = self.temporal_filter.process(right_landmarks)
            logger.info(f"Right hand: {len(right_landmarks)} frames, "
                        f"detection rate: {LiveHandDetector.detection_rate(right_landmarks):.1%}")
        
        # Scale landmarks from [0,1] to pixel space
        frame_w, frame_h = 1920, 1080  # PianoVAM resolution
        if left_landmarks.size > 0:
            left_landmarks = left_landmarks.copy()
            left_landmarks[:, :, 0] *= frame_w
            left_landmarks[:, :, 1] *= frame_h
        if right_landmarks.size > 0:
            right_landmarks = right_landmarks.copy()
            right_landmarks[:, :, 0] *= frame_w
            right_landmarks[:, :, 1] *= frame_h
        
        # Stage 3: Finger Assignment
        # Try BOTH hands for every key, pick the one with higher confidence.
        # The max-distance gate inside the assigner returns None when the
        # hand is clearly not near the key, preventing false assignments.
        logger.info("Stage 3: Finger Assignment")
        synced_events = self.midi_sync.sync_events(midi_events)
        logger.info(f"Synced {len(synced_events)} MIDI events")
        
        assignments = []
        
        for event in synced_events:
            frame_idx = event.frame_idx
            key_idx = event.key_idx
            
            if key_idx not in self.finger_assigner.key_centers:
                continue
            
            # Try both hands
            asgn_r = None
            if frame_idx < len(right_landmarks):
                lm = right_landmarks[frame_idx]
                if not np.any(np.isnan(lm)):
                    asgn_r = self.finger_assigner.assign_from_landmarks(
                        lm, key_idx, 'right', frame_idx, event.onset_time
                    )
            
            asgn_l = None
            if frame_idx < len(left_landmarks):
                lm = left_landmarks[frame_idx]
                if not np.any(np.isnan(lm)):
                    asgn_l = self.finger_assigner.assign_from_landmarks(
                        lm, key_idx, 'left', frame_idx, event.onset_time
                    )
            
            # Pick the better assignment (higher raw Gaussian confidence)
            cands = [a for a in (asgn_r, asgn_l) if a is not None]
            if cands:
                assignments.append(max(cands, key=lambda a: a.confidence))
        
        logger.info(f"Assigned fingers to {len(assignments)} notes")
        
        return assignments
    
    @staticmethod
    def _project_keys_to_pixel_space(key_boundaries_warped, homography):
        """Project key bounding boxes from warped space to pixel space."""
        H_inv = np.linalg.inv(homography)
        result = {}
        for k, (x1, y1, x2, y2) in key_boundaries_warped.items():
            cy = (y1 + y2) / 2.0
            pts_w = np.array([[x1, cy, 1.0],
                              [x2, cy, 1.0],
                              [(x1 + x2) / 2.0, cy, 1.0]], dtype=np.float64)
            pts_p = (H_inv @ pts_w.T).T
            pts_p = pts_p[:, :2] / pts_p[:, 2:3]
            lx, rx = pts_p[0, 0], pts_p[1, 0]
            cy_px = pts_p[2, 1]
            result[k] = (lx, cy_px - 5.0, rx, cy_px + 5.0)
        return result

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

