#!/usr/bin/env python
"""
Batch Preprocessing Script

Preprocesses all samples in the PianoVAM dataset:
1. Detect keyboard regions
2. Filter hand skeletons
3. Extract fingertip positions
4. Create processed data files

Usage:
    python scripts/preprocess_all.py --config configs/default.yaml
"""

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PianoVAMDataset
from src.keyboard.detector import KeyboardDetector
from src.hand.skeleton_loader import SkeletonLoader
from src.hand.temporal_filter import TemporalFilter
from src.hand.fingertip_extractor import FingertipExtractor
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def preprocess_sample(
    sample,
    dataset: PianoVAMDataset,
    keyboard_detector: KeyboardDetector,
    skeleton_loader: SkeletonLoader,
    temporal_filter: TemporalFilter,
    fingertip_extractor: FingertipExtractor,
    output_dir: Path
) -> dict:
    """Preprocess a single sample."""
    result = {
        'id': sample.id,
        'success': False,
        'error': None
    }
    
    try:
        # 1. Load and process keyboard
        corners = sample.metadata.get('keyboard_corners', {})
        if corners:
            keyboard_region = keyboard_detector.detect_from_corners(corners)
            result['keyboard'] = {
                'bbox': keyboard_region.bbox,
                'num_keys': len(keyboard_region.key_boundaries)
            }
        
        # 2. Load skeleton data
        skeleton_data = dataset.load_skeleton(sample)
        hands = skeleton_loader._parse_json(skeleton_data)
        
        # 3. Convert to arrays and filter
        processed_hands = {}
        for hand_type in ['left', 'right']:
            if hands[hand_type]:
                landmarks_array = skeleton_loader.to_array(hands[hand_type])
                filtered = temporal_filter.process(landmarks_array)
                processed_hands[hand_type] = filtered.tolist()
        
        result['hands'] = {
            'left_frames': len(hands['left']),
            'right_frames': len(hands['right'])
        }
        
        # 4. Save processed data
        output_path = output_dir / f"{sample.id}_processed.json"
        with open(output_path, 'w') as f:
            json.dump({
                'id': sample.id,
                'keyboard': result.get('keyboard', {}),
                'left_hand': processed_hands.get('left', []),
                'right_hand': processed_hands.get('right', [])
            }, f)
        
        result['success'] = True
        result['output_path'] = str(output_path)
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {sample.id}: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess PianoVAM dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which split to process"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    keyboard_detector = KeyboardDetector({
        'canny_low': config.keyboard.canny_low,
        'canny_high': config.keyboard.canny_high,
        'hough_threshold': config.keyboard.hough_threshold
    })
    
    skeleton_loader = SkeletonLoader()
    
    temporal_filter = TemporalFilter(
        hampel_window=config.hand.hampel_window,
        hampel_threshold=config.hand.hampel_threshold,
        max_interpolation_gap=config.hand.interpolation_max_gap,
        savgol_window=config.hand.savgol_window,
        savgol_order=config.hand.savgol_order
    )
    
    fingertip_extractor = FingertipExtractor()
    
    # Process splits
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    
    all_results = []
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        dataset = PianoVAMDataset(split=split, streaming=False)
        num_samples = len(dataset)
        if args.max_samples:
            num_samples = min(num_samples, args.max_samples)
        
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(num_samples), desc=f"Processing {split}"):
            sample = dataset[i]
            result = preprocess_sample(
                sample, dataset, keyboard_detector, skeleton_loader,
                temporal_filter, fingertip_extractor, split_dir
            )
            all_results.append(result)
    
    # Summary
    success_count = sum(1 for r in all_results if r['success'])
    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Success: {success_count}/{len(all_results)}")
    
    # Save summary
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

