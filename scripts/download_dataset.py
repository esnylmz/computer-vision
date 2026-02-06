#!/usr/bin/env python
"""
Download PianoVAM Dataset

Downloads the PianoVAM dataset from HuggingFace and caches it locally.

Usage:
    python scripts/download_dataset.py --output_dir ./data --split all
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PianoVAMDataset


def download_split(split: str, output_dir: Path, max_samples: int = None):
    """Download a specific split of the dataset."""
    print(f"\nDownloading {split} split...")
    
    dataset = PianoVAMDataset(
        split=split,
        cache_dir=str(output_dir / "cache"),
        streaming=False
    )
    
    num_samples = len(dataset)
    if max_samples:
        num_samples = min(num_samples, max_samples)
    
    print(f"Found {len(dataset)} samples, downloading {num_samples}")
    
    for i in tqdm(range(num_samples), desc=f"Downloading {split}"):
        sample = dataset[i]
        
        try:
            # Download video
            if sample.video_path:
                dataset.download_file(
                    sample.video_path,
                    output_dir / "videos" / f"{sample.id}.mp4"
                )
            
            # Download skeleton
            if sample.skeleton_path:
                dataset.download_file(
                    sample.skeleton_path,
                    output_dir / "skeletons" / f"{sample.id}.json"
                )
            
            # Download TSV
            if sample.tsv_path:
                dataset.download_file(
                    sample.tsv_path,
                    output_dir / "annotations" / f"{sample.id}.tsv"
                )
                
        except Exception as e:
            print(f"Error downloading sample {sample.id}: {e}")
            continue
    
    print(f"Completed {split} split!")


def main():
    parser = argparse.ArgumentParser(description="Download PianoVAM dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which split to download"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (for testing)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Create directories
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "skeletons").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading PianoVAM dataset to {output_dir}")
    
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    
    for split in splits:
        download_split(split, output_dir, args.max_samples)
    
    print("\nDownload complete!")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

