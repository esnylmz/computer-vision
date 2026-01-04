"""
PianoVAM Dataset Loader

Loads the PianoVAM dataset from HuggingFace and provides convenient access
to video, audio, MIDI, and hand skeleton data for each recording.

Usage:
    from src.data.dataset import PianoVAMDataset
    
    dataset = PianoVAMDataset(split='train')
    sample = dataset[0]
    # sample contains: video_path, audio_path, midi_path, skeleton_path, metadata
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import time

try:
    from datasets import load_dataset
    from datasets import DownloadConfig
except ImportError:
    load_dataset = None
    DownloadConfig = None


@dataclass
class PianoVAMSample:
    """Single sample from PianoVAM dataset."""
    id: str
    video_path: str
    audio_path: str
    midi_path: str
    skeleton_path: str
    tsv_path: str
    metadata: Dict[str, Any]


class PianoVAMDataset:
    """
    PianoVAM Dataset wrapper for piano fingering detection.
    
    Attributes:
        split: One of 'train', 'valid' (or 'validation'), 'test'
        cache_dir: Directory to cache downloaded files
    
    Note:
        The dataset uses 'valid' for the validation split. The loader automatically
        maps 'validation' and 'val' to 'valid' for convenience.
    """
    
    DATASET_NAME = "PianoVAM/PianoVAM_v1.0"
    BASE_URL = "https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0/resolve/main/"
    
    def __init__(
        self, 
        split: str = 'train', 
        cache_dir: str = './data/cache',
        streaming: bool = True,
        timeout: int = 120,
        max_retries: int = 5
    ):
        """
        Initialize PianoVAM dataset.
        
        Args:
            split: Dataset split ('train', 'valid'/'validation', 'test')
            cache_dir: Directory to cache downloaded files
            streaming: If True, stream data without downloading full dataset
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retries for network errors (default: 5)
        
        Note:
            'validation' and 'val' are automatically mapped to 'valid'.
        """
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.streaming = streaming
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load dataset metadata from HuggingFace
        if load_dataset is None:
            raise ImportError("Please install 'datasets' package: pip install datasets")
        
        # Configure download with extended timeout and retries
        download_config = None
        if DownloadConfig is not None:
            download_config = DownloadConfig(
                max_retries=max_retries,
                resume_download=True,
            )
        
        # Load with retry logic for network issues
        self.hf_dataset = self._load_with_retry(
            split=split,
            streaming=streaming,
            download_config=download_config
        )
        
        # Convert to list if not streaming for indexing
        if not streaming:
            self._samples = list(self.hf_dataset)
        else:
            self._samples = None
            self._iterator = None
    
    def _normalize_split_name(self, split: str) -> str:
        """Normalize split name to match dataset conventions."""
        # Map common variations to standard names
        # According to HuggingFace, the dataset uses 'valid' not 'validation'
        split_map = {
            'validation': 'valid',
            'val': 'valid',
            'train': 'train',
            'test': 'test'
        }
        return split_map.get(split.lower(), split.lower())
    
    def _load_with_retry(
        self, 
        split: str, 
        streaming: bool, 
        download_config: Optional[Any]
    ):
        """Load dataset with retry logic for handling timeouts."""
        # Normalize split name (validation -> valid)
        normalized_split = self._normalize_split_name(split)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    print(f"Loading dataset split '{normalized_split}' (attempt {attempt + 1}/{self.max_retries})...")
                else:
                    print(f"Retrying (attempt {attempt + 1}/{self.max_retries})...")
                
                # Load dataset - removed trust_remote_code as it's deprecated
                dataset = load_dataset(
                    self.DATASET_NAME,
                    split=normalized_split,
                    streaming=streaming,
                    download_config=download_config
                )
                return dataset
            except ValueError as e:
                # Check if it's a split name error
                error_str = str(e)
                if 'Bad split' in error_str or 'split' in error_str.lower():
                    # Try to get available splits
                    try:
                        full_dataset = load_dataset(
                            self.DATASET_NAME,
                            download_config=download_config,
                            download_mode="reuse_dataset_if_exists"
                        )
                        if isinstance(full_dataset, dict):
                            available = list(full_dataset.keys())
                            raise ValueError(
                                f"Split '{split}' (mapped to '{normalized_split}') not found. "
                                f"Available splits: {available}. "
                                f"Note: Use 'valid' instead of 'validation'."
                            )
                    except Exception as inner_e:
                        if 'Bad split' not in str(inner_e):
                            # Re-raise original error if inner call fails for different reason
                            raise e
                    raise
                else:
                    # For other ValueError, raise immediately
                    raise
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                if 'Timeout' in error_name or 'timeout' in str(e).lower():
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"Timeout error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif 'Connection' in error_name or 'connection' in str(e).lower():
                    wait_time = 2 ** attempt
                    print(f"Connection error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # For non-network errors, raise immediately
                    raise
        
        # If all retries failed, raise the last error
        raise RuntimeError(
            f"Failed to load dataset after {self.max_retries} attempts. "
            f"Last error: {last_error}\n\n"
            f"Tips:\n"
            f"1. Try using streaming=True to avoid downloading all files at once\n"
            f"2. Check your internet connection\n"
            f"3. Try increasing timeout (current: {self.timeout}s)\n"
            f"4. Set HF_TOKEN in Colab secrets for better rate limits\n"
            f"5. Use 'valid' instead of 'validation' for the validation split"
        )
        
    def __len__(self) -> int:
        if self._samples is not None:
            return len(self._samples)
        # For streaming, we don't know the length
        return -1
    
    def __getitem__(self, idx: int) -> PianoVAMSample:
        if self._samples is None:
            raise ValueError("Cannot index streaming dataset. Set streaming=False.")
        
        row = self._samples[idx]
        return self._row_to_sample(row)
    
    def __iter__(self):
        if self._samples is not None:
            for row in self._samples:
                yield self._row_to_sample(row)
        else:
            for row in self.hf_dataset:
                yield self._row_to_sample(row)
    
    def _row_to_sample(self, row: Dict) -> PianoVAMSample:
        """Convert a dataset row to PianoVAMSample."""
        # Handle different possible column names
        video_path = row.get('video_path', row.get('Video', ''))
        audio_path = row.get('audio_path', row.get('Audio', ''))
        midi_path = row.get('midi_path', row.get('Midi', ''))
        skeleton_path = row.get('handskeleton_path', row.get('Handskeleton', ''))
        tsv_path = row.get('tsv_path', row.get('TSV', ''))
        
        return PianoVAMSample(
            id=row.get('id', row.get('ID', str(hash(video_path)))),
            video_path=self.BASE_URL + video_path if video_path else '',
            audio_path=self.BASE_URL + audio_path if audio_path else '',
            midi_path=self.BASE_URL + midi_path if midi_path else '',
            skeleton_path=self.BASE_URL + skeleton_path if skeleton_path else '',
            tsv_path=self.BASE_URL + tsv_path if tsv_path else '',
            metadata={
                'composer': row.get('composer', row.get('Composer', 'Unknown')),
                'piece': row.get('piece', row.get('Piece', 'Unknown')),
                'performer': row.get('P1_name', row.get('Performer', 'Unknown')),
                'skill_level': row.get('P1_skill', row.get('Skill', 'Unknown')),
                'duration': row.get('duration', row.get('Duration', 0)),
                'keyboard_corners': self._parse_corners(row)
            }
        )
    
    def _parse_corners(self, row: Dict) -> Dict[str, Tuple[int, int]]:
        """Parse keyboard corner coordinates from row."""
        corners = {}
        for key in ['Point_LT', 'Point_RT', 'Point_RB', 'Point_LB']:
            alt_key = key.replace('Point_', '')  # Try without 'Point_' prefix
            value = row.get(key, row.get(alt_key, '0, 0'))
            if isinstance(value, str):
                try:
                    x, y = value.split(', ')
                    corners[key.replace('Point_', '')] = (int(x), int(y))
                except ValueError:
                    corners[key.replace('Point_', '')] = (0, 0)
            else:
                corners[key.replace('Point_', '')] = (0, 0)
        return corners
    
    def download_file(self, url: str, local_path: Optional[Path] = None) -> Path:
        """
        Download a file from URL to local cache.
        
        Args:
            url: URL to download from
            local_path: Optional local path, otherwise uses cache_dir
            
        Returns:
            Path to downloaded file
        """
        if local_path is None:
            filename = url.split('/')[-1]
            local_path = self.cache_dir / filename
        
        if local_path.exists():
            return local_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return local_path
    
    def load_skeleton(self, sample: PianoVAMSample) -> Dict:
        """
        Load hand skeleton data from JSON file.
        
        Args:
            sample: PianoVAMSample object
            
        Returns:
            Dict containing hand landmark data with structure:
            {
                'frames': List[Dict] with 'left_hand' and 'right_hand' landmarks
            }
        """
        if not sample.skeleton_path:
            raise ValueError(f"No skeleton path for sample {sample.id}")
        
        # Download if needed
        local_path = self.download_file(sample.skeleton_path)
        
        with open(local_path, 'r') as f:
            skeleton_data = json.load(f)
        
        return skeleton_data
    
    def load_tsv_annotations(self, sample: PianoVAMSample) -> pd.DataFrame:
        """
        Load TSV annotations with columns:
        onset, key_offset, frame_offset, note, velocity
        
        Args:
            sample: PianoVAMSample object
            
        Returns:
            DataFrame with note annotations
        """
        if not sample.tsv_path:
            raise ValueError(f"No TSV path for sample {sample.id}")
        
        local_path = self.download_file(sample.tsv_path)
        
        df = pd.read_csv(
            local_path, 
            sep='\t',
            names=['onset', 'key_offset', 'frame_offset', 'note', 'velocity'],
            header=None
        )
        
        return df
    
    def get_sample_by_id(self, sample_id: str) -> Optional[PianoVAMSample]:
        """Get a sample by its ID."""
        for sample in self:
            if sample.id == sample_id:
                return sample
        return None


def load_pianovam(
    split: str = 'train',
    cache_dir: str = './data/cache',
    streaming: bool = False
) -> PianoVAMDataset:
    """
    Convenience function to load PianoVAM dataset.
    
    Args:
        split: Dataset split
        cache_dir: Cache directory
        streaming: Whether to stream data
        
    Returns:
        PianoVAMDataset instance
    """
    return PianoVAMDataset(split=split, cache_dir=cache_dir, streaming=streaming)

