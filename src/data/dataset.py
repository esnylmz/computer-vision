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
        split: One of 'train', 'validation' (or 'valid'/'val'), 'test'
        cache_dir: Directory to cache downloaded files
    
    Note:
        The dataset uses 'validation' for the validation split. The loader automatically
        maps 'valid' and 'val' to 'validation' for convenience.
    """
    
    DATASET_NAME = "PianoVAM/PianoVAM_v1.0"
    BASE_URL = "https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0/resolve/main/"
    
    def __init__(
        self, 
        split: str = 'train', 
        cache_dir: str = './data/cache',
        streaming: bool = True,
        timeout: int = 120,
        max_retries: int = 5,
        max_samples: Optional[int] = None
    ):
        """
        Initialize PianoVAM dataset.
        
        Args:
            split: Dataset split ('train', 'validation'/'valid'/'val', 'test')
            cache_dir: Directory to cache downloaded files
            streaming: If True, stream data without downloading full dataset
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retries for network errors (default: 5)
            max_samples: Maximum number of samples to load (None = all). Useful for exploration.
        
        Note:
            'valid' and 'val' are automatically mapped to 'validation'.
        """
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.streaming = streaming
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_samples = max_samples
        
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
            if self.max_samples is not None:
                self._samples = self._samples[:self.max_samples]
        else:
            self._samples = None
            self._iterator = None
    
    # Valid split names that users can request
    # Maps user input -> value in 'split' column of dataset
    SPLIT_COLUMN_MAP = {
        'train': 'train',
        'validation': 'valid',  # HF API uses 'validation', but column has 'valid'
        'valid': 'valid',
        'val': 'valid',
        'test': 'test',
    }
    
    def _get_split_column_value(self, split: str) -> str:
        """
        Convert user's split name to the value used in the dataset's 'split' column.
        
        The PianoVAM dataset has a 'split' column with values: 'train', 'valid', 'test'
        Note: The column uses 'valid' NOT 'validation'!
        
        Args:
            split: User-provided split name ('train', 'validation', 'valid', 'val', 'test')
            
        Returns:
            The corresponding value in the 'split' column ('train', 'valid', or 'test')
        """
        split_lower = split.lower().strip()
        
        if split_lower not in self.SPLIT_COLUMN_MAP:
            valid_inputs = sorted(set(self.SPLIT_COLUMN_MAP.keys()))
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Valid options: {valid_inputs}. "
                f"Use 'validation', 'valid', or 'val' for validation split."
            )
        
        return self.SPLIT_COLUMN_MAP[split_lower]
    
    def _load_with_retry(
        self, 
        split: str, 
        streaming: bool, 
        download_config: Optional[Any],
        _recursion_depth: int = 0
    ):
        """
        Load dataset and filter by split column.
        
        The PianoVAM dataset has a quirk: HuggingFace API splits (train/validation/test) 
        don't always work in streaming mode. We try API splits first, then fall back to 
        loading all data and filtering by the 'split' column.
        
        Args:
            _recursion_depth: Internal parameter to prevent infinite recursion
        """
        # Prevent infinite recursion
        if _recursion_depth > 1:
            raise RuntimeError("Too many recursive calls in dataset loading. This indicates a bug.")
        
        # Get the column value to filter by
        split_column_value = self._get_split_column_value(split)
        
        # Map to HuggingFace API split names
        hf_split_map = {
            'train': 'train',
            'valid': 'validation',  # Column has 'valid', API uses 'validation'
            'test': 'test'
        }
        hf_split_name = hf_split_map.get(split_column_value, split_column_value)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    print(f"Loading dataset split '{split}' (column value: '{split_column_value}')...")
                else:
                    print(f"Retrying (attempt {attempt + 1}/{self.max_retries})...")
                
                # Strategy 1: Try using HuggingFace API splits directly
                try:
                    dataset = load_dataset(
                        self.DATASET_NAME,
                        split=hf_split_name,
                        streaming=streaming,
                        download_config=download_config
                    )
                    # Success! Return it (it should already be filtered by HuggingFace)
                    return dataset
                except (ValueError, KeyError) as split_error:
                    # API split doesn't work, fall back to filtering
                    error_str = str(split_error).lower()
                    if 'bad split' in error_str or 'unknown split' in error_str or 'not found' in error_str:
                        print(f"Note: HuggingFace split '{hf_split_name}' not available, using column filtering...")
                    else:
                        raise
                
                # Strategy 2: Load all data and filter by 'split' column
                if streaming:
                    # Strategy: Load all splits and filter by 'split' column
                    # HuggingFace streaming mode sometimes only exposes 'train' split
                    # So we need to load all available splits and filter
                    try:
                        # Try loading without split parameter to get all splits
                        full_dataset = load_dataset(
                            self.DATASET_NAME,
                            streaming=True,
                            download_config=download_config
                        )
                        
                        # Check what splits we got
                        if isinstance(full_dataset, dict):
                            split_keys = list(full_dataset.keys())
                            print(f"Found splits: {split_keys}")
                        else:
                            # Single dataset, wrap it
                            split_keys = ['default']
                            full_dataset = {'default': full_dataset}
                        
                        # Create a filtered iterator that chains all splits
                        class FilteredStreamingDataset:
                            """A re-iterable filtered dataset wrapper."""
                            def __init__(self, dataset_dict, split_keys, filter_value):
                                self._dataset_dict = dataset_dict
                                self._split_keys = split_keys
                                self._filter_value = filter_value
                            
                            def __iter__(self):
                                # Iterate through each split and filter
                                for split_key in self._split_keys:
                                    split_data = self._dataset_dict[split_key]
                                    for item in split_data:
                                        # Check both 'split' and 'Split' (case variations)
                                        item_split = item.get('split') or item.get('Split', '')
                                        if item_split == self._filter_value:
                                            yield item
                        
                        return FilteredStreamingDataset(full_dataset, split_keys, split_column_value)
                        
                    except Exception as inner_e:
                        # If we can't load all splits in streaming mode, and user wants non-train split,
                        # automatically switch to non-streaming mode
                        if split_column_value != 'train':
                            print(f"Note: Streaming mode only exposes 'train' split. Switching to non-streaming mode...")
                            # Recursively call with streaming=False
                            return self._load_with_retry(split, streaming=False, download_config=download_config, _recursion_depth=_recursion_depth + 1)
                        # For 'train' split, we can use the 'train' split directly
                        print(f"Note: Loading 'train' split...")
                        dataset = load_dataset(
                            self.DATASET_NAME,
                            split='train',
                            streaming=True,
                            download_config=download_config
                        )
                        # Filter to ensure we only get train samples (in case it has all data)
                        return dataset.filter(lambda x: (x.get('split') or x.get('Split', '')) == split_column_value)
                else:
                    # Non-streaming: load all splits
                    try:
                        full_dataset = load_dataset(
                            self.DATASET_NAME,
                            download_config=download_config
                        )
                        # Concatenate all splits
                        from datasets import concatenate_datasets
                        all_splits = [full_dataset[s] for s in full_dataset.keys()]
                        combined = concatenate_datasets(all_splits)
                        # Filter by split column
                        return combined.filter(lambda x: x.get('split', '') == split_column_value)
                    except Exception:
                        # Fallback to just train
                        dataset = load_dataset(
                            self.DATASET_NAME,
                            split='train',
                            streaming=False,
                            download_config=download_config
                        )
                        return dataset.filter(lambda x: x.get('split', '') == split_column_value)
                
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                error_str = str(e).lower()
                
                # Check for HTTP errors (502, 503, 504, etc.)
                is_http_error = (
                    'HTTPError' in error_name or 
                    'HfHubHTTPError' in error_name or
                    '502' in error_str or 
                    '503' in error_str or 
                    '504' in error_str or
                    'bad gateway' in error_str or
                    'service unavailable' in error_str or
                    'gateway timeout' in error_str
                )
                
                is_timeout = 'Timeout' in error_name or 'timeout' in error_str
                is_connection = 'Connection' in error_name or 'connection' in error_str
                
                if is_timeout or is_http_error or is_connection:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    if is_timeout:
                        print(f"Timeout error. Retrying in {wait_time}s...")
                    elif is_http_error:
                        print(f"HTTP error (likely temporary server issue). Retrying in {wait_time}s...")
                    else:
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
            f"1. Check your internet connection\n"
            f"2. Try increasing timeout (current: {self.timeout}s)\n"
            f"3. Set HF_TOKEN in Colab secrets for better rate limits"
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
            count = 0
            for row in self.hf_dataset:
                yield self._row_to_sample(row)
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    break
            
            if count == 0:
                raise ValueError(
                    f"No samples found for split '{self.split}'. "
                    f"This might mean:\n"
                    f"  1. The dataset is empty\n"
                    f"  2. The 'split' column filtering didn't match any rows\n"
                    f"  3. The expected split value '{self._get_split_column_value(self.split)}' doesn't exist in the data\n"
                    f"Try loading with streaming=False to debug, or check the dataset structure."
                )
    
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
    streaming: bool = False,
    max_samples: Optional[int] = None
) -> PianoVAMDataset:
    """
    Convenience function to load PianoVAM dataset.
    
    Args:
        split: Dataset split
        cache_dir: Cache directory
        streaming: Whether to stream data
        max_samples: Maximum number of samples to load (None = all)
        
    Returns:
        PianoVAMDataset instance
    """
    return PianoVAMDataset(
        split=split, 
        cache_dir=cache_dir, 
        streaming=streaming,
        max_samples=max_samples
    )

