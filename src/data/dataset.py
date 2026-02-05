"""
PianoVAM Dataset Loader

Loads the PianoVAM dataset metadata from HuggingFace and provides convenient
access to video, audio, MIDI, and hand skeleton data for each recording.

Architecture:
    The PianoVAM HuggingFace repo stores files in folders:
        Audio/, Video/, MIDI/, Handskeleton/, TSV/
    and a metadata JSON file (metadata_v1.json) that describes each sample.

    HuggingFace's `load_dataset()` does NOT properly load this dataset's
    metadata columns (it only sees Audio files). We therefore download
    `metadata_v1.json` directly and construct sample objects from it.

Usage:
    from src.data.dataset import PianoVAMDataset

    dataset = PianoVAMDataset(split='train')
    for sample in dataset:
        print(sample.id, sample.metadata['composer'])
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

    Loads metadata directly from the HuggingFace repository's metadata_v1.json
    file (a small JSON download) and constructs sample objects with full URLs.
    Individual data files (skeleton, TSV, audio, etc.) are downloaded on demand.

    Attributes:
        split: One of 'train', 'validation' (or 'valid'/'val'), 'test'
        cache_dir: Directory to cache downloaded files

    Split mapping (metadata_v1.json 'split' column values):
        train  → 73 samples
        valid  → 19 samples
        test   → 14 samples
        (+ 1 special sample, excluded by default)
    """

    DATASET_NAME = "PianoVAM/PianoVAM_v1.0"
    BASE_URL = "https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0/resolve/main/"
    METADATA_URL = BASE_URL + "metadata_v1.json"

    # Maps user-facing split names → values in the 'split' column of metadata
    SPLIT_COLUMN_MAP = {
        'train': 'train',
        'validation': 'valid',
        'valid': 'valid',
        'val': 'valid',
        'test': 'test',
    }

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
            streaming: Kept for API compatibility (metadata always loaded eagerly)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retries for network errors (default: 5)
            max_samples: Maximum number of samples to load (None = all)

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

        # Resolve the split column value
        self._split_value = self._get_split_column_value(split)

        # Load metadata from HuggingFace
        metadata = self._load_metadata()

        # Filter by split and apply max_samples
        self._samples: List[Dict[str, Any]] = []
        for key in sorted(metadata.keys(), key=lambda k: int(k)):
            entry = metadata[key]
            if entry.get('split', '') == self._split_value:
                self._samples.append(entry)
                if self.max_samples is not None and len(self._samples) >= self.max_samples:
                    break

        print(f"  ✅ Loaded {len(self._samples)} '{self._split_value}' samples")

    # ------------------------------------------------------------------
    # Split name resolution
    # ------------------------------------------------------------------

    def _get_split_column_value(self, split: str) -> str:
        """
        Convert user's split name to the value used in metadata's 'split' column.

        The PianoVAM metadata uses: 'train', 'valid', 'test'
        Note: The column uses 'valid' NOT 'validation'!
        """
        split_lower = split.lower().strip()

        if split_lower not in self.SPLIT_COLUMN_MAP:
            valid_inputs = sorted(set(self.SPLIT_COLUMN_MAP.keys()))
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Valid options: {valid_inputs}. "
                f"Use 'validation', 'valid', or 'val' for the validation split."
            )

        return self.SPLIT_COLUMN_MAP[split_lower]

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Dict]:
        """
        Download and cache metadata_v1.json from HuggingFace.

        The file is a dict keyed by string indices ("0" .. "106"),
        each value being a dict with fields:
            record_time, split, composer, piece, performance_method,
            performance_type, duration, P1_name, P1_skill, ...,
            Point_LT, Point_RT, Point_RB, Point_LB
        """
        # Try cache first
        cache_path = self.cache_dir / "metadata_v1.json"
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Download with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    print(f"Downloading PianoVAM metadata ...")
                else:
                    print(f"  Retry {attempt + 1}/{self.max_retries} ...")

                resp = requests.get(self.METADATA_URL, timeout=self.timeout)
                resp.raise_for_status()
                metadata = resp.json()

                # Cache for future use
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f)

                return metadata

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_retryable = any(kw in error_str for kw in [
                    'timeout', 'connection', '502', '503', '504',
                    'bad gateway', 'service unavailable', 'gateway timeout'
                ])
                if is_retryable and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Network error, retrying in {wait}s ...")
                    time.sleep(wait)
                elif not is_retryable:
                    raise

        raise RuntimeError(
            f"Failed to download metadata after {self.max_retries} attempts.\n"
            f"Last error: {last_error}\n"
            f"URL: {self.METADATA_URL}"
        )

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> PianoVAMSample:
        return self._row_to_sample(self._samples[idx])

    def __iter__(self):
        for row in self._samples:
            yield self._row_to_sample(row)

    # ------------------------------------------------------------------
    # Row → Sample conversion
    # ------------------------------------------------------------------

    def _row_to_sample(self, row: Dict) -> PianoVAMSample:
        """Convert a metadata dict entry to PianoVAMSample."""
        record_time = row.get('record_time', '')

        # Build file paths from record_time
        video_path = f"Video/{record_time}.mp4" if record_time else ''
        audio_path = f"Audio/{record_time}.wav" if record_time else ''
        midi_path = f"MIDI/{record_time}.mid" if record_time else ''
        skeleton_path = f"Handskeleton/{record_time}.json" if record_time else ''
        tsv_path = f"TSV/{record_time}.tsv" if record_time else ''

        return PianoVAMSample(
            id=record_time or str(hash(str(row))),
            video_path=self.BASE_URL + video_path if video_path else '',
            audio_path=self.BASE_URL + audio_path if audio_path else '',
            midi_path=self.BASE_URL + midi_path if midi_path else '',
            skeleton_path=self.BASE_URL + skeleton_path if skeleton_path else '',
            tsv_path=self.BASE_URL + tsv_path if tsv_path else '',
            metadata={
                'composer': row.get('composer', 'Unknown'),
                'piece': row.get('piece', 'Unknown'),
                'performer': row.get('P1_name', 'Unknown'),
                'skill_level': row.get('P1_skill', 'Unknown'),
                'duration': row.get('duration', ''),
                'performance_method': row.get('performance_method', ''),
                'performance_type': row.get('performance_type', ''),
                'record_time': record_time,
                'keyboard_corners': self._parse_corners(row)
            }
        )

    def _parse_corners(self, row: Dict) -> Dict[str, Tuple[int, int]]:
        """Parse keyboard corner coordinates from row."""
        corners = {}
        for key in ['Point_LT', 'Point_RT', 'Point_RB', 'Point_LB']:
            value = row.get(key, '0, 0')
            if isinstance(value, str):
                try:
                    x, y = value.split(', ')
                    corners[key.replace('Point_', '')] = (int(x), int(y))
                except ValueError:
                    corners[key.replace('Point_', '')] = (0, 0)
            else:
                corners[key.replace('Point_', '')] = (0, 0)
        return corners

    # ------------------------------------------------------------------
    # File download helpers
    # ------------------------------------------------------------------

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

        response = requests.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=local_path.name) as pbar:
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
        """Get a sample by its ID (record_time)."""
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
        streaming: Kept for API compatibility
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
