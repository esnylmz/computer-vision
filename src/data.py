"""
src/data.py — Dataset loading, sampling, video-level splitting, manifest creation.

Handles:
  - Loading PianoVAM v1.0 metadata from HuggingFace
  - Sampling N videos from the TRAIN split only
  - Clipping to first `clip_duration` seconds
  - Video-level train / test split (no frame leakage)
  - Manifest creation (JSON) with all needed paths
  - Download caching (videos downloaded once)
  - Leakage assertion

Usage:
    from src.data import sample_and_split, download_manifest_videos, save_manifest

    manifest, dataset = sample_and_split(N=3, clip_duration=20, frame_step=10)
    local_paths = download_manifest_videos(manifest, dataset)
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# PianoVAM Sample & Dataset
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PianoVAMSample:
    """Single sample from the PianoVAM dataset."""
    id: str
    video_path: str          # full URL
    audio_path: str
    midi_path: str
    skeleton_path: str
    tsv_path: str
    metadata: Dict[str, Any]


class PianoVAMDataset:
    """
    PianoVAM v1.0 dataset wrapper.

    Downloads metadata_v1.json from HuggingFace (cached locally)
    and provides convenient access to samples filtered by split.

    Split mapping (metadata 'split' column):
        train  → ~73 samples
        valid  → ~19 samples
        test   → ~14 samples
    """

    DATASET_NAME = "PianoVAM/PianoVAM_v1.0"
    BASE_URL = (
        "https://huggingface.co/datasets/PianoVAM/PianoVAM_v1.0/resolve/main/"
    )
    METADATA_URL = BASE_URL + "metadata_v1.json"

    SPLIT_MAP = {
        "train": "train",
        "validation": "valid",
        "valid": "valid",
        "val": "valid",
        "test": "test",
    }

    def __init__(
        self,
        split: str = "train",
        cache_dir: str = "./data/cache",
        max_samples: Optional[int] = None,
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries

        split_value = self.SPLIT_MAP.get(split.lower().strip())
        if split_value is None:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Valid options: {list(self.SPLIT_MAP.keys())}"
            )
        self._split_value = split_value

        metadata = self._load_metadata()
        self._samples: List[Dict[str, Any]] = []
        for key in sorted(metadata.keys(), key=lambda k: int(k)):
            entry = metadata[key]
            if entry.get("split", "") == self._split_value:
                self._samples.append(entry)
                if max_samples is not None and len(self._samples) >= max_samples:
                    break

        print(f"  Loaded {len(self._samples)} '{self._split_value}' samples")

    # ── metadata download ─────────────────────────────────────────

    def _load_metadata(self) -> Dict[str, Dict]:
        cache_path = self.cache_dir / "metadata_v1.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                msg = (
                    "Downloading PianoVAM metadata ..."
                    if attempt == 0
                    else f"  Retry {attempt + 1}/{self.max_retries} ..."
                )
                print(msg)
                resp = requests.get(self.METADATA_URL, timeout=self.timeout)
                resp.raise_for_status()
                metadata = resp.json()

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f)
                return metadata

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to download metadata after {self.max_retries} attempts.\n"
            f"Last error: {last_error}\n"
            f"URL: {self.METADATA_URL}"
        )

    # ── container protocol ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> PianoVAMSample:
        return self._row_to_sample(self._samples[idx])

    def __iter__(self):
        for row in self._samples:
            yield self._row_to_sample(row)

    # ── row → sample ─────────────────────────────────────────────

    def _row_to_sample(self, row: Dict) -> PianoVAMSample:
        record_time = row.get("record_time", "")
        video_path = f"Video/{record_time}.mp4" if record_time else ""
        audio_path = f"Audio/{record_time}.wav" if record_time else ""
        midi_path = f"MIDI/{record_time}.mid" if record_time else ""
        skeleton_path = f"Handskeleton/{record_time}.json" if record_time else ""
        tsv_path = f"TSV/{record_time}.tsv" if record_time else ""

        return PianoVAMSample(
            id=record_time or str(hash(str(row))),
            video_path=self.BASE_URL + video_path,
            audio_path=self.BASE_URL + audio_path,
            midi_path=self.BASE_URL + midi_path,
            skeleton_path=self.BASE_URL + skeleton_path,
            tsv_path=self.BASE_URL + tsv_path,
            metadata={
                "composer": row.get("composer", "Unknown"),
                "piece": row.get("piece", "Unknown"),
                "performer": row.get("P1_name", "Unknown"),
                "skill_level": row.get("P1_skill", "Unknown"),
                "duration": row.get("duration", ""),
                "performance_method": row.get("performance_method", ""),
                "performance_type": row.get("performance_type", ""),
                "record_time": record_time,
                "keyboard_corners": self._parse_corners(row),
            },
        )

    @staticmethod
    def _parse_corners(row: Dict) -> Dict[str, Tuple[int, int]]:
        corners: Dict[str, Tuple[int, int]] = {}
        for key in ["Point_LT", "Point_RT", "Point_RB", "Point_LB"]:
            value = row.get(key, "0, 0")
            if isinstance(value, str):
                try:
                    x, y = value.split(", ")
                    corners[key.replace("Point_", "")] = (int(x), int(y))
                except ValueError:
                    corners[key.replace("Point_", "")] = (0, 0)
            else:
                corners[key.replace("Point_", "")] = (0, 0)
        return corners

    # ── file download (cached) ────────────────────────────────────

    def download_file(
        self, url: str, local_path: Optional[Path] = None
    ) -> Path:
        """Download a file from *url* into cache. Returns local Path."""
        if local_path is None:
            filename = url.split("/")[-1]
            local_path = self.cache_dir / filename
        else:
            local_path = Path(local_path)

        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(local_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=local_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return local_path

    # ── auxiliary loaders ─────────────────────────────────────────

    def load_skeleton(self, sample: PianoVAMSample) -> Dict:
        """Download + load hand skeleton JSON for *sample*."""
        local = self.download_file(sample.skeleton_path)
        with open(local, "r") as f:
            return json.load(f)

    def load_tsv_annotations(self, sample: PianoVAMSample) -> pd.DataFrame:
        """Download + load TSV annotations (onset, note, velocity, …)."""
        local = self.download_file(sample.tsv_path)
        return pd.read_csv(
            local, sep="\t",
            names=["onset", "key_offset", "frame_offset", "note", "velocity"],
            header=None, comment="#",
        )

    def get_raw_metadata_keys(self) -> List[str]:
        """Print and return the raw key names in the first metadata entry."""
        if self._samples:
            keys = list(self._samples[0].keys())
            print("Raw metadata keys:", keys)
            return keys
        return []

    def get_sample_by_id(self, sample_id: str) -> Optional[PianoVAMSample]:
        """Get a sample by its ID (record_time)."""
        for row in self._samples:
            if row.get("record_time", "") == sample_id:
                return self._row_to_sample(row)
        return None


# ═══════════════════════════════════════════════════════════════════
# Video helpers (lightweight — used for fps verification)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VideoInfo:
    """Basic video metadata read from the file header."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def get_video_info(video_path: Union[str, Path]) -> VideoInfo:
    """Open *video_path* briefly, read metadata, close immediately."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        duration=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        / max(cap.get(cv2.CAP_PROP_FPS), 1),
    )
    cap.release()
    return info


# ═══════════════════════════════════════════════════════════════════
# Manifest creation & video-level splitting
# ═══════════════════════════════════════════════════════════════════

def sample_and_split(
    N: int = 3,
    clip_duration: float = 20.0,
    frame_step: int = 10,
    train_ratio: float = 0.67,
    cache_dir: str = "./data/cache",
    seed: int = 42,
) -> Tuple[pd.DataFrame, PianoVAMDataset]:
    """
    Sample *N* videos from the TRAIN split, create a video-level
    train / test split, and return a manifest DataFrame.

    Manifest columns:
        video_id, split, clip_start, clip_duration, fps,
        frame_step, video_url, midi_url, skeleton_url, tsv_url,
        composer, piece, performer, keyboard_corners

    Returns:
        (manifest DataFrame, PianoVAMDataset instance)
    """
    # 1) load dataset — TRAIN split only
    dataset = PianoVAMDataset(
        split="train", cache_dir=cache_dir, max_samples=N,
    )

    # 2) print raw metadata keys explicitly (requirement)
    print("\n--- Raw metadata keys from PianoVAM ---")
    dataset.get_raw_metadata_keys()

    # 3) collect samples
    samples = [dataset[i] for i in range(len(dataset))]
    print(f"\nSampled {len(samples)} videos from TRAIN split:")
    for s in samples:
        print(f"  {s.id} | {s.metadata['composer']}: {s.metadata['piece']}")

    # 4) video-level split (deterministic)
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n_train = max(1, int(len(samples) * train_ratio))
    train_ids = {samples[i].id for i in indices[:n_train]}
    test_ids = {samples[i].id for i in indices[n_train:]}

    # 5) leakage assertion
    overlap = train_ids & test_ids
    assert len(overlap) == 0, (
        f"DATA LEAKAGE DETECTED — video_ids in both splits: {overlap}"
    )
    print(f"\nVideo-level split: {len(train_ids)} train / {len(test_ids)} test")
    print("Leakage check PASSED (no video_id appears in both train and test)")

    # 6) build manifest rows
    rows: List[Dict[str, Any]] = []
    for s in samples:
        split_label = "train" if s.id in train_ids else "test"
        rows.append(
            {
                "video_id": s.id,
                "split": split_label,
                "clip_start": 0.0,
                "clip_duration": clip_duration,
                "fps": 60,  # PianoVAM default; updated after download
                "frame_step": frame_step,
                "video_url": s.video_path,
                "midi_url": s.midi_path,
                "skeleton_url": s.skeleton_path,
                "tsv_url": s.tsv_path,
                "composer": s.metadata["composer"],
                "piece": s.metadata["piece"],
                "performer": s.metadata["performer"],
                "keyboard_corners": json.dumps(s.metadata["keyboard_corners"]),
            }
        )

    manifest = pd.DataFrame(rows)
    return manifest, dataset


def download_manifest_videos(
    manifest: pd.DataFrame,
    dataset: PianoVAMDataset,
    verify_fps: bool = True,
) -> Dict[str, str]:
    """
    Download (with caching) every video referenced in the manifest.

    If *verify_fps* is True, reads each video's actual FPS from the
    file header and updates the manifest in-place.

    Returns:
        dict mapping video_id → local file path (str)
    """
    print("\nDownloading / caching videos ...")
    local_paths: Dict[str, str] = {}

    for idx, row in manifest.iterrows():
        vid_id = row["video_id"]
        url = row["video_url"]
        local = dataset.download_file(url)
        local_paths[vid_id] = str(local)

        if verify_fps:
            info = get_video_info(local)
            manifest.at[idx, "fps"] = info.fps
            print(
                f"  {vid_id}: cached -> {local.name}  "
                f"({info.width}x{info.height}, {info.fps:.1f} fps, "
                f"{info.duration:.1f}s)"
            )
        else:
            print(f"  {vid_id}: cached -> {local.name}")

    return local_paths


# ═══════════════════════════════════════════════════════════════════
# Manifest I/O
# ═══════════════════════════════════════════════════════════════════

def save_manifest(manifest: pd.DataFrame, path: str) -> Path:
    """Save manifest DataFrame to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_json(p, orient="records", indent=2)
    print(f"\nManifest saved to {p}")
    return p


def load_manifest(path: str) -> pd.DataFrame:
    """Load a previously saved manifest JSON."""
    return pd.read_json(path, orient="records")
