"""
src/crops.py — Fingertip-centred crop extraction from video frames.

For each detected fingertip, a ``crop_size × crop_size`` pixel patch is
extracted from the original video frame centred on the fingertip position.
These crops are the CNN's input — the model learns to classify
press / no-press purely from **pixels**, with no coordinate features.

Usage:
    from src.crops import extract_crops_for_video, PressCropDataset
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Single-crop helper
# ═══════════════════════════════════════════════════════════════════

def _extract_crop(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = 64,
) -> np.ndarray:
    """
    Extract a *crop_size × crop_size* patch centred on (*cx*, *cy*).

    Out-of-bounds regions are zero-padded (black).
    Returns an RGB uint8 array of shape (crop_size, crop_size, 3).
    """
    h, w = frame_bgr.shape[:2]
    half = crop_size // 2
    x1 = int(cx) - half
    y1 = int(cy) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Compute valid source region
    sx1 = max(x1, 0)
    sy1 = max(y1, 0)
    sx2 = min(x2, w)
    sy2 = min(y2, h)

    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    if sx1 >= sx2 or sy1 >= sy2:
        return crop  # entirely outside frame

    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)
    crop[dy1:dy2, dx1:dx2] = frame_bgr[sy1:sy2, sx1:sx2]

    # BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


# ═══════════════════════════════════════════════════════════════════
# Batch extraction (one video)
# ═══════════════════════════════════════════════════════════════════

def extract_crops_for_video(
    video_path: str,
    landmarks_df: pd.DataFrame,
    crop_size: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract fingertip crops for every detection in *landmarks_df*.

    Groups by ``frame_idx`` so each video frame is read only once.

    Returns:
        crops  — list of (crop_size, crop_size, 3) uint8 arrays
        idxs   — matching landmark DataFrame integer positions
    """
    if landmarks_df.empty:
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    crops: List[np.ndarray] = []
    idxs: List[int] = []

    grouped = landmarks_df.groupby("frame_idx")
    for fidx in tqdm(
        sorted(grouped.groups.keys()), desc="Crops", leave=False,
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue
        for loc, row in grouped.get_group(fidx).iterrows():
            crop = _extract_crop(
                frame, row["x_pixel"], row["y_pixel"], crop_size,
            )
            crops.append(crop)
            idxs.append(loc)

    cap.release()
    return crops, idxs


# ═══════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════

class PressCropDataset(Dataset):
    """
    PyTorch Dataset of fingertip crops + binary press labels.

    ``__getitem__`` returns ``(image_tensor, label_tensor)``
    where *image_tensor* is (3, H, W) float32 in [0, 1].
    """

    def __init__(
        self,
        crops: List[np.ndarray],
        labels: List[float],
        transform=None,
    ):
        assert len(crops) == len(labels), "crops/labels length mismatch"
        self.crops = crops
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int):
        img = (
            torch.from_numpy(self.crops[idx]).permute(2, 0, 1).float() / 255.0
        )
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label
