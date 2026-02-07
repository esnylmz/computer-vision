"""
src/bilstm.py — Temporal refinement model (BiLSTM / GRU).

Refines frame-level CNN press probabilities by modelling temporal
context across frames.  Each input sequence is a single fingertip
track: [press_prob, dx, dy, speed] per timestep.

Usage:
    from src.bilstm import TemporalRefiner, build_sequences, train_refiner
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class TemporalRefiner(nn.Module):
    """
    BiLSTM that refines per-frame press probabilities.

    Input features (per timestep):
        [press_prob, dx, dy, speed]   →  4 dims
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 4)
        Returns:
            logits: (B, T, 1)
        """
        out, _ = self.lstm(x)          # (B, T, 2*H)
        logits = self.head(out)         # (B, T, 1)
        return logits


# ═══════════════════════════════════════════════════════════════════
# Sequence building
# ═══════════════════════════════════════════════════════════════════

def build_sequences(
    landmarks_df: pd.DataFrame,
    press_prob_col: str = "press_prob",
    max_seq_len: int = 128,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build per-fingertip temporal sequences from *landmarks_df*.

    Each sequence contains columns:
        [press_prob, dx, dy, speed]
    with labels from ``press_smooth`` (teacher).

    Sequences longer than *max_seq_len* are chunked.

    Returns:
        features_list  —  list of (T, 4) float32 arrays
        labels_list    —  list of (T,) float32 arrays
    """
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    # Ensure required columns exist
    for col in [press_prob_col, "x_kbd", "y_kbd"]:
        if col not in landmarks_df.columns:
            return features_list, labels_list

    label_col = (
        "press_smooth"
        if "press_smooth" in landmarks_df.columns
        else "press_raw"
    )

    for (hand, fid), grp in landmarks_df.groupby(["hand", "finger_id"]):
        grp = grp.sort_values("frame_idx").reset_index(drop=True)
        if len(grp) < 4:
            continue

        pp = grp[press_prob_col].values.astype(np.float32)
        xk = grp["x_kbd"].values.astype(np.float32)
        yk = grp["y_kbd"].values.astype(np.float32)

        dx = np.concatenate([[0], np.diff(xk)])
        dy = np.concatenate([[0], np.diff(yk)])
        speed = np.sqrt(dx ** 2 + dy ** 2)

        feats = np.stack([pp, dx, dy, speed], axis=1)  # (T, 4)
        labs = grp[label_col].values.astype(np.float32)

        # Chunk long sequences
        for start in range(0, len(feats), max_seq_len):
            end = min(start + max_seq_len, len(feats))
            if end - start < 4:
                continue
            features_list.append(feats[start:end])
            labels_list.append(labs[start:end])

    return features_list, labels_list


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class SeqDataset(Dataset):
    """Padded-sequence dataset for the temporal refiner."""

    def __init__(
        self,
        features: List[np.ndarray],
        labels: List[np.ndarray],
        max_len: int = 128,
    ):
        self.features = features
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        f = self.features[idx]
        l = self.labels[idx]
        T = len(f)
        pad = self.max_len - T

        if pad > 0:
            f = np.pad(f, ((0, pad), (0, 0)))
            l = np.pad(l, (0, pad), constant_values=-1)  # -1 = ignore
        else:
            f = f[: self.max_len]
            l = l[: self.max_len]

        mask = np.ones(self.max_len, dtype=np.float32)
        mask[T:] = 0.0

        return (
            torch.from_numpy(f).float(),
            torch.from_numpy(l).float(),
            torch.from_numpy(mask).float(),
        )


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_refiner(
    train_seqs: Tuple[List[np.ndarray], List[np.ndarray]],
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    max_seq_len: int = 128,
    pos_weight: Optional[float] = None,
) -> Tuple[TemporalRefiner, List[float]]:
    """Train the BiLSTM temporal refiner."""
    feats, labs = train_seqs
    if not feats:
        print("    No sequences to train on — returning untrained model")
        return TemporalRefiner().to(device), []

    ds = SeqDataset(feats, labs, max_len=max_seq_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    model = TemporalRefiner().to(device)
    pw = (
        torch.tensor([pos_weight], device=device)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n = 0
        for f_batch, l_batch, m_batch in tqdm(
            loader, desc=f"BiLSTM {epoch + 1}/{epochs}", leave=False,
        ):
            f_batch = f_batch.to(device)
            l_batch = l_batch.to(device)
            m_batch = m_batch.to(device)

            logits = model(f_batch).squeeze(-1)          # (B, T)
            raw_loss = criterion(logits, l_batch)         # (B, T)

            # mask out padding and ignore tokens
            valid = m_batch * (l_batch >= 0).float()
            loss = (raw_loss * valid).sum() / valid.sum().clamp(min=1)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running += loss.item()
            n += 1

        avg = running / max(n, 1)
        losses.append(avg)
        print(f"    BiLSTM epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return model, losses


# ═══════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_refiner(
    model: TemporalRefiner,
    landmarks_df: pd.DataFrame,
    press_prob_col: str = "press_prob",
    max_seq_len: int = 128,
    device: str = "cpu",
) -> pd.Series:
    """
    Run the trained BiLSTM on *landmarks_df* and return refined
    press probabilities aligned with the DataFrame index.
    """
    model.eval()
    result = pd.Series(
        np.full(len(landmarks_df), np.nan),
        index=landmarks_df.index,
        name="press_prob_refined",
    )

    for col in [press_prob_col, "x_kbd", "y_kbd"]:
        if col not in landmarks_df.columns:
            return result

    for (hand, fid), grp in landmarks_df.groupby(["hand", "finger_id"]):
        grp = grp.sort_values("frame_idx")
        if len(grp) < 4:
            result.loc[grp.index] = grp[press_prob_col].values
            continue

        pp = grp[press_prob_col].values.astype(np.float32)
        xk = grp["x_kbd"].values.astype(np.float32)
        yk = grp["y_kbd"].values.astype(np.float32)
        dx = np.concatenate([[0], np.diff(xk)])
        dy = np.concatenate([[0], np.diff(yk)])
        speed = np.sqrt(dx ** 2 + dy ** 2)
        feats = np.stack([pp, dx, dy, speed], axis=1)

        T = len(feats)
        pad = max_seq_len - T
        if pad > 0:
            feats_padded = np.pad(feats, ((0, pad), (0, 0)))
        else:
            feats_padded = feats[:max_seq_len]

        x = torch.from_numpy(feats_padded).float().unsqueeze(0).to(device)
        logits = model(x).squeeze(0).squeeze(-1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        result.loc[grp.index] = probs[:T]

    return result
