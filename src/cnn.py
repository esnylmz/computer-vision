"""
src/cnn.py — CNN press / no-press classifier.

A lightweight CNN that classifies fingertip-centred image crops as
*press* or *no-press*.  The model receives **only pixels** — no landmark
coordinates are used as features.

Usage:
    from src.cnn import PressNet, train_cnn, predict_cnn
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class PressNet(nn.Module):
    """
    Small CNN for binary press / no-press classification.

    Architecture:
        3×64×64 → Conv→ReLU→Pool (×3) → AdaptivePool → FC → sigmoid
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (shape [B, 1])."""
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════════
# Augmentation (torchvision-free, lightweight)
# ═══════════════════════════════════════════════════════════════════

class SimpleAugment(nn.Module):
    """Random brightness / horizontal-flip / noise (applied on GPU tensors)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # random brightness
            if torch.rand(1).item() < 0.5:
                x = x * (0.7 + 0.6 * torch.rand(1, device=x.device))
                x = x.clamp(0, 1)
            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            # random noise
            if torch.rand(1).item() < 0.3:
                x = x + 0.03 * torch.randn_like(x)
                x = x.clamp(0, 1)
        return x


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_cnn(
    train_dataset,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    pos_weight: Optional[float] = None,
) -> Tuple[PressNet, List[float]]:
    """
    Train *PressNet* on *train_dataset*.

    Args:
        train_dataset:  PressCropDataset.
        epochs:  Number of training epochs.
        batch_size:  Mini-batch size.
        lr:  Learning rate (Adam).
        device:  'cpu' or 'cuda'.
        pos_weight:  Positive class weight for BCE loss (handles imbalance).

    Returns:
        (model, epoch_losses)
    """
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    model = PressNet().to(device)
    augment = SimpleAugment().to(device)
    augment.train()

    pw = (
        torch.tensor([pos_weight], device=device)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: List[float] = []
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for imgs, labels in tqdm(
            loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False,
        ):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            imgs = augment(imgs)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        epoch_losses.append(avg)
        print(f"    Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return model, epoch_losses


# ═══════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_cnn(
    model: PressNet,
    dataset,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run *model* over *dataset* and return press probabilities (N,).
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.eval()
    probs: List[np.ndarray] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(probs) if probs else np.array([])
