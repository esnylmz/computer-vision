"""
Training Loop for Refinement Model

Handles model training with constraint-aware loss.

Usage:
    from src.refinement.train import train_refiner
    
    model = train_refiner(train_dataset, val_dataset, config)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm

from .model import FingeringRefiner, SequenceDataset
from .constraints import BiomechanicalConstraints

logger = logging.getLogger(__name__)


class ConstraintAwareLoss(nn.Module):
    """
    Loss function combining cross-entropy with constraint penalties.
    """
    
    def __init__(
        self,
        constraints: BiomechanicalConstraints,
        constraint_weight: float = 0.1
    ):
        super().__init__()
        self.constraints = constraints
        self.constraint_weight = constraint_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pitches: torch.Tensor,
        hands: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Shape (batch, seq, num_fingers)
            labels: Shape (batch, seq) - 0-indexed finger labels
            pitches: Shape (batch, seq) - MIDI pitches
            hands: Shape (batch, seq) - hand labels (0=left, 1=right)
            mask: Shape (batch, seq) - valid positions
            
        Returns:
            Loss scalar
        """
        batch_size, seq_len, num_fingers = logits.shape
        
        # Cross-entropy loss
        logits_flat = logits.view(-1, num_fingers)
        labels_flat = labels.view(-1)
        ce_loss = self.ce_loss(logits_flat, labels_flat)
        ce_loss = ce_loss.view(batch_size, seq_len)
        
        # Apply mask
        ce_loss = (ce_loss * mask.float()).sum() / mask.float().sum()
        
        # Constraint loss (on predictions)
        if self.constraint_weight > 0:
            predictions = torch.argmax(logits, dim=-1) + 1  # 1-indexed
            constraint_loss = self._compute_constraint_loss(
                predictions, pitches, hands, mask
            )
            total_loss = ce_loss + self.constraint_weight * constraint_loss
        else:
            total_loss = ce_loss
        
        return total_loss
    
    def _compute_constraint_loss(
        self,
        predictions: torch.Tensor,
        pitches: torch.Tensor,
        hands: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute constraint violation loss."""
        batch_size, seq_len = predictions.shape
        total_violations = 0.0
        total_transitions = 0
        
        for b in range(batch_size):
            for i in range(seq_len - 1):
                if mask[b, i] and mask[b, i+1]:
                    f1 = predictions[b, i].item()
                    f2 = predictions[b, i+1].item()
                    p1 = pitches[b, i].item()
                    p2 = pitches[b, i+1].item()
                    hand = 'right' if hands[b, i].item() else 'left'
                    
                    is_valid, _ = self.constraints.is_valid_transition(
                        f1, f2, p1, p2, hand
                    )
                    if not is_valid:
                        total_violations += 1
                    total_transitions += 1
        
        if total_transitions == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return torch.tensor(
            total_violations / total_transitions,
            device=predictions.device
        )


def train_refiner(
    train_dataset: SequenceDataset,
    val_dataset: Optional[SequenceDataset] = None,
    config: Optional[Dict] = None
) -> FingeringRefiner:
    """
    Train the refinement model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        
    Returns:
        Trained model
    """
    config = config or {}
    
    # Config defaults
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.3)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 50)
    patience = config.get('early_stopping_patience', 10)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    
    # Create model
    input_size = train_dataset.feature_extractor.get_input_size()
    model = FingeringRefiner(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    # Loss and optimizer
    constraints = BiomechanicalConstraints()
    criterion = ConstraintAwareLoss(constraints, constraint_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            pitches = batch.get('pitches', torch.zeros_like(labels)).to(device)
            hands = batch.get('hands', torch.zeros_like(labels)).to(device)
            
            optimizer.zero_grad()
            
            logits = model(features, mask)
            loss = criterion(logits, labels, pitches, hands, mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            train_correct += ((predictions == labels) * mask).sum().item()
            train_total += mask.sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
        
        # Validation
        if val_dataset:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            logger.info(f'Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info('Early stopping triggered')
                    break
    
    # Load best model
    if val_dataset and (checkpoint_path / 'best_model.pt').exists():
        model.load_state_dict(torch.load(checkpoint_path / 'best_model.pt'))
    
    return model


def evaluate(
    model: FingeringRefiner,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            pitches = batch.get('pitches', torch.zeros_like(labels)).to(device)
            hands = batch.get('hands', torch.zeros_like(labels)).to(device)
            
            logits = model(features, mask)
            loss = criterion(logits, labels, pitches, hands, mask)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            correct += ((predictions == labels) * mask).sum().item()
            total += mask.sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def collate_fn(batch):
    """Collate function for variable-length sequences."""
    max_len = max(len(item['features']) for item in batch)
    
    features_list = []
    labels_list = []
    mask_list = []
    
    for item in batch:
        seq_len = len(item['features'])
        pad_len = max_len - seq_len
        
        # Pad features
        if pad_len > 0:
            padding = torch.zeros(pad_len, item['features'].shape[1])
            features = torch.cat([item['features'], padding], dim=0)
        else:
            features = item['features']
        
        # Pad labels
        labels = torch.cat([
            item['labels'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        # Create mask
        mask = torch.cat([
            item['mask'],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        
        features_list.append(features)
        labels_list.append(labels)
        mask_list.append(mask)
    
    return {
        'features': torch.stack(features_list),
        'labels': torch.stack(labels_list),
        'mask': torch.stack(mask_list)
    }

