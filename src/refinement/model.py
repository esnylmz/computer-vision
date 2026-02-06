"""
Neural Refinement Model

BiLSTM-based model for refining finger assignments using
temporal context and biomechanical constraints.

Based on:
- Zhao et al. (2022) - Sequence modeling for fingering
- Ramoneda et al. (2022) - ArGNN paper

Usage:
    from src.refinement.model import FingeringRefiner
    
    model = FingeringRefiner(hidden_size=128)
    refined = model(features, initial_assignments)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class FingeringRefiner(nn.Module):
    """
    BiLSTM-based model for refining finger assignments.
    
    Takes sequence of notes with initial finger assignments and
    outputs refined assignments considering temporal context.
    
    Input features per note:
    - MIDI pitch (normalized)
    - Initial finger assignment (one-hot)
    - Time since previous note
    - Hand (left/right)
    """
    
    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_fingers: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Size of input features per note
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_fingers: Number of finger classes (5)
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_fingers = num_fingers
        self.bidirectional = bidirectional
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_fingers)
        )
        
        # Attention mechanism for focusing on relevant context
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(
        self, 
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Shape (batch, seq_len, input_size)
            mask: Optional mask for padding (batch, seq_len)
            
        Returns:
            Finger logits of shape (batch, seq_len, num_fingers)
        """
        # Embed input
        x = self.input_embed(features)  # (batch, seq, hidden)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Self-attention
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attn_mask
        )
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out
        
        # Output layer
        logits = self.output(combined)  # (batch, seq, num_fingers)
        
        return logits
    
    def predict(
        self, 
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get predicted finger indices.
        
        Args:
            features: Shape (batch, seq_len, input_size)
            mask: Optional padding mask
            
        Returns:
            Predicted fingers of shape (batch, seq_len)
        """
        logits = self.forward(features, mask)
        return torch.argmax(logits, dim=-1) + 1  # 1-indexed fingers
    
    def predict_with_confidence(
        self, 
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.
        
        Returns:
            (predictions, confidences) both of shape (batch, seq_len)
        """
        logits = self.forward(features, mask)
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        return predictions + 1, confidences


class FeatureExtractor:
    """
    Extracts features for the refinement model.
    
    Converts raw note data into model input features.
    """
    
    def __init__(self, normalize_pitch: bool = True):
        """
        Args:
            normalize_pitch: Normalize MIDI pitch to [0, 1]
        """
        self.normalize_pitch = normalize_pitch
        self.midi_min = 21
        self.midi_max = 108
    
    def extract(
        self,
        pitches: List[int],
        initial_fingers: List[int],
        onsets: List[float],
        hands: List[str]
    ) -> torch.Tensor:
        """
        Extract features for a sequence of notes.
        
        Args:
            pitches: List of MIDI pitches
            initial_fingers: Initial finger assignments (1-5)
            onsets: Note onset times
            hands: Hand labels ('left' or 'right')
            
        Returns:
            Feature tensor of shape (seq_len, input_size)
        """
        seq_len = len(pitches)
        features = []
        
        for i in range(seq_len):
            note_features = []
            
            # Normalized pitch
            if self.normalize_pitch:
                norm_pitch = (pitches[i] - self.midi_min) / (self.midi_max - self.midi_min)
            else:
                norm_pitch = pitches[i] / 127.0
            note_features.append(norm_pitch)
            
            # One-hot finger encoding
            finger_onehot = [0.0] * 5
            if 1 <= initial_fingers[i] <= 5:
                finger_onehot[initial_fingers[i] - 1] = 1.0
            note_features.extend(finger_onehot)
            
            # Time delta
            if i > 0:
                time_delta = onsets[i] - onsets[i-1]
            else:
                time_delta = 0.0
            note_features.append(min(time_delta, 2.0) / 2.0)  # Normalize, cap at 2s
            
            # Hand encoding
            note_features.append(1.0 if hands[i] == 'right' else 0.0)
            
            # Pitch class (one-hot, 12 classes)
            pitch_class = pitches[i] % 12
            pc_onehot = [0.0] * 12
            pc_onehot[pitch_class] = 1.0
            note_features.extend(pc_onehot)
            
            features.append(note_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_input_size(self) -> int:
        """Get the expected input size."""
        # pitch(1) + finger(5) + delta(1) + hand(1) + pitch_class(12) = 20
        return 20


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for training the refinement model."""
    
    def __init__(
        self,
        sequences: List[dict],
        feature_extractor: FeatureExtractor,
        max_len: int = 256
    ):
        """
        Args:
            sequences: List of sequence dicts with keys:
                       'pitches', 'fingers', 'onsets', 'hands', 'labels'
            feature_extractor: FeatureExtractor instance
            max_len: Maximum sequence length
        """
        self.sequences = sequences
        self.feature_extractor = feature_extractor
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        features = self.feature_extractor.extract(
            seq['pitches'],
            seq['fingers'],
            seq['onsets'],
            seq['hands']
        )
        
        labels = torch.tensor(seq['labels'], dtype=torch.long) - 1  # 0-indexed

        pitches = torch.tensor(seq['pitches'], dtype=torch.long)
        hands = torch.tensor(
            [1 if h == 'right' else 0 for h in seq['hands']],
            dtype=torch.long
        )
        
        # Pad or truncate
        if len(features) > self.max_len:
            features = features[:self.max_len]
            labels = labels[:self.max_len]
            pitches = pitches[:self.max_len]
            hands = hands[:self.max_len]
        
        mask = torch.ones(len(features), dtype=torch.bool)
        
        return {
            'features': features,
            'labels': labels,
            'mask': mask,
            'pitches': pitches,
            'hands': hands,
        }

