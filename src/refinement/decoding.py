"""
Constraint-aware decoding for fingering sequences.

Selects a per-note finger sequence that maximizes model probability while
respecting biomechanical constraints as much as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .constraints import BiomechanicalConstraints


@dataclass
class DecodeResult:
    fingers: List[int]          # 1..5
    log_probability: float


def constrained_viterbi_decode(
    probs: np.ndarray,
    pitches: Sequence[int],
    hands: Sequence[str],
    constraints: Optional[BiomechanicalConstraints] = None,
    mask: Optional[Sequence[bool]] = None,
    invalid_transition_penalty: float = -1e9,
    eps: float = 1e-12,
) -> DecodeResult:
    """
    Decode the most likely finger sequence with transition constraints.

    Args:
        probs: Array of shape (T, 5) with per-note finger probabilities.
        pitches: Length-T MIDI pitches.
        hands: Length-T hand labels ('left'/'right').
        constraints: BiomechanicalConstraints instance (default: intermediate, non-strict).
        mask: Optional length-T boolean mask for valid timesteps.
        invalid_transition_penalty: Additive penalty for invalid transitions.
        eps: Numerical stabilizer for log.

    Returns:
        DecodeResult with decoded fingers (1..5) and total log-probability score.
    """
    if constraints is None:
        constraints = BiomechanicalConstraints()

    if probs.ndim != 2 or probs.shape[1] != 5:
        raise ValueError(f"Expected probs with shape (T, 5), got {probs.shape}")

    T = int(probs.shape[0])
    if T == 0:
        return DecodeResult(fingers=[], log_probability=0.0)

    if len(pitches) != T or len(hands) != T:
        raise ValueError("Length mismatch between probs, pitches, and hands")

    if mask is None:
        valid_T = T
    else:
        if len(mask) != T:
            raise ValueError("Mask length must match T")
        valid_T = int(np.sum(np.asarray(mask, dtype=bool)))

    if valid_T <= 0:
        return DecodeResult(fingers=[], log_probability=0.0)

    probs = probs[:valid_T]
    pitches = list(pitches[:valid_T])
    hands = list(hands[:valid_T])

    logp = np.log(np.clip(probs, eps, 1.0))  # (valid_T, 5)

    dp = np.full((valid_T, 5), -np.inf, dtype=np.float64)
    back = np.full((valid_T, 5), -1, dtype=np.int32)

    dp[0, :] = logp[0, :]

    for t in range(1, valid_T):
        p1 = int(pitches[t - 1])
        p2 = int(pitches[t])
        hand = str(hands[t - 1])

        for f2 in range(5):
            best_score = -np.inf
            best_f1 = -1
            finger2 = f2 + 1

            for f1 in range(5):
                finger1 = f1 + 1
                is_valid, _ = constraints.is_valid_transition(
                    finger1=finger1,
                    finger2=finger2,
                    pitch1=p1,
                    pitch2=p2,
                    hand=hand,
                )
                penalty = 0.0 if is_valid else float(invalid_transition_penalty)

                score = dp[t - 1, f1] + logp[t, f2] + penalty
                if score > best_score:
                    best_score = score
                    best_f1 = f1

            dp[t, f2] = best_score
            back[t, f2] = best_f1

    last_f = int(np.argmax(dp[valid_T - 1, :]))
    best_total = float(dp[valid_T - 1, last_f])

    path = [last_f]
    for t in range(valid_T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()

    fingers = [p + 1 for p in path]
    return DecodeResult(fingers=fingers, log_probability=best_total)

