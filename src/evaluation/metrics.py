"""
Evaluation Metrics for Piano Fingering Detection

Implements standard fingering evaluation metrics from PIG dataset:
- M_gen: General match rate
- M_high: Highest match rate
- IFR: Irrational Fingering Rate

Usage:
    from src.evaluation.metrics import FingeringMetrics
    
    metrics = FingeringMetrics()
    results = metrics.evaluate(predictions, ground_truth)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Evaluation results container."""
    accuracy: float
    m_gen: float
    m_high: float
    ifr: float
    num_notes: int
    per_finger_accuracy: Dict[int, float]
    confusion_matrix: Optional[np.ndarray] = None


class FingeringMetrics:
    """
    Evaluation metrics for piano fingering predictions.
    
    Follows evaluation protocol from:
    - Nakamura et al. (2020) - PIG dataset paper
    - Ramoneda et al. (2022) - ArGNN paper
    """
    
    # Maximum physically possible finger stretch (in semitones)
    MAX_STRETCH = {
        (1, 2): 10,
        (2, 3): 5,
        (3, 4): 4,
        (4, 5): 5,
        (1, 5): 12,
    }
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        predictions: List[int],
        ground_truth: Union[List[int], List[List[int]]],
        pitches: Optional[List[int]] = None
    ) -> EvaluationResult:
        """
        Evaluate fingering predictions against ground truth.
        
        Args:
            predictions: List of predicted fingers (1-5) per note
            ground_truth: List of ground truth fingers, or list of lists
                         if multiple annotators
            pitches: Optional MIDI pitches for IFR calculation
                         
        Returns:
            EvaluationResult with all metrics
        """
        n = len(predictions)
        
        # Normalize ground truth to list of lists
        if ground_truth and isinstance(ground_truth[0], int):
            ground_truth = [[gt] for gt in ground_truth]
        
        assert len(predictions) == len(ground_truth), \
            f"Length mismatch: {len(predictions)} vs {len(ground_truth)}"
        
        # Basic accuracy (exact match with first annotator)
        accuracy = sum(
            p == gt[0] for p, gt in zip(predictions, ground_truth)
        ) / n if n > 0 else 0
        
        # M_gen: Average match rate across all annotators
        m_gen = compute_m_gen(predictions, ground_truth)
        
        # M_high: Match rate with closest ground truth
        m_high = sum(
            p in gt for p, gt in zip(predictions, ground_truth)
        ) / n if n > 0 else 0
        
        # IFR: Irrational Fingering Rate
        ifr = compute_ifr(predictions, pitches) if pitches else 0
        
        # Per-finger accuracy
        per_finger = self._per_finger_accuracy(predictions, ground_truth)
        
        # Confusion matrix
        confusion = self._confusion_matrix(predictions, ground_truth)
        
        return EvaluationResult(
            accuracy=accuracy,
            m_gen=m_gen,
            m_high=m_high,
            ifr=ifr,
            num_notes=n,
            per_finger_accuracy=per_finger,
            confusion_matrix=confusion
        )
    
    def _per_finger_accuracy(
        self,
        predictions: List[int],
        ground_truth: List[List[int]]
    ) -> Dict[int, float]:
        """Compute accuracy for each finger separately."""
        finger_correct = Counter()
        finger_total = Counter()
        
        for p, gt_list in zip(predictions, ground_truth):
            gt = gt_list[0]  # Use first annotator
            finger_total[gt] += 1
            if p == gt:
                finger_correct[gt] += 1
        
        return {
            f: finger_correct[f] / finger_total[f] if finger_total[f] > 0 else 0.0
            for f in range(1, 6)
        }
    
    def _confusion_matrix(
        self,
        predictions: List[int],
        ground_truth: List[List[int]]
    ) -> np.ndarray:
        """Compute 5x5 confusion matrix."""
        matrix = np.zeros((5, 5), dtype=np.int32)
        
        for p, gt_list in zip(predictions, ground_truth):
            gt = gt_list[0]
            if 1 <= p <= 5 and 1 <= gt <= 5:
                matrix[gt - 1, p - 1] += 1
        
        return matrix
    
    def evaluate_by_hand(
        self,
        predictions: List[int],
        ground_truth: List[List[int]],
        hands: List[str]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate separately for left and right hand."""
        results = {}
        
        for hand in ['left', 'right']:
            mask = [h == hand for h in hands]
            if not any(mask):
                continue
            
            hand_preds = [p for p, m in zip(predictions, mask) if m]
            hand_gt = [gt for gt, m in zip(ground_truth, mask) if m]
            
            results[hand] = self.evaluate(hand_preds, hand_gt)
        
        return results
    
    def evaluate_by_skill(
        self,
        predictions: List[int],
        ground_truth: List[List[int]],
        skill_levels: List[str]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate separately by player skill level."""
        results = {}
        
        for skill in set(skill_levels):
            mask = [s == skill for s in skill_levels]
            if not any(mask):
                continue
            
            skill_preds = [p for p, m in zip(predictions, mask) if m]
            skill_gt = [gt for gt, m in zip(ground_truth, mask) if m]
            
            results[skill] = self.evaluate(skill_preds, skill_gt)
        
        return results


def compute_accuracy(
    predictions: List[int],
    ground_truth: List[int]
) -> float:
    """Compute simple accuracy."""
    if not predictions:
        return 0.0
    return sum(p == gt for p, gt in zip(predictions, ground_truth)) / len(predictions)


def compute_m_gen(
    predictions: List[int],
    ground_truth: List[List[int]]
) -> float:
    """
    Compute M_gen: General match rate.
    
    For each note, compute the fraction of annotators that match
    the prediction, then average across all notes.
    """
    if not predictions:
        return 0.0
    
    total = 0.0
    for p, gt_list in zip(predictions, ground_truth):
        matches = sum(p == gt for gt in gt_list)
        total += matches / len(gt_list)
    
    return total / len(predictions)


def compute_ifr(
    predictions: List[int],
    pitches: Optional[List[int]] = None
) -> float:
    """
    Compute Irrational Fingering Rate.
    
    Checks for:
    - Same finger on consecutive different notes
    - Physically impossible stretches
    
    Args:
        predictions: List of finger assignments (1-5)
        pitches: Optional list of MIDI pitches
        
    Returns:
        Fraction of irrational transitions
    """
    if len(predictions) < 2:
        return 0.0
    
    # Maximum stretch between finger pairs
    max_stretch = {
        (1, 2): 10, (1, 3): 11, (1, 4): 12, (1, 5): 12,
        (2, 3): 5, (2, 4): 7, (2, 5): 9,
        (3, 4): 4, (3, 5): 6,
        (4, 5): 5
    }
    
    irrational = 0
    total = len(predictions) - 1
    
    for i in range(total):
        f1, f2 = predictions[i], predictions[i + 1]
        
        # Same finger on consecutive notes
        if f1 == f2:
            if pitches is not None and pitches[i] != pitches[i + 1]:
                irrational += 1
            continue
        
        # Check stretch if pitches available
        if pitches is not None:
            interval = abs(pitches[i + 1] - pitches[i])
            key = tuple(sorted([f1, f2]))
            limit = max_stretch.get(key, 12)
            
            if interval > limit:
                irrational += 1
    
    return irrational / total


def compute_precision_recall(
    predictions: List[int],
    ground_truth: List[int],
    finger: int
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for a specific finger.
    
    Returns:
        (precision, recall, f1) tuple
    """
    tp = sum(p == finger and gt == finger for p, gt in zip(predictions, ground_truth))
    fp = sum(p == finger and gt != finger for p, gt in zip(predictions, ground_truth))
    fn = sum(p != finger and gt == finger for p, gt in zip(predictions, ground_truth))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def aggregate_results(results: List[EvaluationResult]) -> EvaluationResult:
    """Aggregate multiple evaluation results."""
    if not results:
        return EvaluationResult(0, 0, 0, 0, 0, {}, None)
    
    total_notes = sum(r.num_notes for r in results)
    
    # Weighted averages
    accuracy = sum(r.accuracy * r.num_notes for r in results) / total_notes
    m_gen = sum(r.m_gen * r.num_notes for r in results) / total_notes
    m_high = sum(r.m_high * r.num_notes for r in results) / total_notes
    ifr = sum(r.ifr * r.num_notes for r in results) / total_notes
    
    # Aggregate per-finger accuracy
    per_finger = {}
    for f in range(1, 6):
        values = [r.per_finger_accuracy.get(f, 0) for r in results]
        per_finger[f] = np.mean(values) if values else 0
    
    # Sum confusion matrices
    confusion = sum(r.confusion_matrix for r in results if r.confusion_matrix is not None)
    
    return EvaluationResult(
        accuracy=accuracy,
        m_gen=m_gen,
        m_high=m_high,
        ifr=ifr,
        num_notes=total_notes,
        per_finger_accuracy=per_finger,
        confusion_matrix=confusion
    )

