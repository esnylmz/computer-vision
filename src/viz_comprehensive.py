"""
src/viz_comprehensive.py — Comprehensive visualization for evaluation.

Creates publication-quality figures for thesis and presentation:
  - Side-by-side Group comparisons
  - Timeline plots with multiple systems
  - Attention visualizations
  - Error analysis plots
  - Performance breakdown by finger/difficulty

Usage:
    from src.viz_comprehensive import create_comparison_report
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Three-way comparison plots
# ═══════════════════════════════════════════════════════════════════

def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing precision/recall/F1/AUC across groups.
    
    Args:
        metrics_dict: {
            'Group A': {'precision': 0.92, 'recall': 0.88, ...},
            'Group B Auto': {...},
            'Group B Calibrated': {...},
        }
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    metric_names = ['precision', 'recall', 'f1', 'roc_auc']
    groups = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[g].get(metric, 0) for g in groups]
        axes[i].bar(range(len(groups)), values, color=['steelblue', 'darkorange', 'green'])
        axes[i].set_xticks(range(len(groups)))
        axes[i].set_xticklabels(groups, rotation=15, ha='right', fontsize=9)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    fig.suptitle('Performance Comparison: Group A vs Group B', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_timeline_comparison(
    df: pd.DataFrame,
    video_id: str,
    hand: str = 'right',
    finger: str = 'index',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Timeline showing teacher, CNN, and CNN+BiLSTM predictions.
    
    Args:
        df: DataFrame with columns: time_sec, press_smooth, press_prob, press_prob_refined
        video_id: Video identifier
        hand, finger: Which fingertip to plot
        save_path: Optional path to save
    
    Returns:
        Figure
    """
    sub = df[(df['hand'] == hand) & (df['finger_name'] == finger)].sort_values('frame_idx')
    
    if sub.empty:
        print(f"  No data for {hand}/{finger}")
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    # Top: Teacher labels (Group A)
    axes[0].fill_between(sub['time_sec'], 0, sub.get('press_smooth', 0),
                         alpha=0.3, color='gray', label='Teacher (Group A)')
    axes[0].plot(sub['time_sec'], sub.get('press_smooth', 0), 'k-', linewidth=1.5)
    axes[0].set_ylabel('Press Label')
    axes[0].set_title(f'{video_id} — {hand} {finger} (Teacher Labels)', fontsize=11)
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # Middle: CNN only (Group B)
    axes[1].fill_between(sub['time_sec'], 0, sub.get('press_prob', 0),
                         alpha=0.3, color='orange', label='CNN only')
    axes[1].plot(sub['time_sec'], sub.get('press_prob', 0), 'orange', linewidth=1.5)
    # Overlay teacher (faint)
    axes[1].plot(sub['time_sec'], sub.get('press_smooth', 0), 'k--', linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel('Press Probability')
    axes[1].set_title('CNN Predictions (Group B)', fontsize=11)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    # Bottom: CNN + BiLSTM (refined)
    axes[2].fill_between(sub['time_sec'], 0, sub.get('press_prob_refined', 0),
                         alpha=0.3, color='blue', label='CNN + BiLSTM')
    axes[2].plot(sub['time_sec'], sub.get('press_prob_refined', 0), 'blue', linewidth=1.5)
    axes[2].plot(sub['time_sec'], sub.get('press_smooth', 0), 'k--', linewidth=0.8, alpha=0.4)
    axes[2].set_ylabel('Press Probability')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Temporal Refinement (CNN + BiLSTM)', fontsize=11)
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_per_finger_performance(
    df: pd.DataFrame,
    y_true_col: str = 'press_smooth',
    y_pred_col: str = 'press_prob_refined',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing F1 score breakdown by finger.
    
    Useful for identifying which fingers are hardest to detect.
    """
    from sklearn.metrics import f1_score
    
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    f1_scores = []
    
    for fname in fingers:
        sub = df[df['finger_name'] == fname]
        if len(sub) < 10:
            f1_scores.append(0)
            continue
        
        y_true = (sub[y_true_col] > 0.5).astype(int)
        y_pred = (sub[y_pred_col] > 0.5).astype(int)
        
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            f1_scores.append(0)
        else:
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    ax.bar(fingers, f1_scores, color=colors)
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance by Finger')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_crop_examples(
    crops_press: List[np.ndarray],
    crops_no_press: List[np.ndarray],
    n_each: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grid showing example crops for press vs no-press.
    
    Helps visualize what the CNN is learning.
    """
    fig, axes = plt.subplots(2, n_each, figsize=(2 * n_each, 4))
    
    for i in range(n_each):
        # Top row: Press examples
        if i < len(crops_press):
            axes[0, i].imshow(crops_press[i])
            axes[0, i].set_title('Press', fontsize=9, color='green')
        axes[0, i].axis('off')
        
        # Bottom row: No-press examples
        if i < len(crops_no_press):
            axes[1, i].imshow(crops_no_press[i])
            axes[1, i].set_title('No-press', fontsize=9, color='red')
        axes[1, i].axis('off')
    
    fig.suptitle('Fingertip Crop Examples (what the CNN sees)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    losses: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('CNN Training Curve')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════════════════════════
# Comprehensive report
# ═══════════════════════════════════════════════════════════════════

def create_comparison_report(
    output_dir: str,
    metrics_dict: Dict[str, Dict],
    timeline_df: Optional[pd.DataFrame] = None,
    crops_press: Optional[List[np.ndarray]] = None,
    crops_no_press: Optional[List[np.ndarray]] = None,
    training_losses: Optional[List[float]] = None,
):
    """
    Generate a comprehensive evaluation report with all visualizations.
    
    Creates:
      - metrics_comparison.png
      - timeline_comparison.png (if df provided)
      - crop_examples.png (if crops provided)
      - training_curve.png (if losses provided)
      - per_finger_performance.png (if df provided)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison
    plot_metrics_comparison(metrics_dict, str(out / 'metrics_comparison.png'))
    print(f"  ✓ Saved metrics_comparison.png")
    
    # 2. Timeline
    if timeline_df is not None:
        plot_timeline_comparison(timeline_df, 'test_video',
                                 save_path=str(out / 'timeline_comparison.png'))
        print(f"  ✓ Saved timeline_comparison.png")
    
    # 3. Crop examples
    if crops_press and crops_no_press:
        plot_crop_examples(crops_press, crops_no_press,
                          save_path=str(out / 'crop_examples.png'))
        print(f"  ✓ Saved crop_examples.png")
    
    # 4. Training curve
    if training_losses:
        plot_training_curves(training_losses,
                            save_path=str(out / 'training_curve.png'))
        print(f"  ✓ Saved training_curve.png")
    
    # 5. Per-finger performance
    if timeline_df is not None and 'press_prob_refined' in timeline_df.columns:
        plot_per_finger_performance(timeline_df,
                                    save_path=str(out / 'per_finger_performance.png'))
        print(f"  ✓ Saved per_finger_performance.png")
    
    print(f"\n  Report saved to {out}/")
