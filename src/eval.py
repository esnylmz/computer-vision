"""
src/eval.py — Evaluation metrics, confusion matrix, and ROC curve.

Reports precision, recall, F1, ROC-AUC on **test videos only**.

Usage:
    from src.eval import evaluate_predictions, save_eval_plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "CNN",
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, ROC-AUC.

    Args:
        y_true:  Ground-truth binary labels (0/1).
        y_prob:  Predicted probabilities ∈ [0, 1].
        threshold:  Decision threshold for binary metrics.
        label:  Display label for printing.

    Returns:
        dict with keys: precision, recall, f1, roc_auc
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true_bin = (np.asarray(y_true) >= threshold).astype(int)

    n_pos = int(y_true_bin.sum())
    n_neg = int(len(y_true_bin) - n_pos)

    metrics: Dict[str, float] = {}

    if n_pos == 0 or n_neg == 0:
        print(
            f"  [{label}] WARNING: only one class present "
            f"(pos={n_pos}, neg={n_neg}) — some metrics undefined"
        )
        metrics["precision"] = precision_score(
            y_true_bin, y_pred, zero_division=0,
        )
        metrics["recall"] = recall_score(
            y_true_bin, y_pred, zero_division=0,
        )
        metrics["f1"] = f1_score(y_true_bin, y_pred, zero_division=0)
        metrics["roc_auc"] = float("nan")
    else:
        metrics["precision"] = precision_score(y_true_bin, y_pred)
        metrics["recall"] = recall_score(y_true_bin, y_pred)
        metrics["f1"] = f1_score(y_true_bin, y_pred)
        metrics["roc_auc"] = roc_auc_score(y_true_bin, y_prob)

    print(
        f"  [{label}]  prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"AUC={metrics.get('roc_auc', float('nan')):.3f}"
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════

def save_eval_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    label: str = "CNN",
    threshold: float = 0.5,
):
    """
    Save confusion matrix and ROC curve to *output_dir*.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_pred = (y_prob >= threshold).astype(int)
    y_true_bin = (np.asarray(y_true) >= threshold).astype(int)

    # ── Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no-press", "press"])
    ax.set_yticklabels(["no-press", "press"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"Confusion Matrix — {label}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix_{label}.png", dpi=120)
    plt.close(fig)

    # ── ROC Curve ─────────────────────────────────────────────────
    n_pos = int(y_true_bin.sum())
    n_neg = int(len(y_true_bin) - n_pos)
    if n_pos > 0 and n_neg > 0:
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        auc_val = roc_auc_score(y_true_bin, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"roc_curve_{label}.png", dpi=120)
        plt.close(fig)
    else:
        print(f"  [{label}] Skipping ROC curve (single class)")

    print(f"  [{label}] Evaluation plots saved → {out}")


# ═══════════════════════════════════════════════════════════════════
# Event-consistency metric
# ═══════════════════════════════════════════════════════════════════

def event_consistency(
    preds_binary: np.ndarray,
) -> Dict[str, float]:
    """
    Count isolated single-frame presses and other temporal artifacts.

    An *isolated press* = press at t flanked by no-press at t−1 and t+1.
    Fewer is better (more temporally consistent).

    Returns dict: {n_isolated, n_press, isolation_rate}
    """
    p = np.asarray(preds_binary, dtype=int)
    n_press = int(p.sum())
    if n_press == 0 or len(p) < 3:
        return {"n_isolated": 0, "n_press": n_press, "isolation_rate": 0.0}

    padded = np.concatenate([[0], p, [0]])
    isolated = 0
    for i in range(1, len(padded) - 1):
        if padded[i] == 1 and padded[i - 1] == 0 and padded[i + 1] == 0:
            isolated += 1

    return {
        "n_isolated": isolated,
        "n_press": n_press,
        "isolation_rate": isolated / max(n_press, 1),
    }
