"""
Visualization Utilities

Tools for visualizing fingering predictions, hand landmarks,
and evaluation results.

Usage:
    from src.evaluation.visualization import ResultVisualizer
    
    visualizer = ResultVisualizer()
    visualizer.plot_confusion_matrix(results)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ResultVisualizer:
    """Visualization tools for fingering results."""
    
    FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    FINGER_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        title: str = "Finger Confusion Matrix",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: 5x5 confusion matrix
            title: Plot title
            save_name: Filename to save (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize by row (ground truth)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = confusion_matrix / row_sums
        
        sns.heatmap(
            normalized,
            annot=confusion_matrix,
            fmt='d',
            cmap='Blues',
            xticklabels=self.FINGER_NAMES,
            yticklabels=self.FINGER_NAMES,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Finger')
        ax.set_ylabel('True Finger')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_per_finger_accuracy(
        self,
        per_finger_accuracy: Dict[int, float],
        title: str = "Per-Finger Accuracy",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot bar chart of per-finger accuracy."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        fingers = list(range(1, 6))
        accuracies = [per_finger_accuracy.get(f, 0) for f in fingers]
        
        bars = ax.bar(fingers, accuracies, color=self.FINGER_COLORS)
        
        ax.set_xlabel('Finger')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticks(fingers)
        ax.set_xticklabels(self.FINGER_NAMES)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=10
            )
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Metrics Comparison",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare metrics across different methods/settings.
        
        Args:
            metrics_dict: Dict of {method_name: {metric: value}}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(metrics_dict.keys())
        metric_names = ['Accuracy', 'M_gen', 'M_high', 'IFR']
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            values = [
                metrics_dict[method].get('accuracy', 0),
                metrics_dict[method].get('m_gen', 0),
                metrics_dict[method].get('m_high', 0),
                metrics_dict[method].get('ifr', 0)
            ]
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=method)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_fingertip_trajectory(
        self,
        fingertips: np.ndarray,
        finger: int = 2,
        title: str = "Fingertip Trajectory",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot trajectory of a fingertip over time.
        
        Args:
            fingertips: Shape (T, 5, 2) - T frames, 5 fingers, (x, y)
            finger: Finger to plot (1-5)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        finger_idx = finger - 1
        trajectory = fingertips[:, finger_idx, :]
        
        # X coordinate over time
        axes[0].plot(trajectory[:, 0], color=self.FINGER_COLORS[finger_idx])
        axes[0].set_ylabel('X Position')
        axes[0].set_title(f'{self.FINGER_NAMES[finger_idx]} Finger Trajectory')
        
        # Y coordinate over time
        axes[1].plot(trajectory[:, 1], color=self.FINGER_COLORS[finger_idx])
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Y Position')
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig
    
    def draw_hand_on_frame(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        connections: bool = True,
        finger_labels: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Draw hand landmarks on video frame.
        
        Args:
            frame: BGR image
            landmarks: Shape (21, 3) hand landmarks
            connections: Draw finger bone connections
            finger_labels: Optional dict of {finger: label} to display
            
        Returns:
            Annotated frame
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for frame visualization")
        
        result = frame.copy()
        
        # MediaPipe hand connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        if connections:
            for start, end in HAND_CONNECTIONS:
                pt1 = (int(landmarks[start, 0]), int(landmarks[start, 1]))
                pt2 = (int(landmarks[end, 0]), int(landmarks[end, 1]))
                cv2.line(result, pt1, pt2, (200, 200, 200), 2)
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks):
            pt = (int(x), int(y))
            
            # Fingertips in color
            if i in [4, 8, 12, 16, 20]:
                finger_idx = [4, 8, 12, 16, 20].index(i)
                color = tuple(int(c) for c in 
                             bytes.fromhex(self.FINGER_COLORS[finger_idx][1:]))
                color = (color[2], color[1], color[0])  # RGB to BGR
                cv2.circle(result, pt, 8, color, -1)
            else:
                cv2.circle(result, pt, 4, (100, 100, 100), -1)
        
        # Draw finger labels
        if finger_labels:
            for finger, label in finger_labels.items():
                tip_idx = [4, 8, 12, 16, 20][finger - 1]
                pt = (int(landmarks[tip_idx, 0]) + 10, 
                      int(landmarks[tip_idx, 1]) - 10)
                cv2.putText(result, str(label), pt, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result
    
    def draw_keyboard_overlay(
        self,
        frame: np.ndarray,
        key_boundaries: Dict[int, Tuple[int, int, int, int]],
        pressed_keys: Optional[List[int]] = None,
        finger_assignments: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Draw keyboard overlay with pressed keys highlighted.
        
        Args:
            frame: BGR image
            key_boundaries: Dict of key_idx -> (x1, y1, x2, y2)
            pressed_keys: List of currently pressed key indices
            finger_assignments: Dict of key_idx -> finger
            
        Returns:
            Annotated frame
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for frame visualization")
        
        result = frame.copy()
        overlay = frame.copy()
        
        pressed_keys = pressed_keys or []
        finger_assignments = finger_assignments or {}
        
        for key_idx, (x1, y1, x2, y2) in key_boundaries.items():
            midi_pitch = key_idx + 21
            is_black = (midi_pitch % 12) in [1, 3, 6, 8, 10]
            
            if key_idx in pressed_keys:
                # Highlight pressed key
                if key_idx in finger_assignments:
                    finger = finger_assignments[key_idx]
                    color = tuple(int(c) for c in 
                                 bytes.fromhex(self.FINGER_COLORS[finger-1][1:]))
                    color = (color[2], color[1], color[0])
                else:
                    color = (0, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            else:
                # Draw key outline
                color = (50, 50, 50) if is_black else (200, 200, 200)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        return result
    
    def create_summary_figure(
        self,
        metrics: Dict[str, float],
        per_finger: Dict[int, float],
        confusion: np.ndarray,
        title: str = "Evaluation Summary",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Create a summary figure with multiple subplots."""
        fig = plt.figure(figsize=(14, 8))
        
        # Metrics text
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis('off')
        metrics_text = '\n'.join([
            f"Accuracy: {metrics.get('accuracy', 0):.3f}",
            f"M_gen: {metrics.get('m_gen', 0):.3f}",
            f"M_high: {metrics.get('m_high', 0):.3f}",
            f"IFR: {metrics.get('ifr', 0):.3f}",
            f"Total Notes: {metrics.get('num_notes', 0)}"
        ])
        ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='center',
                family='monospace')
        ax1.set_title('Overall Metrics', fontsize=12, fontweight='bold')
        
        # Per-finger accuracy
        ax2 = fig.add_subplot(2, 2, 2)
        fingers = list(range(1, 6))
        accuracies = [per_finger.get(f, 0) for f in fingers]
        ax2.bar(fingers, accuracies, color=self.FINGER_COLORS)
        ax2.set_xlabel('Finger')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Per-Finger Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xticks(fingers)
        ax2.set_xticklabels(self.FINGER_NAMES, rotation=45)
        ax2.set_ylim(0, 1)
        
        # Confusion matrix
        ax3 = fig.add_subplot(2, 2, (3, 4))
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = confusion / row_sums
        sns.heatmap(normalized, annot=confusion, fmt='d', cmap='Blues',
                   xticklabels=self.FINGER_NAMES, yticklabels=self.FINGER_NAMES,
                   ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        return fig

