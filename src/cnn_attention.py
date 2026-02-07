"""
src/cnn_attention.py — CNN with attention visualization (Grad-CAM).

Adds interpretability to the press classifier by visualizing which
pixels the CNN focuses on when making predictions.

Usage:
    from src.cnn_attention import compute_gradcam, visualize_attention
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# Grad-CAM implementation
# ═══════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability.
    
    Shows which regions of the input image contribute most to the prediction.
    """
    
    def __init__(self, model, target_layer_name: str = "features"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            x: Input tensor (1, C, H, W)
            target_class: Target class for gradients (None = predicted class)
        
        Returns:
            Heatmap array (H, W) in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if target_class is None:
            target_class = torch.argmax(output).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)  # Only positive influence
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def compute_gradcam(
    model,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convenience function to compute Grad-CAM.
    
    Args:
        model: Trained PressNet
        image_tensor: (1, 3, H, W) or (3, H, W)
        device: 'cpu' or 'cuda'
    
    Returns:
        Heatmap (H, W) in [0, 1]
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    gradcam = GradCAM(model, target_layer_name="features")
    heatmap = gradcam(image_tensor)
    
    return heatmap


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_attention(
    original_crop: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        original_crop: RGB image (H, W, 3) uint8
        heatmap: Attention map (H', W') float [0, 1]
        alpha: Overlay transparency
    
    Returns:
        Visualization (H, W, 3) uint8
    """
    h, w = original_crop.shape[:2]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (alpha * heatmap_colored + (1 - alpha) * original_crop).astype(np.uint8)
    
    return overlay


def create_attention_grid(
    crops: List[np.ndarray],
    heatmaps: List[np.ndarray],
    labels: List[str],
    n_cols: int = 4,
) -> plt.Figure:
    """
    Create a grid showing multiple crops with attention overlays.
    
    Args:
        crops: List of RGB crops (H, W, 3)
        heatmaps: List of attention maps (H, W)
        labels: List of label strings
        n_cols: Columns in grid
    
    Returns:
        Matplotlib figure
    """
    n = len(crops)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        if i < len(crops):
            overlay = visualize_attention(crops[i], heatmaps[i])
            ax.imshow(overlay)
            ax.set_title(labels[i], fontsize=9)
        ax.axis("off")
    
    # Hide unused subplots
    for i in range(n, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    return fig
