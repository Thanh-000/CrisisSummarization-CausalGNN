# ============================================================================
# CausalCrisis — Loss Functions (V4-active)
# V3 archived losses → see losses_v3.py
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: giảm trọng số cho easy samples, focus vào hard samples.

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights tensor (optional).
        gamma: Focusing parameter (default=2.0).
        label_smoothing: Label smoothing factor (default=0.0 for backward compat).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
