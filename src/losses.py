# ============================================================================
# CausalCrisis V3 — Loss Functions
# Bao gồm: FocalLoss, Orthogonal, SupCon, Adaptive Weighting
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# ============================================================================
# Focal Loss (xử lý class imbalance)
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: giảm trọng số cho easy samples, focus vào hard samples.
    L = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Class weights (optional)
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# Orthogonal Regularization (force C ⊥ S)
# ============================================================================
class OrthogonalLoss(nn.Module):
    """
    Force causal features orthogonal to spurious features.
    L_ortho = |C · S| / (||C|| · ||S||)
    """
    
    def forward(self, C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        # Normalize
        C_norm = F.normalize(C, dim=-1)
        S_norm = F.normalize(S, dim=-1)
        
        # Cosine similarity (should be ~0)
        cos_sim = (C_norm * S_norm).sum(dim=-1)
        return cos_sim.abs().mean()


# ============================================================================
# Supervised Contrastive Loss trên Causal Features 🆕
# ============================================================================
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Kéo causal features cùng class lại gần, đẩy khác class ra xa.
    
    Đây là cải tiến mới — ortho chỉ C⊥S nhưng không đảm bảo C discriminative.
    SupCon đảm bảo causal features cùng class gần nhau across domains.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) — causal features (e.g., concat [C_v, C_t, C_vt])
            labels: (B,) — class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Similarity matrix (B, B)
        sim_matrix = torch.mm(features, features.T) / self.temperature
        
        # Tạo mask cho positive and negative pairs
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_pos.fill_diagonal_(0)  # Bỏ self-pairs
        
        # Nếu không có positive pairs → return 0
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Log-sum-exp stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Mask out self-similarity
        self_mask = torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * (1 - self_mask)
        
        # Log probability
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log-prob over positive pairs
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob.mean()
        return loss


# ============================================================================
# Adaptive Loss Weighting 🆕
# ============================================================================
class AdaptiveLossWeighting(nn.Module):
    """
    Uncertainty-based Adaptive Loss Weighting (Kendall et al., 2018).
    Tự động learn trọng số cho mỗi loss component.
    
    L_total = Σ (1/(2*σ²_i)) * L_i + log(σ_i)
    
    🔧 Fix: init log_vars=2.0 (precision=0.14) thay vì 0.0 (precision=1.0)
    để tránh auxiliary losses nhận full weight ngay lập tức.
    """
    
    def __init__(self, n_losses: int = 4, init_logvar: float = 2.0):
        super().__init__()
        # 🔧 Khởi tạo log_var > 0 → precision thấp ban đầu → gentle start
        self.log_vars = nn.Parameter(torch.full((n_losses,), init_logvar))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: list of loss values [L_focal, L_adv, L_ortho, L_supcon, ...]
        Returns:
            weighted total loss
        """
        total = 0
        for i, loss in enumerate(losses):
            if i >= len(self.log_vars):
                total += loss
                continue
            
            # Precision = 1/σ² = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            # Weighted loss + regularization
            total += precision * loss + self.log_vars[i]
        
        return total
    
    def get_weights(self) -> Dict[str, float]:
        """Return current learned weights (for logging)."""
        weights = {}
        names = ["focal", "adversarial", "orthogonal", "supcon"]
        for i, log_var in enumerate(self.log_vars):
            name = names[i] if i < len(names) else f"loss_{i}"
            weights[name] = torch.exp(-log_var).item()
        return weights


# ============================================================================
# Combined Loss Function (with gradual activation 🔧)
# ============================================================================
class CausalCrisisLoss(nn.Module):
    """
    Combined loss cho CausalCrisis V3.
    
    L = L_focal + ramp * [α₁(L_adv) + α₂(L_ortho) + α₃·L_supcon]
    
    🔧 Fixes:
    1. Gradual loss ramp-up thay vì sudden activation
    2. Adaptive weights bắt đầu với low precision
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        focal_gamma: float = 2.0,
        alpha_adv: float = 0.1,
        alpha_ortho: float = 0.05,
        alpha_supcon: float = 0.1,
        supcon_temperature: float = 0.07,
        use_adaptive: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        # 🔧 New params for stability
        loss_ramp_epochs: int = 20,
        adaptive_init_logvar: float = 2.0,
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.ortho_loss = OrthogonalLoss()
        self.supcon_loss = SupConLoss(temperature=supcon_temperature)
        self.domain_ce = nn.CrossEntropyLoss()
        
        self.alpha_adv = alpha_adv
        self.alpha_ortho = alpha_ortho
        self.alpha_supcon = alpha_supcon
        self.use_adaptive = use_adaptive
        self.loss_ramp_epochs = loss_ramp_epochs
        
        if use_adaptive:
            self.adaptive_weights = AdaptiveLossWeighting(
                n_losses=4, init_logvar=adaptive_init_logvar
            )
        else:
            self.adaptive_weights = None
    
    def _get_ramp_factor(self, epoch: int, warmup_epochs: int) -> float:
        """
        🔧 Gradual ramp-up factor cho auxiliary losses.
        Tránh bật đồng loạt khi Phase 2 bắt đầu.
        
        Epoch < warmup: 0.0 (chỉ focal loss)
        Epoch warmup → warmup+ramp: 0.0 → 1.0 (linear ramp)
        Epoch > warmup+ramp: 1.0 (full weight)
        """
        if epoch < warmup_epochs:
            return 0.0
        ramp_progress = (epoch - warmup_epochs) / max(self.loss_ramp_epochs, 1)
        return min(ramp_progress, 1.0)
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        warmup_epochs: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with gradual activation.
        
        Args:
            model_output: dict from CausalCrisisV3.forward()
            labels: (B,) class labels
            domain_labels: (B,) domain labels (optional)
            epoch: current epoch
            warmup_epochs: number of warmup epochs (Phase 1)
        
        Returns:
            dict with total_loss and individual loss values
        """
        losses = {}
        
        # 🔧 Ramp factor for auxiliary losses
        ramp = self._get_ramp_factor(epoch, warmup_epochs)
        losses["ramp_factor"] = ramp
        
        # 1. Focal Loss (classification) — always active
        L_focal = self.focal_loss(model_output["logits"], labels)
        losses["focal"] = L_focal
        
        # 2. Orthogonal Loss (per-modality)
        L_ortho_v = self.ortho_loss(model_output["C_v"], model_output["S_v"])
        L_ortho_t = self.ortho_loss(model_output["C_t"], model_output["S_t"])
        L_ortho = L_ortho_v + L_ortho_t
        losses["ortho"] = L_ortho
        
        # 3. Adversarial Loss (per-modality)
        L_adv = torch.tensor(0.0, device=labels.device)
        if domain_labels is not None and model_output["domain_logits_v"] is not None:
            L_adv_v = self.domain_ce(model_output["domain_logits_v"], domain_labels)
            L_adv_t = self.domain_ce(model_output["domain_logits_t"], domain_labels)
            L_adv = L_adv_v + L_adv_t
        losses["adversarial"] = L_adv
        
        # 4. SupCon Loss
        causal_concat = torch.cat([
            model_output["C_v"], model_output["C_t"], model_output["C_vt"]
        ], dim=-1)
        L_supcon = self.supcon_loss(causal_concat, labels)
        losses["supcon"] = L_supcon
        
        # Total loss with 🔧 gradual ramp-up
        if self.use_adaptive and self.adaptive_weights is not None:
            # Apply ramp to auxiliary losses BEFORE adaptive weighting
            loss_list = [L_focal, ramp * L_adv, ramp * L_ortho, ramp * L_supcon]
            total = self.adaptive_weights(loss_list)
            losses["adaptive_weights"] = self.adaptive_weights.get_weights()
        else:
            total = (
                L_focal 
                + ramp * self.alpha_adv * L_adv 
                + ramp * self.alpha_ortho * L_ortho 
                + ramp * self.alpha_supcon * L_supcon
            )
        
        losses["total"] = total
        return losses
