# ============================================================================
# CausalCrisis V3 — Core Model Modules
# Bao gồm: Hybrid Disentangler, CrossModal Fusion, Classifier, BackdoorAdj
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Tuple, Dict


# ============================================================================
# Gradient Reversal Layer (GRL)
# ============================================================================
class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer — đảo ngược gradient khi backprop."""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambda_max=1.0, warmup_epochs=10):
        super().__init__()
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self._current_lambda = 0.0
    
    def update_lambda(self, epoch: int):
        """Progressive lambda scheduling."""
        progress = min(epoch / self.warmup_epochs, 1.0)
        # Sigmoid schedule cho smooth transition
        self._current_lambda = self.lambda_max * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self._current_lambda)


# ============================================================================
# Stage 2: Hybrid ICA-Adversarial Disentanglement 🆕
# ============================================================================
class HybridDisentangler(nn.Module):
    """
    Per-Modality Causal Disentanglement với Hybrid ICA-Adversarial.
    
    Cải tiến so với V2:
    - ICA-inspired initialization cho stable decomposition (từ CCA, Jiang 2025)
    - Adversarial refinement cho domain-specific adaptation
    - Per-modality (không joint như CAMO)
    
    Architecture:
        input (768) → ICA whitening → ICA rotation → [C_raw, S_raw]
        C_raw → Causal Refiner → C_m (384)
        S_raw → Spurious Refiner → S_m (384)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        causal_dim: int = 384,
        spurious_dim: int = 384,
        dropout: float = 0.1,
        use_ica_init: bool = True
    ):
        super().__init__()
        self.use_ica_init = use_ica_init
        output_dim = causal_dim + spurious_dim
        
        if use_ica_init:
            # ICA-inspired layers (whitening + rotation)
            # Whitening: center + decorrelate features
            self.ica_whiten = nn.Linear(input_dim, input_dim, bias=True)
            # Rotation: separate into independent components
            self.ica_rotate = nn.Linear(input_dim, output_dim, bias=False)
            
            # Khởi tạo orthogonal cho rotation matrix
            nn.init.orthogonal_(self.ica_rotate.weight)
        else:
            # Standard MLP decomposition (V2 style)
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Causal refinement head
        refine_input = causal_dim if use_ica_init else hidden_dim
        self.causal_head = nn.Sequential(
            nn.Linear(refine_input, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Spurious refinement head
        refine_input_s = spurious_dim if use_ica_init else hidden_dim
        self.spurious_head = nn.Sequential(
            nn.Linear(refine_input_s, spurious_dim),
            nn.LayerNorm(spurious_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — CLIP features
        Returns:
            C_m: (B, causal_dim) — causal features (domain-invariant)
            S_m: (B, spurious_dim) — spurious features (domain-specific)
        """
        if self.use_ica_init:
            # ICA decomposition
            h = self.ica_whiten(x)
            components = self.ica_rotate(h)
            C_raw = components[:, :self.causal_dim]
            S_raw = components[:, self.causal_dim:]
        else:
            # Standard MLP
            h = self.shared(x)
            C_raw = h
            S_raw = h
        
        # Adversarial refinement
        C_m = self.causal_head(C_raw)
        S_m = self.spurious_head(S_raw)
        
        return C_m, S_m


# ============================================================================
# Stage 3: Cross-Modal Causal Fusion
# ============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion: C_v attends to C_t and vice versa.
    Gated mechanism learns how much cross-modal information to use.
    """
    
    def __init__(self, d_model: int = 384, nhead: int = 8):
        super().__init__()
        # Cross-attention: V → T, T → V
        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        
        # Gating mechanism
        self.gate_v = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
    
    def forward(self, C_v: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension (B, D) → (B, 1, D)
        C_v_seq = C_v.unsqueeze(1)
        C_t_seq = C_t.unsqueeze(1)
        
        # Cross-attention
        v2t, attn_v = self.cross_attn_v2t(C_v_seq, C_t_seq, C_t_seq)
        t2v, attn_t = self.cross_attn_t2v(C_t_seq, C_v_seq, C_v_seq)
        
        # Gated fusion
        g_v = self.gate_v(v2t.squeeze(1))
        g_t = self.gate_t(t2v.squeeze(1))
        fused = torch.cat([g_v * v2t.squeeze(1), g_t * t2v.squeeze(1)], dim=-1)
        
        C_vt = self.proj(fused)
        return C_vt


class BilinearFusion(nn.Module):
    """
    Compact Bilinear Pooling for C_vt — simpler alternative. 🆕
    Element-wise interaction + gating.
    """
    
    def __init__(self, d_model: int = 384, output_dim: int = 384):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, C_v: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        interaction = self.bilinear(C_v, C_t)
        concat = torch.cat([C_v, C_t], dim=-1)
        gate = self.gate(concat)
        C_vt = self.norm(gate * interaction)
        return C_vt


# ============================================================================
# Domain Classifier (for adversarial training)
# ============================================================================
class DomainClassifier(nn.Module):
    """Phân loại domain (disaster type) từ features."""
    
    def __init__(self, input_dim: int = 384, num_domains: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_domains)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ============================================================================
# Stage 4: Classifier Head
# ============================================================================
class CausalClassifier(nn.Module):
    """Classification head trên concatenated causal features."""
    
    def __init__(
        self,
        input_dim: int = 1152,  # 384*3
        hidden_dims: list = None,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, C_v: torch.Tensor, C_t: torch.Tensor, C_vt: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([C_v, C_t, C_vt], dim=-1)
        return self.head(combined)


# ============================================================================
# Backdoor Adjustment (Inference-time Causal Intervention)
# ============================================================================
class BackdoorAdjustment:
    """
    P(Y | do(C)) = Σ_s P(Y | C, S=s) · P(s)
    Monte Carlo approximation tại inference time.
    """
    
    def __init__(self, bank_size: int = 1000, n_samples: int = 30):
        self.bank_size = bank_size
        self.n_samples = n_samples
        self.memory_bank_v: list = []  # Spurious visual
        self.memory_bank_t: list = []  # Spurious text
    
    def update(self, S_v: torch.Tensor, S_t: torch.Tensor):
        """Cập nhật memory bank trong training (FIFO)."""
        for i in range(S_v.size(0)):
            self.memory_bank_v.append(S_v[i].detach().cpu())
            self.memory_bank_t.append(S_t[i].detach().cpu())
        
        # FIFO: giữ lại bank_size cuối
        if len(self.memory_bank_v) > self.bank_size:
            self.memory_bank_v = self.memory_bank_v[-self.bank_size:]
            self.memory_bank_t = self.memory_bank_t[-self.bank_size:]
    
    @torch.no_grad()
    def intervene(self, C_combined: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        """Monte Carlo backdoor adjustment."""
        if len(self.memory_bank_v) == 0:
            return classifier.head(C_combined)
        
        logits_list = []
        for _ in range(self.n_samples):
            idx = random.randint(0, len(self.memory_bank_v) - 1)
            # Lấy spurious features ngẫu nhiên và cộng vào
            s_v = self.memory_bank_v[idx].to(C_combined.device)
            s_t = self.memory_bank_t[idx].to(C_combined.device)
            s_combined = torch.cat([s_v, s_t, torch.zeros_like(s_v)], dim=-1)
            
            # Predict với spurious nhiễu
            noisy_input = C_combined + s_combined.unsqueeze(0).expand_as(C_combined)
            logits = classifier.head(noisy_input)
            logits_list.append(logits)
        
        # Trung bình loại bỏ spurious correlation
        return torch.stack(logits_list).mean(dim=0)
    
    @property
    def bank_filled(self) -> bool:
        return len(self.memory_bank_v) >= self.bank_size // 2


# ============================================================================
# Full CausalCrisis V3 Model
# ============================================================================
class CausalCrisisV3(nn.Module):
    """
    CausalCrisis V3 — Full Pipeline
    
    Stage 1: CLIP features (pre-cached, not in this module)
    Stage 2: Per-modality Hybrid ICA-Adversarial Disentanglement
    Stage 3: Cross-modal Causal Fusion (C_vt)
    Stage 4: Classification + Backdoor Adjustment
    
    Improvements over V2:
    1. Hybrid ICA+Adversarial disentanglement (CCA-inspired)
    2. SupCon loss trên causal features
    3. Bilinear OR CrossAttn fusion (ablation)
    4. Adaptive Loss Weighting
    5. 2-Phase training + Cosine Warm Restarts
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        causal_dim: int = 384,
        spurious_dim: int = 384,
        num_classes: int = 2,
        num_domains: int = 7,
        nhead: int = 8,
        dropout: float = 0.1,
        use_ica_init: bool = True,
        fusion_type: str = "cross_attention",
        grl_lambda_max: float = 1.0,
        grl_warmup_epochs: int = 10,
    ):
        super().__init__()
        
        # Stage 2: Per-modality disentanglement
        self.visual_disentangler = HybridDisentangler(
            input_dim=input_dim,
            causal_dim=causal_dim,
            spurious_dim=spurious_dim,
            dropout=dropout,
            use_ica_init=use_ica_init,
        )
        self.text_disentangler = HybridDisentangler(
            input_dim=input_dim,
            causal_dim=causal_dim,
            spurious_dim=spurious_dim,
            dropout=dropout,
            use_ica_init=use_ica_init,
        )
        
        # Stage 3: Cross-modal fusion
        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(d_model=causal_dim, nhead=nhead)
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(d_model=causal_dim, output_dim=causal_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Stage 4: Classifier
        classifier_input = causal_dim * 3  # C_v + C_t + C_vt
        self.classifier = CausalClassifier(
            input_dim=classifier_input,
            num_classes=num_classes,
            dropout=dropout * 2,  # More dropout in classifier
        )
        
        # Domain classifiers (adversarial, per-modality)
        self.grl = GRL(lambda_max=grl_lambda_max, warmup_epochs=grl_warmup_epochs)
        self.domain_cls_v = DomainClassifier(causal_dim, num_domains)
        self.domain_cls_t = DomainClassifier(causal_dim, num_domains)
        
        # Backdoor Adjustment
        self.ba = BackdoorAdjustment()
        
        # Track dimensions
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim
    
    def forward(
        self,
        f_v: torch.Tensor,
        f_t: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        use_ba: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            f_v: (B, 768) — cached CLIP visual features
            f_t: (B, 768) — cached CLIP text features
            domain_labels: (B,) — disaster domain index (training only)
            use_ba: bool — dùng Backdoor Adjustment (eval)
        
        Returns:
            dict with keys: logits, C_v, C_t, S_v, S_t, C_vt, domain_logits_v, domain_logits_t
        """
        # Stage 2: Disentangle
        C_v, S_v = self.visual_disentangler(f_v)
        C_t, S_t = self.text_disentangler(f_t)
        
        # Stage 3: Cross-modal fusion
        C_vt = self.fusion(C_v, C_t)
        
        # Stage 4: Classification
        if use_ba and self.ba.bank_filled and not self.training:
            # Backdoor Adjustment tại inference
            C_combined = torch.cat([C_v, C_t, C_vt], dim=-1)
            logits = self.ba.intervene(C_combined, self.classifier)
        else:
            logits = self.classifier(C_v, C_t, C_vt)
        
        # Update memory bank during training
        if self.training and domain_labels is not None:
            self.ba.update(S_v, S_t)
        
        # Domain classification (adversarial)
        domain_logits_v = None
        domain_logits_t = None
        if self.training and domain_labels is not None:
            # GRL đảo gradient → train causal features domain-invariant
            C_v_reversed = self.grl(C_v)
            C_t_reversed = self.grl(C_t)
            domain_logits_v = self.domain_cls_v(C_v_reversed)
            domain_logits_t = self.domain_cls_t(C_t_reversed)
        
        return {
            "logits": logits,
            "C_v": C_v, "C_t": C_t,
            "S_v": S_v, "S_t": S_t,
            "C_vt": C_vt,
            "domain_logits_v": domain_logits_v,
            "domain_logits_t": domain_logits_t,
        }
    
    def get_trainable_params(self) -> int:
        """Đếm số parameters cần train."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def update_grl(self, epoch: int):
        """Update GRL lambda cho epoch hiện tại."""
        self.grl.update_lambda(epoch)


# ============================================================================
# MLP Baseline (for H1 experiment)
# ============================================================================
class CLIPMLPBaseline(nn.Module):
    """Simple MLP baseline trên CLIP features (no causal)."""
    
    def __init__(
        self,
        input_dim: int = 1536,  # 768 + 768 (concat image + text)
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, f_v: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([f_v, f_t], dim=-1)
        return self.head(combined)
