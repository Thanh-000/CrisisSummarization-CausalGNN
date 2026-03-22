# ============================================================================
# CausalCrisis V3 — Archived Model Modules
# These components proved ineffective in V3 experiments (ceiling ~78% F1w)
# Archived for reference and potential future reuse (H3, H4, H7)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Tuple, Dict

from .models import CLIPTaskAdapter


# ============================================================================
# Gradient Reversal Layer (GRL)
# V3 Result: Destabilized training when combined with ICA disentanglement
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
        self._current_lambda = self.lambda_max * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self._current_lambda)


# ============================================================================
# Hybrid ICA-Adversarial Disentanglement
# V3 Result: Only +0.3% F1 improvement (insignificant)
# ============================================================================
class HybridDisentangler(nn.Module):
    """
    Per-Modality Causal Disentanglement với Hybrid ICA-Adversarial.

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
            self.ica_whiten = nn.Linear(input_dim, input_dim, bias=True)
            self.ica_rotate = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.orthogonal_(self.ica_rotate.weight)
        else:
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        refine_input = causal_dim if use_ica_init else hidden_dim
        self.causal_head = nn.Sequential(
            nn.Linear(refine_input, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

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
        if self.use_ica_init:
            h = self.ica_whiten(x)
            components = self.ica_rotate(h)
            C_raw = components[:, :self.causal_dim]
            S_raw = components[:, self.causal_dim:]
        else:
            h = self.shared(x)
            C_raw = h
            S_raw = h

        C_m = self.causal_head(C_raw)
        S_m = self.spurious_head(S_raw)

        return C_m, S_m


# ============================================================================
# Cross-Modal Causal Fusion
# V3 Result: Degenerate at seq_len=1 (MultiheadAttention gives identity)
# ============================================================================
class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion — degenerate with 1D vectors."""

    def __init__(self, d_model: int = 384, nhead: int = 8):
        super().__init__()
        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.gate_v = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, C_v: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        C_v_seq = C_v.unsqueeze(1)
        C_t_seq = C_t.unsqueeze(1)

        v2t, _ = self.cross_attn_v2t(C_v_seq, C_t_seq, C_t_seq)
        t2v, _ = self.cross_attn_t2v(C_t_seq, C_v_seq, C_v_seq)

        g_v = self.gate_v(v2t.squeeze(1))
        g_t = self.gate_t(t2v.squeeze(1))
        fused = torch.cat([g_v * v2t.squeeze(1), g_t * t2v.squeeze(1)], dim=-1)

        return self.proj(fused)


class BilinearFusion(nn.Module):
    """Compact Bilinear Pooling — simpler alternative."""

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
        return self.norm(gate * interaction)


# ============================================================================
# Domain Classifier (for adversarial training)
# ============================================================================
class DomainClassifier(nn.Module):
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
# Classifier Head (V3)
# ============================================================================
class CausalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1152,
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

    def forward(self, C_v, C_t, C_vt):
        combined = torch.cat([C_v, C_t, C_vt], dim=-1)
        return self.head(combined)


# ============================================================================
# Backdoor Adjustment
# V3 Result: Never activated properly during training
# ============================================================================
class BackdoorAdjustment:
    """P(Y | do(C)) = Σ_s P(Y | C, S=s) · P(s)"""

    def __init__(self, bank_size: int = 1000, n_samples: int = 30):
        self.bank_size = bank_size
        self.n_samples = n_samples
        self.memory_bank_v: list = []
        self.memory_bank_t: list = []

    def update(self, S_v: torch.Tensor, S_t: torch.Tensor):
        for i in range(S_v.size(0)):
            self.memory_bank_v.append(S_v[i].detach().cpu())
            self.memory_bank_t.append(S_t[i].detach().cpu())

        if len(self.memory_bank_v) > self.bank_size:
            self.memory_bank_v = self.memory_bank_v[-self.bank_size:]
            self.memory_bank_t = self.memory_bank_t[-self.bank_size:]

    @torch.no_grad()
    def intervene(self, C_combined, classifier):
        if len(self.memory_bank_v) == 0:
            return classifier.head(C_combined)

        logits_list = []
        for _ in range(self.n_samples):
            idx = random.randint(0, len(self.memory_bank_v) - 1)
            s_v = self.memory_bank_v[idx].to(C_combined.device)
            s_t = self.memory_bank_t[idx].to(C_combined.device)
            s_combined = torch.cat([s_v, s_t, torch.zeros_like(s_v)], dim=-1)
            noisy_input = C_combined + s_combined.unsqueeze(0).expand_as(C_combined)
            logits = classifier.head(noisy_input)
            logits_list.append(logits)

        return torch.stack(logits_list).mean(dim=0)

    @property
    def bank_filled(self):
        return len(self.memory_bank_v) >= self.bank_size // 2


# ============================================================================
# Full CausalCrisis V3 Model (ARCHIVED)
# Best result: 78.3% F1w — far below 90% target
# ============================================================================
class CausalCrisisV3(nn.Module):
    """
    CausalCrisis V3 — Full Pipeline (archived).

    Stage 1: CLIP features (pre-cached)
    Stage 2: Per-modality Hybrid ICA-Adversarial Disentanglement
    Stage 3: Cross-modal Causal Fusion (C_vt)
    Stage 4: Classification + Backdoor Adjustment
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
        use_adapter: bool = True,
        adapter_bottleneck: int = 128,
        adapter_residual_ratio: float = 0.2,
    ):
        super().__init__()

        self.use_adapter = use_adapter
        if use_adapter:
            self.image_adapter = CLIPTaskAdapter(
                dim=input_dim, bottleneck=adapter_bottleneck,
                residual_ratio=adapter_residual_ratio, dropout=dropout,
            )
            self.text_adapter = CLIPTaskAdapter(
                dim=input_dim, bottleneck=adapter_bottleneck,
                residual_ratio=adapter_residual_ratio, dropout=dropout,
            )

        self.visual_disentangler = HybridDisentangler(
            input_dim=input_dim, causal_dim=causal_dim,
            spurious_dim=spurious_dim, dropout=dropout, use_ica_init=use_ica_init,
        )
        self.text_disentangler = HybridDisentangler(
            input_dim=input_dim, causal_dim=causal_dim,
            spurious_dim=spurious_dim, dropout=dropout, use_ica_init=use_ica_init,
        )

        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(d_model=causal_dim, nhead=nhead)
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(d_model=causal_dim, output_dim=causal_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        classifier_input = causal_dim * 3
        self.classifier = CausalClassifier(
            input_dim=classifier_input, num_classes=num_classes,
            dropout=dropout * 2,
        )

        self.grl = GRL(lambda_max=grl_lambda_max, warmup_epochs=grl_warmup_epochs)
        self.domain_cls_v = DomainClassifier(causal_dim, num_domains)
        self.domain_cls_t = DomainClassifier(causal_dim, num_domains)
        self.ba = BackdoorAdjustment()
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim

    def forward(self, f_v, f_t, domain_labels=None, use_ba=False):
        if self.use_adapter:
            f_v = self.image_adapter(f_v)
            f_t = self.text_adapter(f_t)

        C_v, S_v = self.visual_disentangler(f_v)
        C_t, S_t = self.text_disentangler(f_t)
        C_vt = self.fusion(C_v, C_t)

        if use_ba and self.ba.bank_filled and not self.training:
            C_combined = torch.cat([C_v, C_t, C_vt], dim=-1)
            logits = self.ba.intervene(C_combined, self.classifier)
        else:
            logits = self.classifier(C_v, C_t, C_vt)

        if self.training and domain_labels is not None:
            self.ba.update(S_v, S_t)

        domain_logits_v, domain_logits_t = None, None
        if self.training and domain_labels is not None:
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

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_grl(self, epoch: int):
        self.grl.update_lambda(epoch)
