# ============================================================================
# CausalCrisis — Core Model Modules (V4-active)
# Includes: CLIPTaskAdapter, CLIPMLPBaseline, GuidedCrossAttention,
#           ThreeModalityClassifier
# V3 archived components → see models_v3.py
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# CLIP Task Adapter — Domain-Specific Feature Adaptation
# Reference: CLIP-Adapter (Gao et al., 2024)
# ============================================================================
class CLIPTaskAdapter(nn.Module):
    """
    Lightweight residual adapter cho frozen CLIP features.

    Architecture:
        x → MLP(dim → bottleneck → dim) → adapter_out
        output = (1 - ratio) * x + ratio * adapter_out
    """

    def __init__(
        self,
        dim: int = 768,
        bottleneck: int = 128,
        residual_ratio: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        self.ratio = residual_ratio
        self.ln = nn.LayerNorm(dim)

        # Khởi tạo near-zero để ban đầu output ≈ input
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(x)
        out = (1 - self.ratio) * x + self.ratio * adapted
        return self.ln(out)


# ============================================================================
# Guided Cross-Attention (Munia et al., CVPRw 2025)
# Solves seq_len=1 degenerate issue of standard MultiheadAttention
# ============================================================================
class GuidedCrossAttention(nn.Module):
    """
    Guided Cross-Attention — sigmoid masks thay vì softmax attention.

    Munia et al. (CVPRw 2025): "Multimodal Fusion and Classification of
    Crisis-Related Tweets Using CLIP and Cross-Attention"

    So với nn.MultiheadAttention:
    - Không bị degenerate khi seq_len=1 (softmax(x/1) = 1.0)
    - Sigmoid masks giữ element-wise interaction
    - Works well on pooled CLS vectors (B, D) without sequence dim
    """

    def __init__(self, d_model: int = 768, dropout: float = 0.1):
        super().__init__()

        # Self-attention refinement cho mỗi modality
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_t = nn.Linear(d_model, d_model)

        # Projection layers
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_t = nn.Linear(d_model, d_model)

        # Sigmoid masks (key difference vs standard attention)
        self.mask_v = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.mask_t = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        # Layer norms
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, f_v: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_v: (B, D) — visual features
            f_t: (B, D) — text features
        Returns:
            z_fused: (B, 2*D) — concatenated cross-guided features
        """
        # Step 1: Self-attention refinement
        v_refined = self.ln_v(f_v + torch.relu(self.self_attn_v(f_v)))
        t_refined = self.ln_t(f_t + torch.relu(self.self_attn_t(f_t)))

        # Step 2: Projections
        z_v = torch.relu(self.proj_v(v_refined))
        z_t = torch.relu(self.proj_t(t_refined))

        # Step 3: Sigmoid attention masks
        alpha_v = self.mask_v(v_refined)  # (B, D)
        alpha_t = self.mask_t(t_refined)  # (B, D)

        # Step 4: Cross-guidance
        guided_v = alpha_t * z_v  # Text guides vision
        guided_t = alpha_v * z_t  # Vision guides text

        # Step 5: Concatenate
        z_fused = torch.cat([
            self.dropout(guided_v),
            self.dropout(guided_t),
        ], dim=-1)  # (B, 2*D)

        return z_fused


# ============================================================================
# Three-Modality Classifier (V4)
# Supports: 2-modal (img+txt), 3-modal (img+txt+llava)
# Fusion: concat or GuidedCA
# ============================================================================
class ThreeModalityClassifier(nn.Module):
    """
    Flexible classifier supporting 2 or 3 modalities with optional GuidedCA.

    Configs:
    - 2-modal concat:  use_guided_ca=False, use_llava=False → input: img+txt
    - 2-modal GCA:     use_guided_ca=True,  use_llava=False → GCA(img, txt)
    - 3-modal concat:  use_guided_ca=False, use_llava=True  → img+txt+llava
    - 3-modal GCA:     use_guided_ca=True,  use_llava=True  → GCA(GCA(img,txt), llava)
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.2,
        use_guided_ca: bool = True,
        use_llava: bool = True,
    ):
        super().__init__()
        self.use_guided_ca = use_guided_ca
        self.use_llava = use_llava

        if use_guided_ca:
            # Guided CA cho image ↔ text
            self.guided_ca_vt = GuidedCrossAttention(feat_dim, dropout=dropout * 0.5)

            if use_llava:
                # Guided CA cho fused ↔ llava
                self.guided_ca_llava = GuidedCrossAttention(feat_dim, dropout=dropout * 0.5)
                # Fused projection: 2*D (from CA) → D
                self.fuse_proj = nn.Sequential(
                    nn.Linear(feat_dim * 2, feat_dim),
                    nn.LayerNorm(feat_dim),
                    nn.GELU(),
                )
                classifier_input = feat_dim * 2
            else:
                classifier_input = feat_dim * 2
        else:
            if use_llava:
                classifier_input = feat_dim * 3
            else:
                classifier_input = feat_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode = "Guided CA" if use_guided_ca else "Concat"
        modalities = "3-modal (img+txt+llava)" if use_llava else "2-modal (img+txt)"
        print(f"   Architecture: {mode} | {modalities} | {n_params:,} params")

    def forward(
        self,
        f_v: torch.Tensor,
        f_t: torch.Tensor,
        f_llava: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_guided_ca:
            z_vt = self.guided_ca_vt(f_v, f_t)  # (B, 2*D)

            if self.use_llava and f_llava is not None:
                z_vt_proj = self.fuse_proj(z_vt)  # (B, D)
                z_final = self.guided_ca_llava(z_vt_proj, f_llava)  # (B, 2*D)
            else:
                z_final = z_vt
        else:
            if self.use_llava and f_llava is not None:
                z_final = torch.cat([f_v, f_t, f_llava], dim=-1)
            else:
                z_final = torch.cat([f_v, f_t], dim=-1)

        return self.classifier(z_final)


# ============================================================================
# MLP Baseline (for ablation)
# ============================================================================
class CLIPMLPBaseline(nn.Module):
    """Simple MLP baseline on concatenated CLIP features."""

    def __init__(
        self,
        input_dim: int = 1536,
        num_classes: int = 2,
        dropout: float = 0.2,
        use_adapter: bool = False,
        adapter_bottleneck: int = 128,
        adapter_residual_ratio: float = 0.2,
    ):
        super().__init__()
        feat_dim = input_dim // 2

        self.use_adapter = use_adapter
        if use_adapter:
            self.image_adapter = CLIPTaskAdapter(
                dim=feat_dim, bottleneck=adapter_bottleneck,
                residual_ratio=adapter_residual_ratio, dropout=dropout * 0.5,
            )
            self.text_adapter = CLIPTaskAdapter(
                dim=feat_dim, bottleneck=adapter_bottleneck,
                residual_ratio=adapter_residual_ratio, dropout=dropout * 0.5,
            )

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
        if self.use_adapter:
            f_v = self.image_adapter(f_v)
            f_t = self.text_adapter(f_t)
        combined = torch.cat([f_v, f_t], dim=-1)
        return self.head(combined)
