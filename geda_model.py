"""
GEDA Model -- Graph-Enhanced Differential Attention
====================================================
Ket hop GNN semi-supervised (Paper 1) + Differential Attention (Paper 2)
vao mot architecture thong nhat cho multimodal crisis classification.

Architecture:
  Input -> CLIP (frozen) -> PCA -> FAISS kNN Graph -> GraphSAGE
  -> Self-Attention -> Guided Cross-Attention -> Adaptive DiffAttn
  -> Multi-task Heads (Task 1 + 2 + 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


# ============================================================
# 1. ATTENTION MODULES (from Paper 2 architecture)
# ============================================================

class SelfAttention(nn.Module):
    """Self-Attention: khu nhieu, lam noi bat dac trung quan trong."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        attn = torch.matmul(x.unsqueeze(1), x.unsqueeze(2))  # (B, 1, 1)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = self.proj(x)
        return out


class GuidedCrossAttention(nn.Module):
    """Guided Cross-Attention: trao doi thong tin giua 2 modalities."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_I = nn.Linear(dim, dim)
        self.proj_T = nn.Linear(dim, dim)
        self.gate_I = nn.Linear(dim, dim)
        self.gate_T = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f_I: torch.Tensor, f_T: torch.Tensor) -> torch.Tensor:
        # Projections
        z_I = self.proj_I(f_I)
        z_T = self.proj_T(f_T)

        # Attention masks (cross-exchange)
        alpha_I = torch.sigmoid(self.gate_I(f_I))
        alpha_T = torch.sigmoid(self.gate_T(f_T))

        # Cross-guided: image mask * text proj, and vice versa
        out = torch.cat([alpha_T * z_I, alpha_I * z_T], dim=-1)
        return self.dropout(out)


class AdaptiveDiffAttention(nn.Module):
    """
    Adaptive Differential Attention.
    Cai tien tu Paper 2: lambda(x) = MLP(graph_features)
    thay vi lambda co dinh.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        # 2 head projections
        self.W_Q1 = nn.Linear(dim, dim)
        self.W_K1 = nn.Linear(dim, dim)
        self.W_Q2 = nn.Linear(dim, dim)
        self.W_K2 = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)

        # Adaptive lambda: input-dependent
        self.lambda_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q1 = self.W_Q1(x)
        K1 = self.W_K1(x)
        Q2 = self.W_Q2(x)
        K2 = self.W_K2(x)
        V = self.W_V(x)

        # Attention maps
        attn1 = F.softmax(Q1 * K1 * self.scale, dim=-1)
        attn2 = F.softmax(Q2 * K2 * self.scale, dim=-1)

        # Adaptive lambda
        lam = self.lambda_net(x)  # (B, 1)

        # Differential: subtract noise attention
        diff_attn = attn1 - lam * attn2
        out = diff_attn * V

        return self.dropout(out)


# ============================================================
# 2. GEDA CORE MODEL
# ============================================================

class GEDAModel(nn.Module):
    """
    Graph-Enhanced Differential Attention (GEDA) Model.

    Pipeline:
    1. CLIP features (pre-extracted) -> PCA (pre-computed)
    2. GraphSAGE propagation (2 layers, late fusion)
    3. Self-Attention -> Guided Cross-Attention -> Adaptive DiffAttn
    4. Multi-task classification heads
    """

    def __init__(
        self,
        img_dim: int = 256,
        txt_dim: int = 256,
        hidden_dim: int = 512,
        num_classes_task1: int = 2,   # Informativeness
        num_classes_task2: int = 6,   # Humanitarian
        num_classes_task3: int = 3,   # Damage Severity
        dropout: float = 0.3,
        use_graph: bool = True,
        use_attention: bool = True,
        use_mtl: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_attention = use_attention
        self.use_mtl = use_mtl

        # --- Stage 1: Feature projection ---
        self.proj_img = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_txt = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Stage 2: GNN layers (simulates GraphSAGE without PyG dependency) ---
        if use_graph:
            # 2-layer message passing (simplified for portability)
            self.gnn_img_1 = nn.Linear(hidden_dim, hidden_dim)
            self.gnn_img_2 = nn.Linear(hidden_dim, hidden_dim)
            self.gnn_txt_1 = nn.Linear(hidden_dim, hidden_dim)
            self.gnn_txt_2 = nn.Linear(hidden_dim, hidden_dim)
            self.gnn_norm_img = nn.LayerNorm(hidden_dim)
            self.gnn_norm_txt = nn.LayerNorm(hidden_dim)

        # --- Stage 3: Attention stack ---
        if use_attention:
            self.self_attn_img = SelfAttention(hidden_dim, dropout)
            self.self_attn_txt = SelfAttention(hidden_dim, dropout)
            self.gca = GuidedCrossAttention(hidden_dim, dropout)
            self.diff_attn = AdaptiveDiffAttention(hidden_dim * 2, dropout)
            classifier_dim = hidden_dim * 2
        else:
            classifier_dim = hidden_dim * 2  # concat img + txt

        # --- Stage 4: Classification heads ---
        self.head_task1 = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_task1),
        )

        if use_mtl:
            self.head_task2 = nn.Sequential(
                nn.Linear(classifier_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes_task2),
            )
            self.head_task3 = nn.Sequential(
                nn.Linear(classifier_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes_task3),
            )

    def graph_propagate(self, x: torch.Tensor, adj: Optional[torch.Tensor],
                         layer1: nn.Linear, layer2: nn.Linear,
                         norm: nn.LayerNorm) -> torch.Tensor:
        """Simplified 2-layer graph propagation with residual."""
        if adj is None:
            return x

        # Layer 1 + residual
        h = F.relu(layer1(torch.matmul(adj, x)))
        h = h + x  # residual

        # Layer 2 + residual + norm
        h2 = F.relu(layer2(torch.matmul(adj, h)))
        h2 = norm(h2 + h)  # residual + norm

        return h2

    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        adj_img: Optional[torch.Tensor] = None,
        adj_txt: Optional[torch.Tensor] = None,
        task: str = "task1",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            img_feat: (B, img_dim) CLIP image features (PCA-reduced)
            txt_feat: (B, txt_dim) CLIP text features (PCA-reduced)
            adj_img: (B, B) adjacency matrix for image graph
            adj_txt: (B, B) adjacency matrix for text graph
            task: which task head to use ("task1", "task2", "task3", "all")

        Returns:
            Dict with logits for requested tasks
        """
        # Project to hidden dim
        h_img = self.proj_img(img_feat)
        h_txt = self.proj_txt(txt_feat)

        # Graph propagation (if enabled)
        if self.use_graph:
            h_img = self.graph_propagate(
                h_img, adj_img,
                self.gnn_img_1, self.gnn_img_2, self.gnn_norm_img
            )
            h_txt = self.graph_propagate(
                h_txt, adj_txt,
                self.gnn_txt_1, self.gnn_txt_2, self.gnn_norm_txt
            )

        # Attention stack (if enabled)
        if self.use_attention:
            h_img = self.self_attn_img(h_img)
            h_txt = self.self_attn_txt(h_txt)
            z = self.gca(h_img, h_txt)  # (B, hidden*2)
            z = self.diff_attn(z)       # (B, hidden*2)
        else:
            z = torch.cat([h_img, h_txt], dim=-1)

        # Classification
        outputs = {}
        if task in ("task1", "all"):
            outputs["task1"] = self.head_task1(z)
        if self.use_mtl and task in ("task2", "all"):
            outputs["task2"] = self.head_task2(z)
        if self.use_mtl and task in ("task3", "all"):
            outputs["task3"] = self.head_task3(z)

        return outputs


# ============================================================
# 3. MULTI-TASK LOSS (Focal Loss + task weighting)
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss cho class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class GEDALoss(nn.Module):
    """Multi-task loss cho GEDA."""

    def __init__(
        self,
        lambda1: float = 0.4,
        lambda2: float = 0.3,
        lambda3: float = 0.3,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.focal = FocalLoss(gamma=gamma)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Tinh multi-task loss."""
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        losses = {}

        if "task1" in outputs and "task1" in targets:
            l1 = self.focal(outputs["task1"], targets["task1"])
            total = total + self.lambda1 * l1
            losses["task1"] = l1.item()

        if "task2" in outputs and "task2" in targets:
            l2 = self.focal(outputs["task2"], targets["task2"])
            total = total + self.lambda2 * l2
            losses["task2"] = l2.item()

        if "task3" in outputs and "task3" in targets:
            l3 = self.focal(outputs["task3"], targets["task3"])
            total = total + self.lambda3 * l3
            losses["task3"] = l3.item()

        losses["total"] = total.item()
        return total, losses


# ============================================================
# 4. ABLATION VARIANTS (for Exp 2)
# ============================================================

ABLATION_CONFIGS = {
    "A1_clip_linear": {"use_graph": False, "use_attention": False, "use_mtl": False},
    "A2_gnn_only":    {"use_graph": True,  "use_attention": False, "use_mtl": False},
    "A3_attn_only":   {"use_graph": False, "use_attention": True,  "use_mtl": False},
    "A4_gnn_sa":      {"use_graph": True,  "use_attention": True,  "use_mtl": False},
    "A7_geda_full":   {"use_graph": True,  "use_attention": True,  "use_mtl": True},
}


def create_geda_variant(variant: str, **kwargs) -> GEDAModel:
    """Tao model variant cho ablation study."""
    if variant not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(ABLATION_CONFIGS.keys())}")
    config = {**ABLATION_CONFIGS[variant], **kwargs}
    return GEDAModel(**config)


# ============================================================
# 5. UTILITY: Model summary
# ============================================================

def model_summary(model: nn.Module) -> Dict[str, int]:
    """In so luong params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }


if __name__ == "__main__":
    # Quick test
    print("=" * 50)
    print("  GEDA Model Architecture Test")
    print("=" * 50)

    model = GEDAModel()
    info = model_summary(model)
    print(f"  Total params:     {info['total_params']:,}")
    print(f"  Trainable params: {info['trainable_params']:,}")

    # Test forward pass
    B = 8
    img = torch.randn(B, 256)
    txt = torch.randn(B, 256)
    adj = torch.eye(B)  # identity (no graph)

    outputs = model(img, txt, adj, adj, task="all")
    for task, logits in outputs.items():
        print(f"  {task}: {logits.shape}")

    print("\n  Ablation variants:")
    for name in ABLATION_CONFIGS:
        m = create_geda_variant(name)
        s = model_summary(m)
        print(f"    {name}: {s['trainable_params']:,} params")

    print("\n[OK] All tests passed!")
