"""
CausalCrisis Model -- Causal Multimodal Reasoning for Crisis Generalization
============================================================================
Ke thua va mo rong GEDA (Graph-Enhanced Differential Attention)
voi Causal Disentanglement, Gradient Reversal, va Backdoor Adjustment.

Architecture:
  Input -> CLIP (frozen) -> PCA -> Feature Projection
  -> Causal Disentanglement (C, S) per modality
  -> kNN Graph tren Causal features -> GraphSAGE
  -> Self-Attention -> Guided Cross-Attention (= C_vt)
  -> Causal Intervention (do-calculus) -> Adaptive DiffAttn
  -> Multi-task Heads (Task 1 + 2 + 3)

SCM (Structural Causal Model):
  D -> S_v, D -> S_t        (domain gay ra spurious)
  D ⊥ C_v, D ⊥ C_t          (domain khong anh huong causal)
  C_v, C_t -> C_vt -> y      (chi causal quyet dinh nhan)
  S_v, S_t ⊥/ y              (spurious khong lien quan nhan)

References:
  - GEDA (baseline): geda_model.py
  - CAMO (Ma et al., 2025): Adversarial disentanglement
  - CIRL (Lv et al., CVPR 2022): Causal representation learning
  - Ganin et al., JMLR 2016: Domain-adversarial training (GRL)
  - Sun et al., ICLR 2025: Causal representation from multimodal data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
from itertools import combinations


# ============================================================
# 1. ATTENTION MODULES (ke thua nguyen ban tu GEDA)
# ============================================================

class GraphSAGELayer(nn.Module):
    """
    True GraphSAGE aggregator logic:
    h_N = Aggregate(adj, x)
    h_out = ReLU(W * Concat(x, h_N))
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # GraphSAGE concatenates the node's own feature with the aggregated neighborhood feature
        self.proj = nn.Linear(in_dim * 2, out_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            h_N = torch.sparse.mm(adj, x)
        else:
            h_N = torch.matmul(adj, x)
        
        # Concat original features with aggregated neighbor features
        h_cat = torch.cat([x, h_N], dim=-1)
        return F.relu(self.proj(h_cat))


class SelfAttention(nn.Module):
    """
    Replaced SE-Net gating with a standard 2-layer MLP to act as a localized feature refiner,
    avoiding the misleading 'Self-Attention' terminology for single pooled vectors while
    conceptually retaining the step of refining individual modality features.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(dim // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return self.dropout(self.net(x)) + x


class GuidedCrossAttention(nn.Module):
    """
    Real Cross-Attention using MultiheadAttention between Image and Text pooled vectors.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # batch_first=True makes MHA expect (B, Seq, Dim)
        self.mha_I = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_T = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_I = nn.LayerNorm(dim)
        self.norm_T = nn.LayerNorm(dim)

    def forward(self, f_I: torch.Tensor, f_T: torch.Tensor) -> torch.Tensor:
        # Reshape to (B, Seq=1, D)
        q_I = f_I.unsqueeze(1)
        kv_T = f_T.unsqueeze(1)
        
        q_T = f_T.unsqueeze(1)
        kv_I = f_I.unsqueeze(1)
        
        # Image attends to Text
        out_I, _ = self.mha_I(query=q_I, key=kv_T, value=kv_T)
        out_I = self.norm_I(out_I.squeeze(1) + f_I) # Residual + Norm
        
        # Text attends to Image
        out_T, _ = self.mha_T(query=q_T, key=kv_I, value=kv_I)
        out_T = self.norm_T(out_T.squeeze(1) + f_T)
        
        return torch.cat([out_I, out_T], dim=-1)


class AdaptiveDiffAttention(nn.Module):
    """
    Token-wise Differential Attention over the [Image, Text] dual-token sequence.
    Applies the mathematical definition of Differential Attention over the 2 modalities.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % 2 == 0, "dim here is 2 * module_dim"
        self.single_dim = dim // 2
        self.num_heads = int(num_heads)
        self.head_dim = int(self.single_dim // self.num_heads)
        
        self.W_Q1 = nn.Linear(self.single_dim, self.single_dim)
        self.W_K1 = nn.Linear(self.single_dim, self.single_dim)
        self.W_Q2 = nn.Linear(self.single_dim, self.single_dim)
        self.W_K2 = nn.Linear(self.single_dim, self.single_dim)
        self.W_V = nn.Linear(self.single_dim, self.single_dim)
        self.W_O = nn.Linear(self.single_dim, self.single_dim)
        
        self.lambda_net = nn.Sequential(
            nn.Linear(self.single_dim * 2, self.single_dim // 4),
            nn.ReLU(),
            nn.Linear(self.single_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is concatenated: (B, 2D) -> Split into (B, 2, D)
        B = x.size(0)
        tokens = x.view(B, 2, self.single_dim) # (B, Seq=2, D)
        
        # Compute lambda from flat feature
        lam = self.lambda_net(x).view(B, 1, 1, 1) # (B, 1, 1, 1) to broadcast over heads and seq
        
        def compute_qkv(W_Q, W_K):
            Q = W_Q(tokens).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 2, d)
            K = W_K(tokens).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 2, d)
            return Q, K
            
        Q1, K1 = compute_qkv(self.W_Q1, self.W_K1)
        Q2, K2 = compute_qkv(self.W_Q2, self.W_K2)
        V = self.W_V(tokens).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn1 = F.softmax(torch.matmul(Q1, K1.transpose(-2, -1)) * self.scale, dim=-1) # (B, H, 2, 2)
        attn2 = F.softmax(torch.matmul(Q2, K2.transpose(-2, -1)) * self.scale, dim=-1)
        
        # Differential Attention Core
        diff_attn = F.relu(attn1 - lam * attn2)
        diff_attn = self.dropout(diff_attn)
        
        out = torch.matmul(diff_attn, V) # (B, H, 2, d)
        out = out.transpose(1, 2).contiguous().view(B, 2, self.single_dim) # (B, 2, D)
        out = self.dropout(self.W_O(out)) # (B, 2, D)
        
        # Resudual connection & flatten back to (B, 2D)
        out = out + tokens
        return out.view(B, self.single_dim * 2)


# ============================================================
# 2. CAUSAL DISENTANGLEMENT MODULE (MOI)
# ============================================================

class CausalDisentangler(nn.Module):
    """
    Phan ra features thanh Causal (C) va Spurious (S).

    Ly thuyet SCM:
      x_v = g_v(C_v, S_v)  =>  nguoc lai: (C_v, S_v) = psi_v(h_v)
      D -> S_v  nhung  D ⊥ C_v
      C_v -> y  (chi causal quyet dinh nhan)

    Kien truc: 2 nhanh MLP song song + Reconstruction decoder.
    Rang buoc:
      - Orthogonal loss: C ⊥ S (vuong goc, khong trung lap)
      - Adversarial loss: C phai domain-invariant (qua GRL)
      - Reconstruction: h ≈ dec(C, S) (dam bao du thong tin)
    """

    def __init__(self, input_dim: int = 512,
                 causal_dim: int = 256,
                 spurious_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()

        # Nhanh Causal: trich xuat dac trung bat bien mien
        self.causal_enc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, causal_dim),
            nn.LayerNorm(causal_dim),
        )

        # Nhanh Spurious: trich xuat dac trung domain-specific
        self.spurious_enc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, spurious_dim),
            nn.LayerNorm(spurious_dim),
        )

        # Reconstruction decoder: dam bao C + S chua du thong tin
        self.decoder = nn.Sequential(
            nn.Linear(causal_dim + spurious_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, input_dim) projected features tu Stage 1
        Returns:
            c: (B, causal_dim) causal features
            s: (B, spurious_dim) spurious features
            h_recon: (B, input_dim) reconstructed (cho L_recon)
        """
        c = self.causal_enc(h)
        s = self.spurious_enc(h)
        h_recon = self.decoder(torch.cat([c, s], dim=-1))
        return c, s, h_recon


# ============================================================
# 3. CONDITIONAL MMD + SPURIOUS DOMAIN CLASSIFIER (MOI)
# ============================================================

# Deleted duplicate mmd definition


class DomainClassifier(nn.Module):
    """
    Phan loai domain (disaster type) tu spurious features.
    S la domain-specific -> Classification accuracy phai cao.
    CrisisMMD co 7 disaster events => num_domains = 7.
    """

    def __init__(self, feature_dim: int = 256,
                 num_domains: int = 7,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) - S features
        Returns:
            domain_logits: (B, num_domains)
        """
        return self.net(features)


def compute_grl_lambda(epoch: int, max_epochs: int, warmup=20, max_lambda=0.15, gamma=10) -> float:
    """Giữ nguyên api name để không lỗi file trainer tạm thời (dù không dùng GRL nữa). 
    Tuy nhiên trainer vẫn dùng hàm này cho loss weights nếu cần."""
    if epoch < warmup:
        return 0.0
    p = (epoch - warmup) / max(max_epochs - warmup, 1)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


# ============================================================
# 4. CAUSAL INTERVENTION MODULE (MOI)
# ============================================================

class CausalIntervention(nn.Module):
    """
    Backdoor Adjustment tren Cross-Modal Causal Features.

    Ly thuyet:
      P(y | do(C_vt)) = sum_d P(y | C_vt, D=d) * P(D=d)

    Phuong phap:
    1. Duy tri Memory Bank luu moving average cua C_vt theo domain
    2. Khi training: tron C_vt hien tai voi centroids domain khac
    3. Khi inference: dung C_vt goc (khong intervention)

    Ly do dung Memory Bank:
    - Trong few-shot, moi forward pass co the thieu mau tu nhieu domain
    - Memory Bank tich luy dan qua cac epoch => on dinh hon
    """

    def __init__(self, feature_dim: int = 512,
                 num_domains: int = 7,
                 momentum: float = 0.9,
                 mix_ratio: float = 0.3):
        super().__init__()
        self.num_domains = num_domains
        self.momentum = momentum
        self.mix_ratio = mix_ratio

        # Memory bank: running mean per domain
        self.register_buffer('centroids', torch.zeros(num_domains, feature_dim))
        self.register_buffer('counts', torch.zeros(num_domains))
        self.register_buffer('initialized', torch.zeros(num_domains, dtype=torch.bool))

    @torch.no_grad()
    def update_memory(self, c_vt: torch.Tensor, domains: torch.Tensor):
        """Cap nhat centroids (chi khi training). Dung EMA."""
        # Issue 38: Exponential decay cho counts tranh tich luy vo tan
        self.counts *= 0.99
        for d in range(self.num_domains):
            mask = (domains == d)
            if mask.sum() == 0:
                continue

            batch_mean = c_vt[mask].mean(dim=0)

            if not self.initialized[d]:
                self.centroids[d] = batch_mean
                self.initialized[d] = True
            else:
                self.centroids[d] = (
                    self.momentum * self.centroids[d] +
                    (1 - self.momentum) * batch_mean
                )
            self.counts[d] += mask.sum()

    def forward(self, c_vt: torch.Tensor,
                domains: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backdoor Adjustment:
          do(C_vt) ≈ C_vt + mix_ratio * sum_d P(d) * (centroid_d - C_vt)
        
        Applies during BOTH training and inference to avoid distribution shift.
        """
        if self.training and domains is not None:
            # Update memory bank
            self.update_memory(c_vt.detach(), domains)

        # Compute domain weights P(D=d)
        total = self.counts.sum()
        if total == 0:
            return c_vt
            
        # Hardest negative domain intervention (Contrastive):
        # Find the domain centroid that is furthest from each sample
        dists = torch.cdist(c_vt, self.centroids) # (B, num_domains)
        
        # Ignore uninitialized domains
        invalid_mask = ~self.initialized
        dists[:, invalid_mask] = -1.0
        
        # Get hardest negative centroid for each sample
        furthest_idx = dists.argmax(dim=1)
        hardest_negatives = self.centroids[furthest_idx]
        
        # Contrastive Intervention: push/pull towards hardest negative to make invariant
        c_vt_do = c_vt + self.mix_ratio * (hardest_negatives - c_vt)
        return c_vt_do


# ============================================================
# 5. CAUSAL CRISIS MODEL (Tong hop)
# ============================================================

class CausalCrisisModel(nn.Module):
    """
    CausalCrisis = GEDA + Causal Disentanglement + do-calculus Intervention.

    Ke thua toan bo kien truc GEDA, bo sung:
    - Stage 2: CausalDisentangler (per modality)
    - Stage 3c: CausalIntervention (do-calculus)
    - DomainClassifier (adversarial, voi GRL)
    - Graph tren Causal features

    Pipeline:
    proj -> disentangle -> GNN(causal) -> SelfAttn -> GCA -> Intervene -> DiffAttn -> Heads

    Toggle switches cho ablation study:
    - use_causal: True/False  (bat/tat Stage 2 disentanglement)
    - use_intervention: True/False  (bat/tat Stage 3c do-calculus)
    - use_graph: True/False  (bat/tat GNN)
    - use_attention: True/False  (bat/tat Attention stack)
    - use_mtl: True/False  (bat/tat Multi-task learning)
    """

    def __init__(
        self,
        img_dim: int = 256,
        txt_dim: int = 256,
        hidden_dim: int = 512,
        causal_dim: int = 256,
        spurious_dim: int = 256,
        num_domains: int = 7,
        num_classes_task1: int = 2,
        num_classes_task2: int = 6,
        num_classes_task3: int = 3,
        dropout: float = 0.3,
        use_graph: bool = True,
        use_attention: bool = True,
        use_mtl: bool = True,
        use_causal: bool = True,
        use_intervention: bool = True,
        intervention_momentum: float = 0.9,
        intervention_mix: float = 0.3,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_attention = use_attention
        self.use_mtl = use_mtl
        self.use_causal = use_causal
        self.use_intervention = use_intervention

        # ── Stage 1: Feature Projection (GEDA giu nguyen) ──
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

        # ── Stage 2: Causal Disentanglement (MOI) ──
        if use_causal:
            self.disentangle_img = CausalDisentangler(
                hidden_dim, causal_dim, spurious_dim, dropout
            )
            self.disentangle_txt = CausalDisentangler(
                hidden_dim, causal_dim, spurious_dim, dropout
            )
            # Issue 34: Share weights giua cac modalities tiet kiem param, dong bo gradient
            self.domain_cls_spurious = DomainClassifier(spurious_dim, num_domains)
            gnn_input_dim = causal_dim
        else:
            # Fallback: nhu GEDA goc
            gnn_input_dim = hidden_dim

        # ── Stage 3a: GNN layers (GEDA style, doi dim neu causal) ──
        if use_graph:
            self.gnn_img_1 = GraphSAGELayer(gnn_input_dim, gnn_input_dim)
            self.gnn_img_2 = GraphSAGELayer(gnn_input_dim, gnn_input_dim)
            self.gnn_txt_1 = GraphSAGELayer(gnn_input_dim, gnn_input_dim)
            self.gnn_txt_2 = GraphSAGELayer(gnn_input_dim, gnn_input_dim)
            self.gnn_norm_img = nn.LayerNorm(gnn_input_dim)
            self.gnn_norm_txt = nn.LayerNorm(gnn_input_dim)

        # ── Stage 3b: Attention Stack (GEDA style, doi dim) ──
        if use_attention:
            self.self_attn_img = SelfAttention(gnn_input_dim, dropout)
            self.self_attn_txt = SelfAttention(gnn_input_dim, dropout)
            self.norm_img = nn.LayerNorm(gnn_input_dim)
            self.norm_txt = nn.LayerNorm(gnn_input_dim)
            
            self.gca = GuidedCrossAttention(gnn_input_dim, num_heads=4, dropout=dropout)
            self.norm_gca = nn.LayerNorm(gnn_input_dim * 2)
            
            # DiffAttn can act directly on concatenated features
            self.diff_attn = AdaptiveDiffAttention(gnn_input_dim * 2, num_heads=4, dropout=dropout)
            self.norm_diff = nn.LayerNorm(gnn_input_dim * 2)

        # ── Stage 3c: Causal Intervention (MOI) ──
        # Guard: intervention chi co y nghia khi co attention (C_vt)
        if self.use_intervention:
            print("  [Init] Enabled CausalIntervention")
            self.causal_intervention = CausalIntervention(
                feature_dim=gnn_input_dim * 2, # GCA output dim
                num_domains=num_domains,
                momentum=intervention_momentum,
                mix_ratio=intervention_mix,
            )


        # Re-calculate classifier_dim based on final feature size
        gnn_output_dim = gnn_input_dim
        if use_attention:
            # GCA output is 2 * gnn_input_dim
            classifier_dim = gnn_output_dim * 2
        else:
            # Concatenation of img and txt features
            classifier_dim = gnn_output_dim * 2

        # GCA Dim Check:
        if self.use_attention:
            assert classifier_dim == (causal_dim * 2 if use_causal else hidden_dim * 2), "Classifier dim mismatch with Guided Cross Attention output"


        # ── Stage 4: Classification Heads (GEDA giu nguyen) ──
        self.head_task1 = nn.Sequential(
            nn.Linear(classifier_dim, gnn_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_input_dim, num_classes_task1),
        )

        if use_mtl:
            self.head_task2 = nn.Sequential(
                nn.Linear(classifier_dim, gnn_input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gnn_input_dim, num_classes_task2),
            )
            self.head_task3 = nn.Sequential(
                nn.Linear(classifier_dim, gnn_input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gnn_input_dim, num_classes_task3),
            )

        # Issue 32: Proper weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def graph_propagate(self, x: torch.Tensor, adj: Optional[torch.Tensor],
                        layer1: nn.Module, layer2: nn.Module,
                        norm: nn.LayerNorm) -> torch.Tensor:
        """True GraphSAGE 2-layer propagation."""
        if adj is None:
            # Fallback when no graph is used => behaves roughly like identity (handled earlier)
            return x
        
        h1 = layer1(x, adj)
        h2 = layer2(h1, adj)
            
        h2 = norm(h2 + h1)
        return h2

    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        task: str = "task1",
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass CausalCrisis.

        Args:
            img_feat: (B, img_dim) CLIP image features (PCA-reduced)
            txt_feat: (B, txt_dim) CLIP text features (PCA-reduced)
            adj: (B, B) adjacency matrix (tren causal features neu use_causal)
            domain_labels: (B,) integer domain ID (0..6 cho 7 disasters)
            task: "task1", "task2", "task3", or "all"
            grl_lambda: GRL strength (0=no reversal, 1=full reversal)

        Returns:
            Dict chua:
              - "task1/2/3": logits phan loai
              - "domain_cv", "domain_ct": domain predictions (cho L_adv)
              - "domain_sv", "domain_st": domain predictions (spurious, ko GRL)
              - "c_v", "s_v", "c_t", "s_t": disentangled features (cho L_orth)
              - "h_img_recon", "h_txt_recon": reconstruction (cho L_recon)
              - "h_img_orig", "h_txt_orig": original projected (cho L_recon)
              - "c_vt_original": pre-intervention C_vt
              - "c_vt_intervened": post-intervention C_vt (cho L_int)
        """
        outputs = {}

        # De tranh test domain leakage: Test CausalIntervention phai su dung Centroid tu Test thay vi Update tu Ground Truth Test Domain!
        if not self.training:
            domain_labels = None

        # ── Stage 1: Feature Projection ──
        h_img = self.proj_img(img_feat)   # (B, hidden_dim=512)
        h_txt = self.proj_txt(txt_feat)   # (B, hidden_dim=512)

        # ── Stage 2: Causal Disentanglement ──
        if self.use_causal:
            c_v, s_v, h_img_recon = self.disentangle_img(h_img)   # (B, 256) each
            c_t, s_t, h_txt_recon = self.disentangle_txt(h_txt)   # (B, 256) each

            # Domain predictions (cho spurious classifier)
            outputs["domain_sv"] = self.domain_cls_spurious(s_v)
            outputs["domain_st"] = self.domain_cls_spurious(s_t)

            # Luu cho loss computation
            outputs["c_v"] = c_v
            outputs["s_v"] = s_v
            outputs["c_t"] = c_t
            outputs["s_t"] = s_t
            outputs["h_img_recon"] = h_img_recon
            outputs["h_txt_recon"] = h_txt_recon
            outputs["h_img_orig"] = h_img   # KHONG DETACH (Regularize projection layer tu L_recon)
            outputs["h_txt_orig"] = h_txt   # KHONG DETACH

            # Chi dua Causal vao pipeline tiep theo
            feat_img = c_v   # (B, causal_dim=256)
            feat_txt = c_t   # (B, causal_dim=256)
        else:
            feat_img = h_img  # (B, hidden_dim=512) — nhu GEDA goc
            feat_txt = h_txt

        # ── Stage 3a: GNN Propagation (tren causal features) ──
        if self.use_graph and adj is not None:
            feat_img = self.graph_propagate(
                feat_img, adj,
                self.gnn_img_1, self.gnn_img_2, self.gnn_norm_img
            )
            feat_txt = self.graph_propagate(
                feat_txt, adj,
                self.gnn_txt_1, self.gnn_txt_2, self.gnn_norm_txt
            )

        # ── Stage 3b: Attention Stack ──
        if self.use_attention:
            # SelfAttention already includes residual connection internally
            feat_img_attn = self.self_attn_img(feat_img)
            feat_txt_attn = self.self_attn_txt(feat_txt)
            feat_img = self.norm_img(feat_img_attn)
            feat_txt = self.norm_txt(feat_txt_attn)
            
            c_vt_attn = self.gca(feat_img, feat_txt)  # (B, causal_dim*2 = 512)
            c_vt = self.norm_gca(torch.cat([feat_img, feat_txt], dim=-1) + c_vt_attn)
            
            outputs["c_vt_original"] = c_vt

            # ── Stage 3c: Causal Intervention (do-calculus) ──
            # Intervention should always run (train and test) to prevent distribution shift
            if (self.use_causal and self.use_intervention
                    and self.use_attention and hasattr(self, 'causal_intervention')):
                c_vt_do = self.causal_intervention(c_vt, domain_labels)
                outputs["c_vt_intervened"] = c_vt_do
            else:
                c_vt_do = c_vt

            # ── Stage 3d: DiffAttn (implicit causal intervention) ──
            z_attn = self.diff_attn(c_vt_do)  # (B, 512)
            z = self.norm_diff(c_vt_do + z_attn)
        else:
            z = torch.cat([feat_img, feat_txt], dim=-1)

        # ── Stage 4: Classification ──
        # Issue 35: check constraint with hasattr
        if task in ("task1", "all") and hasattr(self, 'head_task1'):
            outputs["task1"] = self.head_task1(z)
        if self.use_mtl and task in ("task2", "all") and hasattr(self, 'head_task2'):
            outputs["task2"] = self.head_task2(z)
        if self.use_mtl and task in ("task3", "all") and hasattr(self, 'head_task3'):
            outputs["task3"] = self.head_task3(z)

        return outputs


# ============================================================
# 6. LOSS FUNCTION: CausalCrisisLoss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets, smoothing=0.0):
        # Apply label smoothing manually
        if smoothing > 0.0:
            n_classes = inputs.size(1)
            # targets is 1d tensor of class indices
            targets_one_hot = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1), 1)
            targets_smooth = targets_one_hot * (1 - smoothing) + (smoothing / n_classes)
            
            p = torch.softmax(inputs, dim=1)
            log_p = torch.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(targets_smooth * log_p, dim=1)
            
            # pt for focal loss is based on the original target class probability
            pt = p.gather(1, targets.unsqueeze(-1)).squeeze()
            
            loss = ce_loss * ((1 - pt) ** self.gamma)
            
            if self.weight is not None:
                batch_weights = self.weight[targets]
                loss = loss * batch_weights
                
            return loss.mean() # Return mean loss for batch
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()


def mmd_rbf(x, y, gammas=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """
    Computes the Multi-Kernel RBF MMD (Maximum Mean Discrepancy) between two feature sets.
    Args:
        x (torch.Tensor): Features from domain X (N, D)
        y (torch.Tensor): Features from domain Y (M, D)
        gammas (list): Kernel parameters for RBF.
    Returns:
        torch.Tensor: MMD value.
    """
    x_sq = (x ** 2).sum(dim=-1).unsqueeze(1) # (N, 1)
    y_sq = (y ** 2).sum(dim=-1).unsqueeze(0) # (1, M)
    
    dxx = x_sq + x_sq.transpose(0, 1) - 2. * torch.matmul(x, x.transpose(0, 1))
    dyy = y_sq.transpose(0, 1) + y_sq - 2. * torch.matmul(y, y.transpose(0, 1))
    dxy = x_sq + y_sq - 2. * torch.matmul(x, y.transpose(0, 1))

    total_mmd = 0.0
    for gamma in gammas:
        kxx = torch.exp(-gamma * dxx)
        kyy = torch.exp(-gamma * dyy)
        kxy = torch.exp(-gamma * dxy)
        total_mmd += kxx.mean() + kyy.mean() - 2. * kxy.mean()

    return total_mmd


def conditional_mmd_loss(features: torch.Tensor, domains: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes Conditional MMD loss.
    The goal is to make causal features domain-invariant *within* each task label.
    Args:
        features (torch.Tensor): Causal features (B, D)
        domains (torch.Tensor): Domain labels (B,)
        labels (torch.Tensor): Task labels (B,)
    Returns:
        torch.Tensor: Conditional MMD loss.
    """
    if features.numel() == 0 or domains.numel() == 0 or labels.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    unique_labels = labels.unique()
    unique_domains = domains.unique()

    loss = 0.0
    count = 0

    for label in unique_labels:
        label_mask = (labels == label)
        if label_mask.sum() == 0:
            continue

        # Collect features for this label across all domains
        label_features = features[label_mask]
        label_domains = domains[label_mask]

        # Compare features between all pairs of domains for this label
        for i, dom1 in enumerate(unique_domains):
            for j, dom2 in enumerate(unique_domains):
                if i >= j: # Only compute unique pairs (including self-comparison for variance)
                    continue

                mask1 = (label_domains == dom1)
                mask2 = (label_domains == dom2)

                if mask1.sum() > 1 and mask2.sum() > 1: # Need at least 2 samples per domain to compute MMD
                    x = label_features[mask1]
                    y = label_features[mask2]
                    loss += mmd_rbf(x, y)
                    count += 1
    
    return loss / max(1, count) # Average MMD over all valid pairs


class CausalCrisisLoss(nn.Module):
    """
    L = L_cls + alpha_adv * L_adv + alpha_orth * L_orth
      + alpha_recon * L_recon + alpha_int * L_int

    Moi thanh phan tuong ung mot rang buoc SCM:
      L_cls   : P(y | C)       -> phan loai tu causal features
      L_adv   : D ⊥ C          -> ep causal domain-invariant
      L_orth  : C ⊥ S          -> vuong goc causal/spurious
      L_recon : h ≈ dec(C, S)  -> dam bao thong tin day du
      L_int   : do(C_vt) -> y  -> prediction bat bien khi intervention
    """

    def __init__(self, recon_w=1.0, orth_w=0.1, adv_w=0.1, int_w=0.1, label_smoothing=0.1,
                 gamma: float = 2.0,
                 task_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)):
        super().__init__()
        self.alpha_recon = recon_w
        self.alpha_orth = orth_w
        self.alpha_adv = adv_w
        self.alpha_int = int_w

        self.focal = FocalLoss(gamma=gamma)
        self.recon = nn.MSELoss(reduction='none')
        self.ce_adv = nn.CrossEntropyLoss(reduction='none')
        # Thuc chat kl_div nen co the dung luon KLDivLoss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.label_smoothing = label_smoothing
        self.task_weights = task_weights

    def set_phase(self, phase: int):
        """Thay doi trong so cac ham loss theo phase. Increased alpha_recon to prevent info loss."""
        # Note: L_recon uses MSE over 512 dims, produces ~0.005. Needs large alpha!
        if phase == 1:    # Epoch 0-50: focus classification
            self.alpha_adv = 0.05; self.alpha_orth = 0.01
            self.alpha_recon = 1.0; self.alpha_int = 0.0
        elif phase == 2:  # Epoch 50-120: them causal
            self.alpha_adv = 0.1; self.alpha_orth = 0.05
            self.alpha_recon = 2.0; self.alpha_int = 0.1
        else:             # Phase 3: fine-tune
            self.alpha_adv = 0.15; self.alpha_orth = 0.1
            self.alpha_recon = 2.0; self.alpha_int = 0.2

    def orthogonal_loss(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """HSIC: kiem tra statistical independence, khong chi linear (Issue 36).
        Efficient O(N^2) trace computation without explicitly forming H."""
        n = c.shape[0]
        if n <= 1:
            return torch.tensor(0.0, device=c.device)
            
        # Issue 37: L2 Normalize features to prevent HSIC trace explosion
        c = F.normalize(c, dim=-1)
        s = F.normalize(s, dim=-1)
        
        K = c @ c.T
        L = s @ s.T
        
        # Efficient centering (K H L H trace equivalent)
        Kc = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
        Lc = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()
        
        return (Kc * Lc).sum() / ((n-1)**2)

    def reconstruction_loss(self, original: torch.Tensor,
                            reconstructed: torch.Tensor) -> torch.Tensor:
        """L_recon: h ≈ decoder(C, S)."""
        return F.mse_loss(reconstructed, original)

    def intervention_consistency_loss(
        self,
        logits_original: torch.Tensor,
        logits_intervened: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_int: P(y|C_vt) ≈ P(y|do(C_vt)).
        KL Divergence distills unbiased intervened features prediction into the original features.
        """
        p = F.softmax(logits_intervened.detach(), dim=-1)
        q = F.log_softmax(logits_original, dim=-1)
        return F.kl_div(q, p, reduction='batchmean')

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        domain_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Tinh tong loss.

        Args:
            outputs: dict tu CausalCrisisModel.forward()
            targets: dict {"task1": (B,), ...}
            domain_labels: (B,) domain integer labels
            mask: (B,) bool tensor — chi tinh loss tren labeled samples

        Returns:
            total_loss, loss_dict
        """
        # Tim device
        device = None
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        total = torch.tensor(0.0, device=device)
        losses = {}

        # ── L_cls: Focal Loss cho classification ──
        task_lambdas = {
            "task1": self.task_weights[0],
            "task2": self.task_weights[1],
            "task3": self.task_weights[2],
        }
        for t in ["task1", "task2", "task3"]:
            if t in outputs and t in targets:
                logits = outputs[t]
                labels = targets[t]
                if mask is not None:
                    logits = logits[mask]
                    labels = labels[mask]
                if len(labels) > 0:
                    l_cls = self.focal(logits, labels.to(logits.device), smoothing=self.label_smoothing)
                    total = total + task_lambdas[t] * l_cls
                    losses[f"cls_{t}"] = l_cls.item()

        # ── L_adv: Adversarial/Conditional MMD domain loss ──
        if "c_v" in outputs and domain_labels is not None and "task1" in targets:
            dom = domain_labels
            v_sv = outputs.get("domain_sv", None)
            v_st = outputs.get("domain_st", None)
            c_v = outputs["c_v"]
            c_t = outputs["c_t"]
            task1_lbls = targets["task1"]

            # CRITICAL: Prevent Data Leakage and Mismatched Shapes
            # Only calculate MMD and Adversarial losses on Labeled samples!
            if mask is not None:
                dom = dom[mask]
                task1_lbls = task1_lbls[mask]
                c_v = c_v[mask]
                c_t = c_t[mask]
                if v_sv is not None:
                    v_sv = v_sv[mask]
                    v_st = v_st[mask]

            if len(dom) > 0 and v_sv is not None:
                # 1. Spurious phai giu duoc domain thong tin => CrossEntropy pass
                l_adv_s = F.cross_entropy(v_sv, dom) + F.cross_entropy(v_st, dom)
                
                # 2. Causal phai giong nhau giua cac domain trong CUNG 1 TASK => Conditional MMD
                l_mmd_v = conditional_mmd_loss(c_v, dom, task1_lbls)
                l_mmd_t = conditional_mmd_loss(c_t, dom, task1_lbls)
                l_adv_c = l_mmd_v + l_mmd_t
                
                total = total + self.alpha_adv * (l_adv_s + l_adv_c)
                losses["adv_s"] = l_adv_s.item()
                losses["adv_c"] = l_adv_c.item()
                losses["adv_causal"] = l_adv_c.item()
                losses["adv_spurious"] = l_adv_s.item()

        # ── L_orth: Orthogonal regularization ──
        if "c_v" in outputs:
            c_v, s_v = outputs["c_v"], outputs["s_v"]
            c_t, s_t = outputs["c_t"], outputs["s_t"]
            
            if len(c_v) > 0:
                l_orth = self.orthogonal_loss(c_v, s_v) + self.orthogonal_loss(c_t, s_t)
                total = total + self.alpha_orth * l_orth
                losses["orth"] = l_orth.item()

        # ── L_recon: Reconstruction ──
        if "h_img_recon" in outputs:
            h_ir, h_io = outputs["h_img_recon"], outputs["h_img_orig"]
            h_tr, h_to = outputs["h_txt_recon"], outputs["h_txt_orig"]
            
            if len(h_ir) > 0:
                l_recon = (self.reconstruction_loss(h_io, h_ir).mean()
                           + self.reconstruction_loss(h_to, h_tr).mean())
                total = total + self.alpha_recon * l_recon
                losses["recon"] = l_recon.item()

        # ── L_int: Intervention consistency ──
        # NOTE: L_int duoc tinh o Trainer level vi can forward 2 lan
        # (1 lan voi C_vt goc, 1 lan voi C_vt_do)
        # Trainer se goi intervention_consistency_loss truc tiep

        losses["total"] = total.item()
        return total, losses


# ============================================================
# 7. ABLATION CONFIGS (mo rong tu GEDA)
# ============================================================

ABLATION_CONFIGS = {
    # --- GEDA baselines (tuong thich nguoc) ---
    "geda_baseline": {
        "use_graph": True, "use_attention": True, "use_mtl": True,
        "use_causal": False, "use_intervention": False,
    },
    "geda_no_graph": {
        "use_graph": False, "use_attention": True, "use_mtl": True,
        "use_causal": False, "use_intervention": False,
    },

    # --- CausalCrisis variants ---
    "causal_full": {
        "use_graph": True, "use_attention": True, "use_mtl": True,
        "use_causal": True, "use_intervention": True,
    },
    "causal_no_intervention": {
        "use_graph": True, "use_attention": True, "use_mtl": True,
        "use_causal": True, "use_intervention": False,
    },
    "causal_no_graph": {
        "use_graph": False, "use_attention": True, "use_mtl": True,
        "use_causal": True, "use_intervention": True,
    },
    "causal_no_diffattn": {
        "use_graph": True, "use_attention": False, "use_mtl": True,
        "use_causal": True, "use_intervention": True,
    },
    "causal_no_disentangle": {
        "use_graph": True, "use_attention": True, "use_mtl": True,
        "use_causal": False, "use_intervention": False,
    },
}


def create_causal_variant(variant: str, **kwargs) -> CausalCrisisModel:
    """Tao model variant cho ablation study."""
    if variant not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from {list(ABLATION_CONFIGS.keys())}"
        )
    config = {**ABLATION_CONFIGS[variant], **kwargs}
    return CausalCrisisModel(**config)


# ============================================================
# 8. UTILITY
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


# ============================================================
# 9. SANITY TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CausalCrisis Model Architecture Test")
    print("=" * 60)

    B = 16   # batch size
    D = 256  # PCA dim
    NUM_DOMAINS = 7

    # --- Test 1: Full CausalCrisis ---
    print("\n[Test 1] CausalCrisis FULL (causal + intervention)")
    model = CausalCrisisModel(
        img_dim=D, txt_dim=D, hidden_dim=512,
        causal_dim=256, spurious_dim=256, num_domains=NUM_DOMAINS,
        use_causal=True, use_intervention=True,
    )
    info = model_summary(model)
    print(f"  Total params:     {info['total_params']:,}")
    print(f"  Trainable params: {info['trainable_params']:,}")

    img = torch.randn(B, D)
    txt = torch.randn(B, D)
    adj = torch.eye(B)
    domains = torch.randint(0, NUM_DOMAINS, (B,))

    model.train()
    outputs = model(img, txt, adj, domains, task="all", grl_lambda=0.5)

    print("\n  Output keys:", list(outputs.keys()))
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"    {key}: {val.shape}")

    # --- Test 2: Loss computation ---
    print("\n[Test 2] CausalCrisisLoss")
    criterion = CausalCrisisLoss()

    targets = {
        "task1": torch.randint(0, 2, (B,)),
        "task2": torch.randint(0, 6, (B,)),
        "task3": torch.randint(0, 3, (B,)),
    }

    total_loss, loss_dict = criterion(outputs, targets, domain_labels=domains)
    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

    # --- Test 3: Backward pass ---
    print("\n[Test 3] Backward pass")
    total_loss.backward()
    print("  Backward OK!")

    # --- Test 4: GRL lambda schedule ---
    print("\n[Test 4] GRL lambda schedule")
    for epoch in [0, 50, 100, 200, 400, 500]:
        lam = compute_grl_lambda(epoch, 500)
        print(f"  Epoch {epoch:4d}: lambda = {lam:.4f}")

    # --- Test 5: GEDA backward-compatible ---
    print("\n[Test 5] GEDA baseline (no causal)")
    model_geda = CausalCrisisModel(
        img_dim=D, txt_dim=D, hidden_dim=512,
        use_causal=False, use_intervention=False,
    )
    out_geda = model_geda(img, txt, adj, task="task1")
    print(f"  Output keys: {list(out_geda.keys())}")
    print(f"  task1 logits: {out_geda['task1'].shape}")

    # --- Test 6: All ablation variants ---
    print("\n[Test 6] Ablation variants:")
    for name in ABLATION_CONFIGS:
        m = create_causal_variant(name, img_dim=D, txt_dim=D, hidden_dim=512)
        s = model_summary(m)
        print(f"  {name:30s}: {s['trainable_params']:>10,} params")

    print("\n" + "=" * 60)
    print("  [OK] All CausalCrisis tests passed!")
    print("=" * 60)
