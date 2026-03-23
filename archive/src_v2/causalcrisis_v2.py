import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CausalGNNModule(nn.Module):
    """
    Stage 3B: Graph Neural Network
    Nối các Causal Vector kết tinh bằng đồ thị k-NN và message passing.
    """
    def __init__(self, in_dim: int, hidden_dim: int, dropout=0.3):
        super().__init__()
        self.conv1 = GraphSAGELayer(in_dim, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.conv1(x, adj)
        h = self.dropout(h)
        h = self.conv2(h, adj)
        return self.norm(x + h) # Residual connection

class GuidedFusion(nn.Module):
    """
    Stage 2A: Guided Self-Attention + Cross-Attention
    """
    def __init__(self, img_dim: int, txt_dim: int, hidden_dim: int):
        super().__init__()
        self.self_attn_img = nn.MultiheadAttention(embed_dim=img_dim, num_heads=4, batch_first=True)
        self.self_attn_txt = nn.MultiheadAttention(embed_dim=txt_dim, num_heads=4, batch_first=True)

        self.proj_I = nn.Linear(img_dim, hidden_dim)
        self.mask_I = nn.Linear(img_dim, hidden_dim)

        self.proj_T = nn.Linear(txt_dim, hidden_dim)
        self.mask_T = nn.Linear(txt_dim, hidden_dim)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_I: torch.Tensor, f_T: torch.Tensor):
        I = f_I.unsqueeze(1)  # [B, 1, D_img]
        T = f_T.unsqueeze(1)  # [B, 1, D_txt]

        I_sa, _ = self.self_attn_img(I, I, I)
        T_sa, _ = self.self_attn_txt(T, T, T)

        I_sa = I_sa.squeeze(1)
        T_sa = T_sa.squeeze(1)

        z_I = self.act(self.proj_I(I_sa))
        alpha_I = self.sigmoid(self.mask_I(I_sa))

        z_T = self.act(self.proj_T(T_sa))
        alpha_T = self.sigmoid(self.mask_T(T_sa))

        # Vision-guided text & Text-guided vision
        v_guided = alpha_T * z_I
        t_guided = alpha_I * z_T

        z = torch.cat([v_guided, t_guided], dim=-1)
        return z


class DifferentialAttention(nn.Module):
    """
    Stage 2B: Differential Attention
    """
    def __init__(self, dim_fusion: int, dim_qk: int):
        super().__init__()
        self.W_Q = nn.Linear(dim_fusion, 2 * dim_qk)
        self.W_K = nn.Linear(dim_fusion, 2 * dim_qk)
        self.W_V = nn.Linear(dim_fusion, dim_qk)
        self.lambda_param = nn.Parameter(torch.tensor(0.5))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z: torch.Tensor):
        Q = self.W_Q(z)
        K = self.W_K(z)
        V = self.W_V(z)

        Dq = V.size(-1)
        Q1, Q2 = Q[:, :Dq], Q[:, Dq:]
        K1, K2 = K[:, :Dq], K[:, Dq:]

        Q1_ = Q1.unsqueeze(1)
        K1_ = K1.unsqueeze(1)
        Q2_ = Q2.unsqueeze(1)
        K2_ = K2.unsqueeze(1)
        V_  = V.unsqueeze(1)

        attn1 = self.softmax(torch.bmm(Q1_, K1_.transpose(1, 2)) / (Dq ** 0.5))
        attn2 = self.softmax(torch.bmm(Q2_, K2_.transpose(1, 2)) / (Dq ** 0.5))

        out = (attn1 - self.lambda_param * attn2) @ V_
        out = out.squeeze(1)
        return out



class CausalDisentanglerV2(nn.Module):
    """
    Stage 3A: Causal Disentanglement via MLP
    Nhận input là unified representations (z_img_general + z_txt_general)
    Phân rã thành X_c (Causal) và X_s (Spurious)
    """
    def __init__(self, in_dim: int, causal_dim: int, spurious_dim: int, dropout=0.3):
        super().__init__()
        self.causal_enc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, causal_dim)
        )
        self.spurious_enc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, spurious_dim)
        )

    def forward(self, z_unified: torch.Tensor):
        xc = self.causal_enc(z_unified)
        xs = self.spurious_enc(z_unified)
        return xc, xs


class DomainClassifier(nn.Module):
    """Dùng cho GRL (Gradient Reversal Layer) để ép X_c độc lập với Domain"""
    def __init__(self, in_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_dim, in_dim // 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(in_dim // 2, num_domains))
        )
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

class EdgeHeterophilyScorer(nn.Module):
    """
    Phase 3c: Học mức độ Heterophily (lệch pha) giữa 2 node.
    Đầu vào: x_i, x_j (causal features) và p_i, p_j (xác suất/pseudo-labels).
    Đầu ra: Trọng số (0.0 đến 1.0). 1.0 (Homophilic/Tốt), 0.0 (Heterophilic/Rác).
    """
    def __init__(self, in_dim: int, num_classes: int, dropout=0.3):
        super().__init__()
        # Ghép chênh lệch feature và tương tác phân phối nhãn
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1)
        )
        
    def forward(self, x_i, x_j, p_i, p_j):
        feat_diff = torch.abs(x_i - x_j) # Dấu hiệu khác biệt causal
        prob_sim = p_i * p_j             # Mức độ chung nhãn
        concat_feat = torch.cat([feat_diff, prob_sim], dim=-1)
        
        score = torch.sigmoid(self.net(concat_feat))
        return score.squeeze(-1)

# ==========================================================
# CAUSAL CRISIS V2 MODEL MASTER
# ==========================================================
class CausalCrisisV2Model(nn.Module):
    """
    CausalCrisis v2 - Phase 1 & 2
    Bao gồm: Modality Disentanglement -> Causal Disentanglement -> [GNN] -> Classification
    """
    def __init__(self, 
                 img_dim: int = 256, # Nếu có PCA thì là 256, nếu RAW CLIP thì 1024
                 txt_dim: int = 256, # Nếu có PCA thì là 256, nếu RAW CLIP thì 768
                 hidden_dim: int = 256, 
                 causal_dim: int = 256, 
                 spurious_dim: int = 256, 
                 num_domains: int = 7, 
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # Stage 2: DiffFusionEncoder
        self.guided_fusion = GuidedFusion(img_dim, txt_dim, hidden_dim)
        fusion_dim = hidden_dim * 2
        self.diff_attn = DifferentialAttention(dim_fusion=fusion_dim, dim_qk=fusion_dim)
        
        # Stage 3A: Causal Disentangle
        self.causal_disentangle = CausalDisentanglerV2(fusion_dim, causal_dim, spurious_dim, dropout)
        
        # Stage 3B: GNN Module Phase 2
        self.gnn = CausalGNNModule(causal_dim, causal_dim, dropout)
        
        # Stage 3C: Heterophily Scorer (Phase 3c)
        self.heterophily_scorer = EdgeHeterophilyScorer(causal_dim, num_classes, dropout)
        
        # GRL Discriminator
        self.domain_classifier = DomainClassifier(causal_dim, num_domains)
        
        self.classifier = nn.Sequential(
            nn.Linear(causal_dim, causal_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(causal_dim // 2, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, adj: torch.Tensor = None, backdoor_xs: torch.Tensor = None):
        outputs = {}
        
        # Stage 2: DiffFusionEncoder (GuidedFusion + DiffAttn)
        z = self.guided_fusion(img_feat, txt_feat)
        z_prime = self.diff_attn(z)
        
        outputs["z_unified"] = z_prime
        
        # Stage 3A: Causal Disentangle
        xc, xs = self.causal_disentangle(z_prime)
        
        outputs["xc"] = xc
        outputs["xs"] = xs
        
        # Domain logits cho Loss Adversarial (L_disc)
        outputs["domain_logits"] = self.domain_classifier(xc)
        
        # Stage 3B: Causal Graph Neural Network (Phase 2)
        if adj is not None:
            xc = self.gnn(xc, adj)
            outputs["xc_graph"] = xc
        else:
            outputs["xc_graph"] = xc # Keep for GRL consistency
        
        # Stage 4/5: Classification & Backdoor Adjustment (Merged GNN + BA)
        if backdoor_xs is not None:
            # Inference: Backdoor Intervention (Expectation over M samples from Bank)
            M = backdoor_xs.shape[1]
            xc_expand = outputs["xc_graph"].unsqueeze(1).expand(-1, M, -1) # (batch, M, causal_dim)
            
            # Dung hòa X_gnn và X_s ở cấp độ Vector Space (Addition)
            combined = xc_expand + backdoor_xs 
            logits_M = self.classifier(combined) # (batch, M, num_classes)
            
            # P(Y|do(X)) = E_{Xs}[ P(Y|Xc,Xs) ]
            probs_M = torch.softmax(logits_M, dim=-1)
            expected_probs = probs_M.mean(dim=1)
            outputs["logits_ba"] = torch.log(expected_probs + 1e-8)
        else:
            # Training: Use current batch's Xs to learn P(Y | Xc, Xs)
            xs_detached = xs.detach()
            combined = outputs["xc_graph"] + xs_detached # Dung hòa GNN và Xs
            outputs["logits_ba"] = self.classifier(combined)
        
        # GNN Only logits (cho Loss Phase 2 không có BA)
        outputs["logits_gnn"] = self.classifier(outputs["xc_graph"])
        
        # Phase 1 Standard Classification (P(Y | Xc))
        outputs["logits"] = self.classifier(xc)
        
        return outputs
