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

class ModalityProjector(nn.Module):
    """
    Stage 2: Modality Disentanglement
    Tách đặc trưng của mỗi modality thành:
    - z_general (shared semantics)
    - z_specific (modality-unique semantics)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout=0.3):
        super().__init__()
        self.general_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim)
        )
        self.specific_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x: torch.Tensor):
        z_general = self.general_proj(x)
        z_specific = self.specific_proj(x)
        return z_general, z_specific


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
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_domains)
        )
        
    def forward(self, x: torch.Tensor):
        return self.net(x)

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
        
        # Stage 2
        self.img_proj = ModalityProjector(img_dim, hidden_dim, dropout)
        self.txt_proj = ModalityProjector(txt_dim, hidden_dim, dropout)
        
        # Stage 3A
        unified_dim = hidden_dim * 2
        self.causal_disentangle = CausalDisentanglerV2(unified_dim, causal_dim, spurious_dim, dropout)
        
        # Stage 3B: GNN Module Phase 2
        self.gnn = CausalGNNModule(causal_dim, causal_dim, dropout)
        
        # GRL Discriminator
        self.domain_classifier = DomainClassifier(causal_dim, num_domains)
        
        # Stage 4
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

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, adj: torch.Tensor = None):
        outputs = {}
        
        # Stage 2: Modality Disentangle
        z_img_g, z_img_s = self.img_proj(img_feat)
        z_txt_g, z_txt_s = self.txt_proj(txt_feat)
        
        outputs["z_img_g"] = z_img_g
        outputs["z_img_s"] = z_img_s
        outputs["z_txt_g"] = z_txt_g
        outputs["z_txt_s"] = z_txt_s
        
        # Thống nhất đặc trưng chung (Unified space)
        z_unified = torch.cat([z_img_g, z_txt_g], dim=-1)
        outputs["z_unified"] = z_unified
        
        # Stage 3A: Causal Disentangle
        xc, xs = self.causal_disentangle(z_unified)
        
        outputs["xc"] = xc
        outputs["xs"] = xs
        
        # Domain logits cho Loss Adversarial (L_disc)
        outputs["domain_logits"] = self.domain_classifier(xc)
        
        # Stage 3B: Causal Graph Neural Network (Phase 2)
        if adj is not None:
            xc = self.gnn(xc, adj)
            outputs["xc_graph"] = xc
        
        # Stage 4: Classification
        outputs["logits"] = self.classifier(xc)
        
        return outputs
