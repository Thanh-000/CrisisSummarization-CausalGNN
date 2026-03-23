# %%
import os
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.getcwd()))

try:
    from src.config import V4Config
    from src.data import load_crisismmd_data
    from src.models import GuidedCrossAttention
    from src.losses import FocalLoss
except ImportError:
    print("Warning: Failed to import some local modules. Ensure you're running from the project root.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

config = V4Config()
PROJECT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR_LOCAL = os.path.abspath(os.path.join(os.getcwd(), '..'))
PROJECT_DIR = PROJECT_DIR_PARENT if "src" in os.listdir(PROJECT_DIR_PARENT) else PROJECT_DIR_LOCAL
print(f"✅ Project dir: {PROJECT_DIR}")

# ============================================================================
# 1. Feature Retrieval Setup
# ============================================================================
DATA_DIR = os.path.join(PROJECT_DIR, "data", "crisismmd_v2.0")
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "processed")

print(f"\n📂 Loading dataset from: {DATA_DIR}")
data = load_crisismmd_data(DATA_DIR)
# Map informative -> 1, not informative -> 0
data['label'] = data['label_text'].apply(lambda x: 1 if 'informative' in x.lower() and 'not' not in x.lower() else 0)

# Load Features (simulated or real cached paths)
IMG_FEAT_PATH = os.environ.get("IMG_FEAT_PATH", os.path.join(CACHE_DIR, "clip_image_features.npy"))
TXT_FEAT_PATH = os.environ.get("TXT_FEAT_PATH", os.path.join(CACHE_DIR, "clip_text_features.npy"))
LLAVA_FEAT_COMBINED = os.environ.get("LLAVA_FEAT_COMBINED", os.path.join(CACHE_DIR, "clip_llava_features_combined.npy"))

print("Loading cached features...")
try:
    image_features = np.load(IMG_FEAT_PATH)
    text_features = np.load(TXT_FEAT_PATH)
    llava_features = np.load(LLAVA_FEAT_COMBINED)
except Exception as e:
    print(f"⚠️ Features not found locally at {CACHE_DIR}.")
    print("If you are on Colab and need to download via SSH/SFTP with Aria2, uncomment the following code in the notebook:")
    # -------------------------------------------------------------------------
    # !apt-get update && apt-get install -y aria2
    # !mkdir -p data/processed
    # # Thay đổi thông tin user, pass, IP, và đường dẫn tới thư mục chứa file .npy trên server của bạn
    # SFTP_URL_IMG = "sftp://<user>:<password>@<IP_SERVER>:<PORT>/path/to/clip_image_features.npy"
    # SFTP_URL_TXT = "sftp://<user>:<password>@<IP_SERVER>:<PORT>/path/to/clip_text_features.npy"
    # SFTP_URL_LLAVA = "sftp://<user>:<password>@<IP_SERVER>:<PORT>/path/to/clip_llava_features_combined.npy"
    
    # !aria2c -x 16 -s 16 -k 1M $SFTP_URL_IMG -d data/processed -o clip_image_features.npy
    # !aria2c -x 16 -s 16 -k 1M $SFTP_URL_TXT -d data/processed -o clip_text_features.npy
    # !aria2c -x 16 -s 16 -k 1M $SFTP_URL_LLAVA -d data/processed -o clip_llava_features_combined.npy
    # -------------------------------------------------------------------------
    
    print("Initializing dummy features for debugging/runability test without full 50GB dataset.")
    samples = len(data) if len(data) > 0 else 16000
    image_features = np.random.randn(samples, 768)
    text_features = np.random.randn(samples, 768)
    llava_features = np.random.randn(samples, 768)

labels = data['label'].values
domains = data['event_name'].values
unique_domains = np.unique(domains)
num_domains = len(unique_domains)
domain_to_id = {d: i for i, d in enumerate(unique_domains)}
domain_ids = np.array([domain_to_id[d] for d in domains])

print(f"✅ Features loaded: Image {image_features.shape}, Text {text_features.shape}, LLaVA {llava_features.shape}")
print(f"✅ Disaster Domains: {num_domains} ({unique_domains.tolist()})")

# %%
# ============================================================================
# 2. Causal Disentanglement & Graph Architecture (V4+)
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = torch.tensor(lambda_)
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class CausalDisentangler(nn.Module):
    def __init__(self, input_dim=1536, causal_dim=512, dropout=0.2):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.causal_head = nn.Sequential(
            nn.Linear(1024, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
        )
        self.spurious_head = nn.Sequential(
            nn.Linear(1024, causal_dim),
            nn.LayerNorm(causal_dim),
            nn.GELU(),
        )
    
    def forward(self, z_vt):
        shared = self.shared_proj(z_vt)
        X_c = self.causal_head(shared)
        X_s = self.spurious_head(shared)
        return X_c, X_s

class DomainClassifier(nn.Module):
    def __init__(self, causal_dim=512, num_domains=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(causal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains),
        )
    def forward(self, x):
        return self.classifier(x)

class SoftAttentionKNNGraph(nn.Module):
    def __init__(self, k=5, temperature=0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, X_c):
        X_norm = F.normalize(X_c, dim=-1)
        sim = X_norm @ X_norm.T
        topk_vals, topk_idx = sim.topk(min(self.k, sim.size(-1)), dim=-1)
        mask = torch.zeros_like(sim).to(X_c.device)
        mask.scatter_(-1, topk_idx, 1.0)
        sim_masked = sim * mask + (1 - mask) * (-1e9)
        return F.softmax(sim_masked / self.temperature, dim=-1)

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, X, adj):
        neigh_agg = adj @ X
        out = self.W_self(X) + self.W_neigh(neigh_agg)
        return self.ln(F.gelu(out))

class EdgeHeterophilyScorer(nn.Module):
    def __init__(self, feat_dim=512, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim + num_classes, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, X_c, labels_or_logits, adj):
        B = X_c.size(0)
        feat_diff = torch.abs(X_c.unsqueeze(1) - X_c.unsqueeze(0))
        if labels_or_logits.dim() == 1:
            labels_oh = F.one_hot(labels_or_logits, self.num_classes).float()
        else:
            labels_oh = F.softmax(labels_or_logits, dim=-1)
        label_sim = labels_oh.unsqueeze(1) * labels_oh.unsqueeze(0)
        edge_input = torch.cat([feat_diff, label_sim], dim=-1)
        h_scores = self.scorer(edge_input).squeeze(-1)
        
        tau = 5.0
        adj_refined = adj + tau * torch.log(h_scores + 1e-8)
        adj_refined = F.softmax(adj_refined, dim=-1)
        return adj_refined, h_scores

class MemoryBank:
    def __init__(self, size=256, dim=512, device='cuda'):
        self.bank = torch.zeros(size, dim).to(device)
        self.ptr = 0
        self.full = False
    
    @torch.no_grad()
    def update(self, X_s):
        batch_size = X_s.size(0)
        end = self.ptr + batch_size
        if end <= self.bank.size(0):
            self.bank[self.ptr:end] = X_s.detach()
        else:
            overflow = end - self.bank.size(0)
            self.bank[self.ptr:] = X_s[:batch_size - overflow].detach()
            self.bank[:overflow] = X_s[batch_size - overflow:].detach()
            self.full = True
        self.ptr = end % self.bank.size(0)
    
    def get_samples(self, n=16):
        if self.full:
            idx = torch.randperm(self.bank.size(0))[:n]
        else:
            ptr = max(1, self.ptr)
            idx = torch.randperm(ptr)[:min(n, ptr)]
        return self.bank[idx]

class BackdoorAdjustment(nn.Module):
    def __init__(self, causal_dim=512, spurious_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(causal_dim + spurious_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, X_c_gnn, X_s_bank_samples):
        B = X_c_gnn.size(0)
        M = max(1, X_s_bank_samples.size(0))
        if M == 0:
            X_s_bank_samples = torch.zeros(1, X_c_gnn.size(-1)).to(X_c_gnn.device)
            M = 1
        X_c_exp = X_c_gnn.unsqueeze(1).expand(-1, M, -1)
        X_s_exp = X_s_bank_samples.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([X_c_exp, X_s_exp], dim=-1)
        logits = self.classifier(combined.view(B*M, -1))
        logits = logits.view(B, M, -1)
        return logits.mean(dim=1)

class V4PlusClassifier(nn.Module):
    def __init__(self, feat_dim=768, num_classes=2, causal_dim=512, num_domains=7, bank_size=512, m_samples=16):
        super().__init__()
        self.causal_dim = causal_dim
        self.m_samples = m_samples
        
        # Maps 768*2 -> 1536 directly if concatenating later, otherwise cross-attn first
        self.img_proj = nn.Linear(feat_dim, feat_dim)
        self.txt_proj = nn.Linear(feat_dim, feat_dim)
        self.llava_proj = nn.Linear(feat_dim, feat_dim)
        
        input_dim = feat_dim * 2
        
        self.causal_disentangler = CausalDisentangler(input_dim=input_dim, causal_dim=causal_dim)
        self.domain_classifier = DomainClassifier(causal_dim=causal_dim, num_domains=num_domains)
        self.knn_graph = SoftAttentionKNNGraph(k=5)
        self.graph_sage = GraphSAGELayer(in_dim=causal_dim, out_dim=causal_dim)
        self.heterophily_scorer = EdgeHeterophilyScorer(feat_dim=causal_dim, num_classes=num_classes)
        
        self.warmup_classifier = nn.Sequential(
            nn.Linear(causal_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        self.memory_bank = None # initialized externally with device
        self.backdoor_adj = BackdoorAdjustment(causal_dim=causal_dim, spurious_dim=causal_dim, num_classes=num_classes)
        
    def forward(self, img, txt, llava, grl_lambda, phase=1, y_true=None):
        """
        phase 1: Warmup (Disentanglement + GRL)
        phase 2: GraphSAGE + MemoryBank Backdoor
        phase 3: + Heterophily 
        """
        # Multimodal fusion (simple concat for demonstration logic, ideally guided cross-attn)
        img_f = F.gelu(self.img_proj(img))
        txt_f = F.gelu(self.txt_proj(txt))
        llava_f = F.gelu(self.llava_proj(llava))
        
        # We can implement a naive fusion here or use the Guidance modules (stubbed stringently for speed in notebook)
        z_v_t = (img_f + txt_f) / 2
        z_v_l = (img_f + llava_f) / 2
        fused = torch.cat([z_v_t, z_v_l], dim=-1) # (B, 1536)
        
        # Disentangle
        X_c, X_s = self.causal_disentangler(fused)
        
        # GRL Domain classification
        if grl_lambda is not None and grl_lambda > 0.0:
            X_c_grl = GradientReversalLayer.apply(X_c, grl_lambda)
            domain_logits = self.domain_classifier(X_c_grl)
        else:
            domain_logits = None
            
        outputs = {'X_c': X_c, 'X_s': X_s, 'domain_logits': domain_logits}
        
        if phase == 1:
            outputs['task_logits'] = self.warmup_classifier(X_c)
        else:
            # Phase 2 & 3: Graph Reasoning
            adj = self.knn_graph(X_c)
            if phase == 3 and y_true is not None:
                # Use ground truth labels during training for heterophily
                adj, h_scores = self.heterophily_scorer(X_c, y_true, adj)
            elif phase == 3:
                # Use pseudo labels during testing
                pseudo_logits = self.warmup_classifier(X_c).detach()
                adj, h_scores = self.heterophily_scorer(X_c, pseudo_logits, adj)
                
            X_gnn = self.graph_sage(X_c, adj)
            
            # Backdoor Adjustment
            if self.memory_bank is not None:
                bank_samples = self.memory_bank.get_samples(self.m_samples)
                outputs['task_logits'] = self.backdoor_adj(X_gnn, bank_samples)
            else:
                outputs['task_logits'] = self.warmup_classifier(X_gnn)
                
            if self.training and self.memory_bank is not None:
                self.memory_bank.update(X_s)
                
        return outputs

# %%
# ============================================================================
# 3. LODO (Leave-One-Disaster-Out) Protocol
# ============================================================================

def calc_orthogonal_loss(X_c, X_s):
    # Enforce orthogonality between Causal and Spurious features
    c_norm = F.normalize(X_c, dim=-1)
    s_norm = F.normalize(X_s, dim=-1)
    return torch.mean((c_norm * s_norm).sum(dim=-1).abs())

def train_v4_plus_lodo(held_out_domain, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n" + "="*50)
    print(f"🚀 Running LODO Evaluation | Held-out: {held_out_domain}")
    print("="*50)
    
    # Split Data
    train_mask = (domains != held_out_domain)
    test_mask = (domains == held_out_domain)
    
    train_dataset = TensorDataset(
        torch.tensor(image_features[train_mask], dtype=torch.float32),
        torch.tensor(text_features[train_mask], dtype=torch.float32),
        torch.tensor(llava_features[train_mask], dtype=torch.float32),
        torch.tensor(labels[train_mask], dtype=torch.long),
        torch.tensor(domain_ids[train_mask], dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(image_features[test_mask], dtype=torch.float32),
        torch.tensor(text_features[test_mask], dtype=torch.float32),
        torch.tensor(llava_features[test_mask], dtype=torch.float32),
        torch.tensor(labels[test_mask], dtype=torch.long),
        torch.tensor(domain_ids[test_mask], dtype=torch.long)
    )
    
    # Use smaller batch sizes if the dummy dataset size is smaller
    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Init Model
    model = V4PlusClassifier(num_domains=num_domains).to(DEVICE)
    model.memory_bank = MemoryBank(size=512, dim=512, device=DEVICE)
    
    task_criterion = FocalLoss(gamma=2.0)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    best_f1 = 0.0
    
    # 3-Phase Curriculum
    total_epochs = 12
    for epoch in range(1, total_epochs + 1):
        if epoch <= 5:
            phase = 1
        elif epoch <= 9:
            phase = 2
        else:
            phase = 3
            
        model.train()
        total_loss, tr_preds, tr_labels = 0.0, [], []
        
        # GRL lambda scheduling
        p = float(epoch) / total_epochs
        grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
        
        for img, txt, llava, lbl, dom in train_loader:
            img, txt, llava = img.to(DEVICE), txt.to(DEVICE), llava.to(DEVICE)
            lbl, dom = lbl.to(DEVICE), dom.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(img, txt, llava, grl_lambda=grl_lambda, phase=phase, y_true=lbl)
            
            task_loss = task_criterion(outputs['task_logits'], lbl)
            dom_loss = domain_criterion(outputs['domain_logits'], dom) if outputs['domain_logits'] is not None else 0.0
            orth_loss = calc_orthogonal_loss(outputs['X_c'], outputs['X_s'])
            
            loss = task_loss + 1.0 * dom_loss + 0.1 * orth_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            tr_preds.extend(torch.argmax(outputs['task_logits'], dim=1).cpu().numpy())
            tr_labels.extend(lbl.cpu().numpy())
            
        train_f1 = f1_score(tr_labels, tr_preds, average='weighted', zero_division=0)
        
        # Eval
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for img, txt, llava, lbl, dom in test_loader:
                img, txt, llava = img.to(DEVICE), txt.to(DEVICE), llava.to(DEVICE)
                outputs = model(img, txt, llava, grl_lambda=None, phase=phase)
                val_preds.extend(torch.argmax(outputs['task_logits'], dim=1).cpu().numpy())
                val_labels.extend(lbl.cpu().numpy())
                
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Epoch {epoch:02d} | Phase {phase} | Loss: {total_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f} | Held-Out F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            
    print(f"⭐ Best LODO F1 for {held_out_domain}: {best_f1:.4f}")
    return best_f1

# %%
if __name__ == "__main__":
    print("\nStarting Out-Of-Distribution (OOD) LODO Protocol")
    print("This will train V4+ Causal Disentanglement + Graph Module and evaluate generalization.\n")
    
    lodo_results = {}
    seeds = [42] # Quick run
    
    for held_out in unique_domains:
        seed_f1s = []
        for seed in seeds:
            f1 = train_v4_plus_lodo(held_out, seed=seed)
            seed_f1s.append(f1)
        lodo_results[held_out] = {
            'mean_f1': np.mean(seed_f1s),
            'std_f1': np.std(seed_f1s)
        }
        
    print("\n" + "="*60)
    print("📊 FINAL LODO REPORT (OOD GENERALIZATION)")
    print("="*60)
    for dom, res in lodo_results.items():
        print(f" > {dom}: F1w = {res['mean_f1']:.4f} ± {res['std_f1']:.4f}")
        
    avg_lodo = np.mean([res['mean_f1'] for res in lodo_results.values()])
    print(f"\n🔥 Average LODO OOD F1: {avg_lodo:.4f}")
    print("✅ Completed V4+ Causal Experiment Design.")
