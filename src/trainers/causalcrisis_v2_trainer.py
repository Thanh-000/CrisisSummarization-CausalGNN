import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ==========================================================
# 1. CORE LOSS FUNCTIONS CHO CAUSAL CRISIS V2 (PHASE 1)
# ==========================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon):
    Kéo các mẫu cùng nhãn lại gần nhau và đẩy mẫu khác nhãn ra xa.
    Hoạt động trên không gian Modal-General (z_img_g, z_txt_g).
    """
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # L2 Normalize the features
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask-out self-contrast cases (đường chéo chính)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob với độ an toàn chống NaN
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Mean log_prob over positive cases
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos.mean()
        return loss

def orthogonal_loss(v1, v2):
    """
    Orthogonal Loss: Tính toán Dot Product Penalty giữa 2 vector normalized.
    Giúp 2 không gian (General và Specific, hoặc Causal và Spurious) tách bạch thông tin.
    """
    v1_norm = F.normalize(v1, p=2, dim=-1)
    v2_norm = F.normalize(v2, p=2, dim=-1)
    return torch.abs((v1_norm * v2_norm).sum(dim=-1)).mean()

# ==========================================================
# 2. MIXUP AUGMENTATION
# ==========================================================

def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Mixup trên không gian đại diện chung (Unified Representation Space).
    Tạo rào cản nội suy tuyến tính giúp mô hình chống overfitting với features.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================================
# 3. SCHEDULE GRL (GRADIENT REVERSAL LAYER LAMBDA)
# ==========================================================

def get_grl_lambda(epoch, max_epochs, warmup=20, max_lambda=10.0, gamma=10):
    """
    Tăng dần trọng số GRL theo Cosine hoặc Logistic Curve.
    Trong epochs đầu (Warmup), phạt domain rất nhẹ hoặc bằng 0.
    Theo CAMO, max_lambda có thể lên đến 10.
    """
    if epoch < warmup:
        return 0.0
    p = (epoch - warmup) / max(max_epochs - warmup, 1)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))

class GradientReversalFunction(torch.autograd.Function):
    """
    Lớp Autograd để thực hiện nghịch đảo gradient (GRL).
    Chiều đi (forward): Giữ nguyên input.
    Chiều về (backward): Đảo ngược gradient và nhân với lambda.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None

# ==========================================================
# 4. TRAINING LOOP CHO CAUSALCRISIS V2 (PHASE 1)
# ==========================================================

class Phase1Trainer:
    def __init__(self, model, optimizer, device, max_epochs=200):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_domain = nn.CrossEntropyLoss()
        self.criterion_supcon = SupConLoss(temperature=0.7)
        
        # Hyperparameters Phase 1 (Tuned to prevent loss dominance)
        self.alpha_task = 1.0
        self.alpha_supcon = 0.1   # Giảm mạnh SupCon xuống 0.1 để tránh nhiễu Task Loss
        self.alpha_orth = 0.1     # Giảm Orthogonal xuống 0.1
        self.grl_warmup = 5       # Rút ngắn Warmup để GRL sớm có tác dụng

    def train_epoch(self, dataloader, epoch, use_mixup=True):
        self.model.train()
        total_loss = 0
        total_loss_task = 0
        total_loss_sup = 0
        total_loss_orth = 0
        total_loss_disc = 0
        
        all_preds = []
        all_targets = []
        
        # Ramp-up Lambda cho GRL: max_lambda=0.1 (quan trọng! Nếu >= 1.0 mô hình sẽ sập gẫy)
        grl_lambda = get_grl_lambda(epoch, self.max_epochs, warmup=self.grl_warmup, max_lambda=0.1)
        
        for batch in dataloader:
            if len(batch) == 4:
                img_feat, txt_feat, labels, domains = [b.to(self.device) for b in batch]
            else:
                img_feat, txt_feat, labels = [b.to(self.device) for b in batch]
                domains = torch.zeros_like(labels)
            
            self.optimizer.zero_grad()
            out = self.model(img_feat, txt_feat)
            z_unified, xc, xs = out["z_unified"], out["xc"], out["xs"]
            
            # [A] Orthogonal Loss
            loss_orth = orthogonal_loss(out["z_img_g"], out["z_img_s"]) + \
                        orthogonal_loss(out["z_txt_g"], out["z_txt_s"]) + \
                        orthogonal_loss(xc, xs)
            
            # [B] SupCon Loss (Tương quan kéo gần cùng nhãn)
            try:
                loss_supcon = self.criterion_supcon(z_unified, labels)
            except:
                loss_supcon = torch.tensor(0.0).to(self.device)
            
            # [C] Mixup
            if use_mixup and epoch >= 5: 
                # Chỉnh alpha lớn lên (1.0) để mixup bẻ cong không gian mạnh hơn, giảm overfitting
                mixed_z, y_a, y_b, lam = mixup_data(z_unified, labels, alpha=1.0, device=self.device)
                mixed_xc, _ = self.model.causal_disentangle(mixed_z)
                mixed_logits = self.model.classifier(mixed_xc)
                
                loss_task = mixup_criterion(self.criterion_cls, mixed_logits, y_a, y_b, lam)
                preds = torch.argmax(out["logits"], dim=1) 
            else:
                loss_task = self.criterion_cls(out["logits"], labels)
                preds = torch.argmax(out["logits"], dim=1)
                
            # [D] GRL Domain Loss
            xc_reversed = GradientReversalFunction.apply(xc, grl_lambda)
            domain_logits_adv = self.model.domain_classifier(xc_reversed)
            loss_disc = self.criterion_domain(domain_logits_adv, domains)
            
            loss = (self.alpha_task * loss_task) + \
                   (self.alpha_supcon * loss_supcon) + \
                   (self.alpha_orth * loss_orth) + \
                   (0.01 * loss_disc)  # [HOTFIX] Khóa hẳn hàm GRL xuống 1% để Task Loss dẫn đường
                   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_task += loss_task.item()
            if isinstance(loss_supcon, torch.Tensor):
                total_loss_sup += loss_supcon.item()
            total_loss_orth += loss_orth.item()
            total_loss_disc += loss_disc.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Chỉ in breakdown loss ở batch cuối/tổng hợp để dễ xem
        if epoch % 5 == 0 or epoch == 1:
            n_b = len(dataloader)
            print(f"      [Loss Breakdown] Task: {total_loss_task/n_b:.3f} | SupCon: {total_loss_sup/n_b:.3f} | Orth: {total_loss_orth/n_b:.3f} | GRL: {total_loss_disc/n_b:.3f}")
            
        return total_loss / len(dataloader), epoch_f1

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            if len(batch) == 4:
                img_feat, txt_feat, labels, domains = [b.to(self.device) for b in batch]
            else:
                img_feat, txt_feat, labels = [b.to(self.device) for b in batch]
                
            out = self.model(img_feat, txt_feat)
            loss = self.criterion_cls(out["logits"], labels)
            
            total_loss += loss.item()
            preds = torch.argmax(out["logits"], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        bAcc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        return total_loss / len(dataloader), f1, bAcc

# ==========================================================
# 5. K-NN GRAPH & PHASE 2 TRAINER
# ==========================================================

def build_knn_graph(features: torch.Tensor, k: int=5, drop_edge_p: float=0.0, training: bool=False, temperature: float=0.1) -> torch.Tensor:
    """
    Tạo ma trận kề (Soft-Attention Graph) dựa trên Top-K Cosine Similarity.
    Sử dụng Softmax để lấy Local Attention thay vì Hard-Edge (1.0/0.0).
    """
    batch_size = features.size(0)
    # Cosine similarity matrix
    norm_feat = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.matmul(norm_feat, norm_feat.T) 
    
    # Loại trừ kết nối với chính mình để không làm nhiễu Attention
    sim_matrix.fill_diagonal_(-float('inf'))
    
    # Lấy top-k láng giềng
    k = min(k, batch_size - 1)
    if k <= 0: return torch.eye(batch_size, device=features.device)
    
    topk_sim, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    # GAT-like Softmax Attention Weights
    weights = F.softmax(topk_sim / temperature, dim=1)
    
    # Tạo ma trận kề Soft
    adj = torch.zeros((batch_size, batch_size), device=features.device)
    adj.scatter_(1, topk_indices, weights)
    
    # Symmetric Graph Transformation
    adj = (adj + adj.t()) / 2.0
    
    # DropEdge augmentation
    if training and drop_edge_p > 0.0:
        mask = torch.rand_like(adj) > drop_edge_p
        adj = adj * mask.float()
        
    # Row Normalization for GraphSAGE
    rowsum = adj.sum(1, keepdim=True)
    adj = adj / torch.clamp(rowsum, min=1e-8)
    
    return adj



class MemoryBank:
    """Kho lưu trữ FIFO K mẫu Spurious Features (K=256) cho Backdoor Adjustment"""
    def __init__(self, size=256, dim=256, device='cuda'):
        self.size = size
        self.dim = dim
        self.device = device
        self.bank = torch.zeros(size, dim, device=device)
        self.ptr = 0
        self.is_full = False
        
    def update(self, features):
        batch_size = features.size(0)
        features = features.detach().clone()
        if batch_size >= self.size:
            self.bank = features[-self.size:]
            self.ptr = 0
            self.is_full = True
            return
            
        end_idx = self.ptr + batch_size
        if end_idx <= self.size:
            self.bank[self.ptr:end_idx] = features
            self.ptr = end_idx
        else:
            overflow = end_idx - self.size
            self.bank[self.ptr:] = features[:batch_size-overflow]
            self.bank[:overflow] = features[batch_size-overflow:]
            self.ptr = overflow
            
        if self.ptr >= self.size:
            self.ptr = 0
            self.is_full = True
            
    def sample(self, M=4):
        if not self.is_full and self.ptr == 0:
            return torch.zeros(M, self.dim, device=self.device)
        max_idx = self.size if self.is_full else self.ptr
        indices = torch.randint(0, max_idx, (M,), device=self.device)
        return self.bank[indices]

class Phase2Trainer(Phase1Trainer):
    """
    Phase 2 Trainer: Kết tụ sức mạnh của GNN bằng cách kết nối các mẫu trong Batch thành mạng nhện.
    Tính năng mới:
    - DropEdge(0.3) chống Overfitting cho GNN
    - Backdoor Adjustment qua Memory Bank
    - Progressive Introduction (Warmup -> GNN Intro -> Full)
    """
    def __init__(self, model, optimizer, device, max_epochs=200, k_neighbors=5, memory_size=256, m_samples=4):
        super().__init__(model, optimizer, device, max_epochs)
        self.k_neighbors = k_neighbors
        self.drop_edge_p = 0.3
        self.m_samples = m_samples
        
        # Khởi tạo Memory Bank lưu Spurious Features để làm Backdoor Intervention
        spurious_dim = model.causal_disentangle.spurious_enc[-1].out_features
        self.memory_bank = MemoryBank(size=memory_size, dim=spurious_dim, device=device)

    def train_epoch(self, dataloader, epoch, use_mixup=False):
        self.model.train()
        self.current_epoch = epoch
        total_loss = 0
        total_loss_task_p1 = 0
        total_loss_ba = 0
        
        all_preds = []
        all_targets = []
        
        # Schedule GNN & DropEdge 
        enable_gnn = True 
        enable_dropedge = True
        
        # Nhận Mode Config từ class (default is E)
        config_mode = getattr(self, "config_mode", "E")
        
        if config_mode == "E":
            enable_backdoor = False # Cắt đứt hoàn toàn Backdoor Adjustment
            # GNN Weight Ramp Tuyến tính Max=0.2 tại Epoch 15
            alpha_gnn = min(0.2, 0.2 * (epoch / 15.0))
        elif config_mode == "C":
            enable_backdoor = epoch >= 10 # Delay BA
            # Ramp Tuyến tính Max=0.2 tại Epoch 15
            alpha_gnn = min(0.2, 0.2 * (epoch / 15.0))
        elif config_mode == "G_ONLY":
            enable_backdoor = False
            alpha_gnn = 1.0 # FULL GNN POWER
        elif config_mode == "REVAMP":
            enable_backdoor = True # Chạy BA liền từ đầu vì đã dọn Linear Classifier rác
            alpha_gnn = min(0.3, 0.3 * (epoch / 15.0)) # Trọng số GNN an toàn (30% quyền lực)
        else:
            enable_backdoor = epoch >= 20
            import math
            alpha_gnn = 0.5 * (1.0 - math.cos(math.pi * min(epoch, 15) / 15.0)) / 2.0
            
        grl_lambda = get_grl_lambda(epoch, self.max_epochs, warmup=self.grl_warmup, max_lambda=0.1)
        
        for batch in dataloader:
            if len(batch) == 4:
                img_feat, txt_feat, labels, domains = [b.to(self.device) for b in batch]
            else:
                img_feat, txt_feat, labels = [b.to(self.device) for b in batch]
                domains = torch.zeros_like(labels)
            
            self.optimizer.zero_grad()
            
            # --- 1. CHẠY NHÁP PHASE 1 ĐỂ LẤY Xc VÀ LƯU Xs VÀO MEMORY BANK ---
            out_draft = self.model(img_feat, txt_feat)
            
            if config_mode == "G_ONLY":
                # DETACH Xc, Xs để chặn hoàn toàn gradient chảy về MLP từ nhánh GNN
                xc_draft = out_draft["xc"].detach()
                xs_draft = out_draft["xs"].detach()
            else:
                xc_draft, xs_draft = out_draft["xc"], out_draft["xs"]
                
             # Update FIFO Spurious Bank
            self.memory_bank.update(xs_draft)
            
            # --- 2. XÂY GRAPH NẾU ENABLE ---
            adj = None
            if enable_gnn:
                adj = build_knn_graph(xc_draft, self.k_neighbors,
                                      drop_edge_p=self.drop_edge_p if enable_dropedge else 0.0,
                                      training=True)
                
            # --- 3. FORWARD CHÍNH THỨC ---
            # training=True -> backdoor_xs = None. Model tự dùng Xs detached của lô hiện tại để build logits_ba
            out = self.model(img_feat, txt_feat, adj=adj, backdoor_xs=None)
            
            # [A] Phase 1 Losses (Orthogonal, SupCon, ML_Task)
            loss_orth = orthogonal_loss(out["z_img_g"], out["z_img_s"]) + \
                        orthogonal_loss(out["z_txt_g"], out["z_txt_s"]) + \
                        orthogonal_loss(out["xc"], out["xs"])
            
            try:
                loss_supcon = self.criterion_supcon(out["z_unified"], labels) 
            except:
                loss_supcon = torch.tensor(0.0).to(self.device)
                
            # Nhánh chuẩn P1
            loss_task_p1 = self.criterion_cls(out["logits"], labels)
            
            # [B] Nhánh BA-GNN hoặc thuần GNN
            if enable_backdoor:
                loss_task_gnn = self.criterion_cls(out["logits_ba"], labels)
                preds = torch.argmax(out["logits_ba"], dim=1)
            else:
                loss_task_gnn = self.criterion_cls(out["logits_gnn"], labels)
                if enable_gnn:
                    preds = torch.argmax(out["logits_gnn"], dim=1)
                else:
                    preds = torch.argmax(out["logits"], dim=1)
                
            # [C] GRL Domain Loss trên Graph Causal Features
            xc_reversed = GradientReversalFunction.apply(out["xc_graph"], grl_lambda)
            domain_logits_adv = self.model.domain_classifier(xc_reversed)
            loss_disc = self.criterion_domain(domain_logits_adv, domains)
                
            # --- 4. TỔNG HỢP LOSS ---
            if config_mode == "G_ONLY":
                loss = loss_task_gnn
            else:
                loss = (self.alpha_task * loss_task_p1) + \
                       (self.alpha_supcon * loss_supcon) + \
                       (self.alpha_orth * loss_orth) + \
                       (alpha_gnn * loss_task_gnn) + \
                       (0.01 * loss_disc)
                   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_task_p1 += loss_task_p1.item()
            total_loss_ba += loss_task_gnn.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        if epoch % 5 == 0 or epoch == 1:
            n_b = len(dataloader)
            print(f"      [Phase 2 Loss] Total: {total_loss/n_b:.3f} | MLP Task: {total_loss_task_p1/n_b:.3f} | GNN Task: {total_loss_ba/n_b:.3f} | GNN_Wt: {alpha_gnn:.2f}")
            
        return total_loss / len(dataloader), epoch_f1

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        epoch = getattr(self, 'current_epoch', 15)
        config_mode = getattr(self, "config_mode", "E")
        enable_gnn = True
        
        if config_mode == "E":
            enable_backdoor = False
        elif config_mode == "C":
            enable_backdoor = epoch >= 10
        elif config_mode == "G_ONLY":
            enable_backdoor = False
        elif config_mode == "REVAMP":
            enable_backdoor = True
        else:
            enable_backdoor = False
        
        for batch in dataloader:
            if len(batch) == 4:
                img_feat, txt_feat, labels, domains = [b.to(self.device) for b in batch]
            else:
                img_feat, txt_feat, labels = [b.to(self.device) for b in batch]
                
            out_draft = self.model(img_feat, txt_feat)
            
            # Nếu chưa enable GNN thì bỏ qua adj tạo graph
            adj = None
            if enable_gnn:
                adj = build_knn_graph(out_draft["xc"], self.k_neighbors, drop_edge_p=0.0, training=False)
            
            if enable_backdoor:
                # Backdoor Adjustment trong lúc Test
                backdoor_xs = None
                if self.memory_bank.is_full or self.memory_bank.ptr > 0:
                    bank_samples = self.memory_bank.sample(M=self.m_samples)
                    backdoor_xs = bank_samples.unsqueeze(0).expand(img_feat.size(0), self.m_samples, -1)
                else:
                    backdoor_xs = out_draft["xs"].unsqueeze(1)
                    
                out = self.model(img_feat, txt_feat, adj=adj, backdoor_xs=backdoor_xs)
                loss = self.criterion_cls(out["logits_ba"], labels)
                preds = torch.argmax(out["logits_ba"], dim=1)
            else:
                out = self.model(img_feat, txt_feat, adj=adj, backdoor_xs=None)
                if enable_gnn:
                    loss = self.criterion_cls(out["logits_gnn"], labels)
                    preds = torch.argmax(out["logits_gnn"], dim=1)
                else:
                    loss = self.criterion_cls(out["logits"], labels)
                    preds = torch.argmax(out["logits"], dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        bAcc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        return total_loss / len(dataloader), f1, bAcc
