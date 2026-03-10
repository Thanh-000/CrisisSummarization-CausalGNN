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
        self.alpha_supcon = 0.5   # Giảm từ 3.0 xuống 0.5
        self.alpha_orth = 0.1     # Giảm từ 1.0 xuống 0.1
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
        
        # Ramp-up Lambda cho GRL
        grl_lambda = get_grl_lambda(epoch, self.max_epochs, warmup=self.grl_warmup, max_lambda=1.0)
        
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
            # Nếu batch size quá nhỏ hoặc chỉ có 1 class trong batch, SupCon dễ gây NaN/Loss to
            try:
                loss_supcon = self.criterion_supcon(z_unified, labels)
            except:
                loss_supcon = torch.tensor(0.0).to(self.device)
            
            # [C] Mixup
            if use_mixup and epoch >= 5: # Kích hoạt sớm từ Epoch 5 thay vì 30
                mixed_z, y_a, y_b, lam = mixup_data(z_unified, labels, alpha=0.5, device=self.device)
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
                   loss_disc  
                   
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
