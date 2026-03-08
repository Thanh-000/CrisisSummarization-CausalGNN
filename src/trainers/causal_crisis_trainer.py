"""
CausalCrisis Trainer -- Training loop cho Causal Multimodal Reasoning
=====================================================================
Ke thua GEDATrainer, bo sung:
  - Domain labels extraction tu CrisisMMD (event_name)
  - GRL lambda scheduling (sigmoid warmup)
  - Graph xay tren Causal features
  - CausalCrisisLoss voi 5 thanh phan
  - Intervention consistency loss (L_int)

Pipeline: Extract CLIP -> PCA -> Causal Disentangle -> kNN graph (causal)
          -> Train CausalCrisis -> Evaluate
"""

import os
import sys
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ============================================================
# 1. FEATURE EXTRACTION (ke thua tu geda_trainer.py)
# ============================================================

def extract_clip_features_with_domain(dataset_path, task="task1",
                                       split="train", device="cuda"):
    """
    Extract CLIP features cho CrisisMMD, kem theo domain (event_name).
    Mo rong tu geda_trainer.extract_clip_features.

    Returns:
        img_features: (N, 512) CLIP image embeddings
        txt_features: (N, 512) CLIP text embeddings
        labels: (N,) string labels
        event_names: (N,) string event names (disaster type)
    """
    import open_clip
    from PIL import Image

    cache_dir = f"{dataset_path}/.cache/clip_features"
    os.makedirs(cache_dir, exist_ok=True)

    cache_img = f"{cache_dir}/{task}_{split}_img.npy"
    cache_txt = f"{cache_dir}/{task}_{split}_txt.npy"

    # Map task to TSV file
    task_map = {
        "task1": "task_informative_text_img",
        "task2": "task_humanitarian_text_img",
        "task3": "task_damage_text_img",
    }
    tsv_name = f"{task_map[task]}_{split}.tsv"

    # Auto-extract zip nếu thư mục chưa có (CrisisMMD v2.0 ship dạng zip)
    for zip_name in ["crisismmd_datasplit_all.zip", "crisismmd_datasplit_settingA.zip"]:
        zip_path = f"{dataset_path}/{zip_name}"
        extract_dir = zip_path.replace(".zip", "")
        if os.path.exists(zip_path) and not os.path.isdir(extract_dir):
            import zipfile
            print(f"  📦 Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dataset_path)
            print(f"  ✅ Extracted to {extract_dir}")

    candidates = [
        f"{dataset_path}/crisismmd_datasplit_settingA/{tsv_name}",
        f"{dataset_path}/crisismmd_datasplit_all/{tsv_name}",
        f"{dataset_path}/{tsv_name}",
    ]
    tsv_path = None
    for c in candidates:
        if os.path.exists(c):
            tsv_path = c
            break
    if tsv_path is None:
        raise FileNotFoundError(f"TSV not found in: {candidates}")

    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t')

    # Lay label
    lbl_col = 'label' if 'label' in df.columns else [
        c for c in df.columns if 'label' in c.lower()
    ][0]
    labels = df[lbl_col].astype(str).values

    # Lay event name (domain)
    event_col = None
    for col_name in ['event_name', 'event', 'disaster_type']:
        if col_name in df.columns:
            event_col = col_name
            break
    if event_col is None:
        # Fallback: dung cot dau tien (thuong la event_name)
        event_col = df.columns[0]
        print(f"  [WARN] Khong tim thay cot event_name, dung '{event_col}'")

    event_names = df[event_col].astype(str).values

    # Check cache
    if os.path.exists(cache_img) and os.path.exists(cache_txt):
        print(f"  Loading cached features: {task}/{split}")
        img_features = np.load(cache_img)
        txt_features = np.load(cache_txt)
        return img_features, txt_features, labels, event_names

    print(f"  Extracting CLIP features: {task}/{split}")

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    img_candidates = [c for c in df.columns if ('image' in c.lower() or 'img' in c.lower()) and 'label' not in c.lower() and 'id' not in c.lower()]
    
    img_col = None
    for priority_col in ['image_path', 'image_url', 'image_info', 'image']:
        if priority_col in df.columns:
            img_col = priority_col
            break
    if img_col is None and len(img_candidates) > 0:
        img_col = img_candidates[-1] # Try the last candidate which is usually image_info or image_url instead of something early like image_id

    txt_col = [c for c in df.columns
               if 'text' in c.lower() and 'label' not in c.lower()
               and 'llava' not in c.lower()][0]

    img_features = []
    txt_features = []
    failed_indices = []

    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]

            # Text
            texts = batch_df[txt_col].tolist()
            tokens = tokenizer(texts).to(device)
            t_feat = model.encode_text(tokens)
            t_feat = F.normalize(t_feat, dim=-1)
            txt_features.append(t_feat.cpu().numpy())

            # Image
            batch_imgs = []
            current_batch_offset = 0
            for _, row in batch_df.iterrows():
                img_path = f"{dataset_path}/{row[img_col]}"
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = preprocess(img)
                except Exception as e:
                    img = torch.zeros(3, 224, 224)
                    failed_indices.append(i + current_batch_offset)
                    print(f"  [WARN] Failed image: {img_path}: {e}")
                batch_imgs.append(img)
                current_batch_offset += 1

            img_batch = torch.stack(batch_imgs).to(device)
            i_feat = model.encode_image(img_batch)
            i_feat = F.normalize(i_feat, dim=-1)
            img_features.append(i_feat.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"    {i}/{len(df)}...")

    img_features = np.concatenate(img_features, axis=0)
    txt_features = np.concatenate(txt_features, axis=0)

    # Issue 42: Remove failed samples
    if failed_indices:
        print(f"  [INFO] Removing {len(failed_indices)} failed samples.")
        keep_mask = np.ones(len(img_features), dtype=bool)
        keep_mask[failed_indices] = False
        img_features = img_features[keep_mask]
        txt_features = txt_features[keep_mask]
        labels = labels[keep_mask]
        event_names = event_names[keep_mask]

    np.save(cache_img, img_features)
    np.save(cache_txt, txt_features)
    return img_features, txt_features, labels, event_names


# ============================================================
# 2. GRAPH CONSTRUCTION (ke thua tu geda_trainer.py)
# ============================================================

def build_knn_graph(features, k=16, use_faiss=True, as_sparse=True):
    """
    Build k-NN adjacency matrix.
    Returns normalized adjacency: D^{-1}A.
    Uses sparse tensor by default to prevent OOM on full dataset.
    """
    n = features.shape[0]

    if use_faiss:
        try:
            import faiss
            d = features.shape[1]
            index = faiss.IndexFlatIP(d)
            feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            index.add(feats_norm.astype(np.float32))
            _, indices = index.search(feats_norm.astype(np.float32), k + 1)
            indices = indices[:, 1:]
        except ImportError:
            use_faiss = False

    if not use_faiss:
        nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        nn_model.fit(features)
        _, indices = nn_model.kneighbors(features)
        indices = indices[:, 1:]

    if as_sparse:
        # Build Sparse COO tensor
        rows = []
        cols = []
        for i in range(n):
            for j in indices[i]:
                if j < n:
                    rows.extend([i, j])
                    cols.extend([j, i])
        
        # Self loop
        rows.extend(list(range(n)))
        cols.extend(list(range(n)))
        
        indices_tensor = torch.tensor([rows, cols], dtype=torch.long)
        values_tensor = torch.ones(len(rows), dtype=torch.float32)
        
        adj = torch.sparse_coo_tensor(indices_tensor, values_tensor, (n, n)).coalesce()
        
        # Row normalization
        deg = torch.sparse.sum(adj, dim=1).to_dense() # (n,)
        deg_inv = 1.0 / (deg + 1e-8)
        
        # To normalize A_sym = D^{-1} A, we can do row-wise multiplication
        # Find indices and multiply their values by deg_inv[row]
        v = adj.values()
        r = adj.indices()[0]
        v_norm = v * deg_inv[r]
        
        adj_normalized = torch.sparse_coo_tensor(adj.indices(), v_norm, (n, n)).coalesce()
        return adj_normalized
    else:
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in indices[i]:
                if j < n:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        adj += np.eye(n, dtype=np.float32)
        row_sum = adj.sum(axis=1, keepdims=True)
        adj = adj / (row_sum + 1e-8)

        return torch.tensor(adj, dtype=torch.float32)


# ============================================================
# 3. CAUSALCRISIS TRAINER
# ============================================================

class CausalCrisisTrainer:
    """
    Training loop cho CausalCrisis model.

    Khac biet voi GEDATrainer:
    - Domain labels handling
    - GRL lambda scheduling (sigmoid warmup)
    - CausalCrisisLoss voi 5 thanh phan
    - Graph xay tren Causal features (o experiment level)
    - Logging domain accuracy de verify invariance

    Features ke thua:
    - Full-graph training (nhu Paper 1)
    - Shuffle split (40% train / 60% val tu labeled data)
    - Early stopping voi harmonic mean F1
    - Crash-safe checkpointing
    """

    def __init__(
        self,
        model,
        criterion,
        device="cuda",
        lr=1e-4,
        weight_decay=1e-3,
        max_epochs=500,
        patience=150,
        checkpoint_dir="/content/causal_results/checkpoints",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=15, min_lr=1e-6
        )
        self._val_split = None  # Cache validation_split thay vi shuffle tung epoch

    def _shuffle_split(self, n_labeled, train_frac=0.8):
        """Paper 1 style: split label data dung de tinh loss va evaluate."""
        if self._val_split is not None:
             return self._val_split
        perm = torch.randperm(n_labeled)
        n_train = int(n_labeled * train_frac)
        self._val_split = (perm[:n_train], perm[n_train:])
        return self._val_split

    def train_epoch(self, img_feat, txt_feat, labels, adj,
                    domain_labels, labeled_mask, task="task1",
                    grl_lambda=1.0):
        """Train 1 epoch voi shuffle split + causal losses."""
        self.model.train()

        # Shuffle split tren labeled data
        labeled_idx = torch.where(labeled_mask)[0]
        n_labeled = len(labeled_idx)
        train_idx, val_idx = self._shuffle_split(n_labeled)
        train_idx = labeled_idx[train_idx]
        val_idx = labeled_idx[val_idx]

        # Forward pass (full graph)
        outputs = self.model(
            img_feat, txt_feat, adj,
            domain_labels=domain_labels,
            task=task,
            grl_lambda=grl_lambda,
        )

        # Compute loss (chi tren labeled train subset)
        train_mask = torch.zeros(len(img_feat), dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True

        total_loss, loss_dict = self.criterion(
            outputs, labels,
            domain_labels=domain_labels,
            mask=train_mask,
        )

        # L_int (intervention consistency loss)
        if "c_vt_original" in outputs and "c_vt_intervened" in outputs:
            import warnings
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 # Lay task dau tien de lam target logits can ban
                 task_key = list(labels.keys())[0] if labels else task
                 
                 z_orig_attn = self.model.diff_attn(outputs["c_vt_original"])
                 z_orig = self.model.norm_diff(outputs["c_vt_original"] + z_orig_attn)
                 logits_orig = getattr(self.model, f"head_{task_key}")(z_orig)
                 
                 z_int_attn = self.model.diff_attn(outputs["c_vt_intervened"])
                 z_int = self.model.norm_diff(outputs["c_vt_intervened"] + z_int_attn)
                 logits_int = getattr(self.model, f"head_{task_key}")(z_int)

                 l_int = self.criterion.intervention_consistency_loss(logits_orig, logits_int)
                 total_loss = total_loss + self.criterion.alpha_int * l_int
                 loss_dict["int"] = l_int.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Khong step scheduler o day nua, chuyen sang train loop do dung ReduceLROnPlateau

        # Evaluate on val subset
        self.model.eval()
        with torch.no_grad():
            outputs_val = self.model(
                img_feat, txt_feat, adj,
                domain_labels=domain_labels,
                task=task,
                grl_lambda=0.0,  # khong GRL cho val
            )

        val_f1s = {}
        train_f1s = {}
        for t, logits in outputs_val.items():
            if t in labels and t.startswith("task"):
                # Val F1
                pred_val = logits[val_idx].argmax(dim=-1).cpu().numpy()
                true_val = labels[t][val_idx].cpu().numpy()
                val_f1s[t] = f1_score(true_val, pred_val, average='weighted', zero_division=0)

                # Train F1
                pred_tr = logits[train_idx].argmax(dim=-1).cpu().numpy()
                true_tr = labels[t][train_idx].cpu().numpy()
                train_f1s[t] = f1_score(true_tr, pred_tr, average='weighted', zero_division=0)

        # Harmonic mean F1
        avg_val_f1 = np.mean(list(val_f1s.values())) if val_f1s else 0
        avg_train_f1 = np.mean(list(train_f1s.values())) if train_f1s else 0
        if avg_val_f1 + avg_train_f1 > 0:
            hm_f1 = 2 * avg_val_f1 * avg_train_f1 / (avg_val_f1 + avg_train_f1)
        else:
            hm_f1 = 0

        # Domain accuracy (verify causal = invariant)
        domain_acc = None
        if "domain_cv" in outputs_val and domain_labels is not None:
            dom_pred = outputs_val["domain_cv"][val_idx].argmax(dim=-1).cpu().numpy()
            dom_true = domain_labels[val_idx].cpu().numpy()
            domain_acc = accuracy_score(dom_true, dom_pred)
        
        # Log Learning Rate
        current_lr = self.optimizer.param_groups[0]['lr']

        return {
            "train_loss": total_loss.item(),
            "val_f1": avg_val_f1,
            "train_f1": avg_train_f1,
            "hm_f1": hm_f1,
            "val_f1s": val_f1s,
            "loss_components": loss_dict,
            "grl_lambda": grl_lambda,
            "domain_acc_causal": domain_acc,
            "lr": current_lr,
        }

    @torch.no_grad()
    def evaluate(self, img_feat, txt_feat, labels, adj,
                 domain_labels, test_mask, task="task1"):
        """Evaluate on test set."""
        self.model.eval()
        outputs = self.model(
            img_feat, txt_feat, adj,
            domain_labels=domain_labels,
            task=task,
            grl_lambda=0.0,  # No GRL at test time
        )

        test_idx = torch.where(test_mask)[0]
        results = {}

        for t, logits in outputs.items():
            if t in labels and t.startswith("task"):
                preds = logits[test_idx].argmax(dim=-1).cpu().numpy()
                trues = labels[t][test_idx].cpu().numpy()

                results[t] = {
                    "accuracy": float(accuracy_score(trues, preds)),
                    "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
                    "weighted_f1": float(f1_score(trues, preds, average='weighted', zero_division=0)),
                    "macro_f1": float(f1_score(trues, preds, average='macro', zero_division=0)),
                    "micro_f1": float(f1_score(trues, preds, average='micro', zero_division=0)),
                }

        # Domain invariance check
        if "domain_cv" in outputs and domain_labels is not None:
            dom_pred = outputs["domain_cv"][test_idx].argmax(dim=-1).cpu().numpy()
            dom_true = domain_labels[test_idx].cpu().numpy()
            results["domain_invariance"] = {
                "causal_domain_acc": float(accuracy_score(dom_true, dom_pred)),
                # Ly tuong: causal_domain_acc ≈ 1/num_domains (chance level)
            }

        return results

    def train(self, img_feat, txt_feat, labels, adj,
              domain_labels, labeled_mask, test_mask,
              task="task1", run_name="causal_crisis",
              use_causal_graph=True, k_neighbors=16):
        """
        Full training loop voi early stopping + GRL scheduling + Rebuild graph.
        """
        # Move to device
        img_feat = img_feat.to(self.device)
        txt_feat = txt_feat.to(self.device)
        adj = adj.to(self.device)
        labels = {t: l.to(self.device) for t, l in labels.items()}
        domain_labels = domain_labels.to(self.device)
        labeled_mask = labeled_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        best_hm_f1 = 0
        patience_counter = 0
        best_state = None
        history = []

        print(f"\n{'='*60}")
        print(f"  Training: {run_name}")
        print(f"  Labeled: {labeled_mask.sum().item()}, Test: {test_mask.sum().item()}")
        print(f"  Max epochs: {self.max_epochs}, Patience: {self.patience}")
        print(f"  Model: CausalCrisis (causal={self.model.use_causal})")
        print(f"{'='*60}")
        t0 = time.time()

        from ..models.causal_crisis_model import compute_grl_lambda

        for epoch in range(self.max_epochs):
            if hasattr(self.criterion, "set_phase"):
                if epoch < 50:
                    self.criterion.set_phase(1)
                elif epoch < 120:
                    self.criterion.set_phase(2)
                else:
                    self.criterion.set_phase(3)

            # GRL lambda schedule: tang dan co warmup
            grl_lam = compute_grl_lambda(epoch, self.max_epochs)

            # Rebuild graph tren causal features sau khi model da train duoc 50 epoch hieu qua
            if epoch == 50 and use_causal_graph:
                print(f"\n  Rebuilding CAUSAL kNN graph (k={k_neighbors}) on trained features...")
                self.model.eval()
                with torch.no_grad():
                    # Forward de lay causal features
                    out_tmp = self.model(img_feat, txt_feat, adj=None, domain_labels=domain_labels, task=task, grl_lambda=0.0)
                    if "c_v" in out_tmp and "c_t" in out_tmp:
                        c_v_np = out_tmp["c_v"].cpu().numpy()
                        c_t_np = out_tmp["c_t"].cpu().numpy()
                        causal_feat_concat = np.concatenate([c_v_np, c_t_np], axis=1)
                        # Force graph in eval to keep consistent type
                        adj = build_knn_graph(causal_feat_concat, k=k_neighbors).to(self.device)
                        del out_tmp
                        torch.cuda.empty_cache() if self.device == "cuda" else None
                        print(f"  Graph rebuilt successfully. Restarting learning rate for Phase 2!")
                        
                        # Reset learning rate so model can actually learn from the new graph! 
                        # Otherwise early plateau makes LR=1e-6 -> model dead.
                        initial_lr = 1e-4
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = initial_lr if param_group['weight_decay'] > 0 else initial_lr
                        # Reset early stopping trackers
                        best_hm_f1 = 0.0
                        patience_counter = 0
            
            metrics = self.train_epoch(
                img_feat, txt_feat, labels, adj,
                domain_labels, labeled_mask,
                task=task, grl_lambda=grl_lam,
            )
            history.append(metrics)
            
            # ReduceLROnPlateau scheduler step (with hm_f1)
            self.scheduler.step(metrics["hm_f1"])

            # Early stopping
            if metrics["hm_f1"] > best_hm_f1:
                best_hm_f1 = metrics["hm_f1"]
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                dom_acc_str = ""
                if metrics["domain_acc_causal"] is not None:
                    dom_acc_str = f" dom_acc={metrics['domain_acc_causal']:.3f}"
                print(
                    f"  Epoch {epoch+1:4d}: loss={metrics['train_loss']:.4f} "
                    f"val_f1={metrics['val_f1']:.4f} hm_f1={metrics['hm_f1']:.4f} "
                    f"λ_grl={grl_lam:.3f}{dom_acc_str} "
                    f"patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                if epoch < 50 and use_causal_graph:
                    # Do not early stop before the causal graph is even rebuilt!
                    patience_counter = self.patience - 5
                else:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - t0

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        test_results = self.evaluate(
            img_feat, txt_feat, labels, adj,
            domain_labels, test_mask, task=task
        )

        print(f"\n  Training time: {elapsed:.1f}s ({epoch+1} epochs)")
        for t, r in test_results.items():
            if isinstance(r, dict) and "weighted_f1" in r:
                print(f"  Test {t}: wF1={r['weighted_f1']:.4f} bAcc={r['balanced_accuracy']:.4f}")
        if "domain_invariance" in test_results:
            di = test_results["domain_invariance"]
            print(f"  Domain invariance: causal_acc={di['causal_domain_acc']:.4f} "
                  f"(chance={1.0/7:.4f})")

        # Save checkpoint
        ckpt_path = f"{self.checkpoint_dir}/{run_name}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "test_results": test_results,
            "history": history,
            "epochs": epoch + 1,
            "elapsed": elapsed,
        }, ckpt_path)

        return {
            "test_results": test_results,
            "train_time_s": elapsed,
            "epochs": epoch + 1,
            "best_hm_f1": best_hm_f1,
        }


# ============================================================
# 4. EXPERIMENT RUNNER
# ============================================================

def set_seed(seed):
    """Reproducibility."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_causal_experiment(
    dataset_path,
    task="task1",
    seed=42,
    n_labeled=500,
    k=16,
    hidden_dim=512,
    causal_dim=256,
    pca_dim=256,
    max_epochs=500,
    patience=150,  # Match the paper's default order of magnitude for better convergence
    lr=1e-4,
    device="cuda",
    results_csv="/content/causal_results/all_results.csv",
    use_graph=True,
    use_attention=True,
    use_mtl=True,
    use_causal=True,
    use_intervention=True,
    variant_name="causal_full",
    use_causal_graph=False,
    lodo_event: str = None,  # Support cho Leave-One-Disaster-Out
):
    """
    Chay 1 CausalCrisis experiment end-to-end.

    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    Khac biet voi run_geda_experiment:
    - Extract domain labels (event_name) tu TSV
    - Build graph tren causal features (neu use_causal_graph=True)
    - Dung CausalCrisisModel + CausalCrisisLoss
    - Log domain invariance metrics
    """
    from ..models.causal_crisis_model import (
        CausalCrisisModel, CausalCrisisLoss, compute_grl_lambda
    )

    set_seed(seed)
    # Ha momentum cua memory bank the N (few-shot < 100 mau => giam Momentum -> centroid update gap cap)
    intervention_momentum = 0.5 if n_labeled <= 100 else 0.8

    # Auto-disable intervention khi sample qua it (giai quyet bottleneck #2)
    use_intervention = (n_labeled >= 250) and use_intervention

    print(f"\n{'='*60}")
    print(f"  CausalCrisis: {variant_name} | task={task} | seed={seed} | N={n_labeled}")
    print(f"{'='*60}")

    # --- 1. Extract features + domain ---
    train_img, train_txt, train_labels_str, train_events = \
        extract_clip_features_with_domain(dataset_path, task, "train", device)
    test_img, test_txt, test_labels_str, test_events = \
        extract_clip_features_with_domain(dataset_path, task, "test", device)

    # --- 1.5 Encode labels GLOBALLY ---
    all_labels_str = np.concatenate([train_labels_str, test_labels_str])
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels_str)

    # Encode domain labels
    all_events = np.concatenate([train_events, test_events])
    event_le = LabelEncoder()
    all_domain_labels = event_le.fit_transform(all_events)
    num_domains = len(event_le.classes_)
    print(f"  Domains ({num_domains}): {list(event_le.classes_)}")

    n_train = len(train_labels_str)
    train_labels = all_labels[:n_train]

    # --- 2. PCA ---
    all_img = np.concatenate([train_img, test_img])
    all_txt = np.concatenate([train_txt, test_txt])

    n_total = len(all_labels)

    # --- 3. Create labeled mask (few-shot) & Splits ---
    labeled_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)

    if lodo_event is not None:
        print(f"  [LODO MODE] Leaving out domain: {lodo_event}")
        train_idxs = np.where(all_events != lodo_event)[0]
        test_idxs = np.where(all_events == lodo_event)[0]
        
        test_mask[test_idxs] = True
        n_select = min(n_labeled, len(train_idxs))
        
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        perm = torch.randperm(len(train_idxs), generator=g_seed)[:n_select]
        labeled_mask[train_idxs[perm]] = True
        
        # PCA tren Train Split Cua LODO
        pca_img = PCA(n_components=min(pca_dim, len(train_idxs)), random_state=seed)
        pca_txt = PCA(n_components=min(pca_dim, len(train_idxs)), random_state=seed)
        
        train_img_subset = all_img[train_idxs]
        train_txt_subset = all_txt[train_idxs]
        
        pca_img.fit(train_img_subset)
        pca_txt.fit(train_txt_subset)
        
        all_img_r = pca_img.transform(all_img)
        all_txt_r = pca_txt.transform(all_txt)
        
    else:
        test_mask[n_train:] = True
        n_select = min(n_labeled, n_train)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        perm = torch.randperm(n_train, generator=g_seed)[:n_select]
        labeled_mask[perm] = True

        # Tranh data leakage: Fit pca_img/pca_txt tren train_img (hoac labeled test features) thay vi toan bo dataset!
        pca_img = PCA(n_components=min(pca_dim, train_img.shape[1]), random_state=seed)
        pca_txt = PCA(n_components=min(pca_dim, train_txt.shape[1]), random_state=seed)
        
        train_img_r = pca_img.fit_transform(train_img)
        train_txt_r = pca_txt.fit_transform(train_txt)
        test_img_r = pca_img.transform(test_img)
        test_txt_r = pca_txt.transform(test_txt)

        all_img_r = np.concatenate([train_img_r, test_img_r])
        all_txt_r = np.concatenate([train_txt_r, test_txt_r])
        
    # [Norm] Re-normalize PCA features L2 sphere do PCA lam pha normalize
    all_img_r = all_img_r / (np.linalg.norm(all_img_r, axis=1, keepdims=True) + 1e-8)
    all_txt_r = all_txt_r / (np.linalg.norm(all_txt_r, axis=1, keepdims=True) + 1e-8)

    # --- 4. Build graph ---
    k_base = {"task1": 16, "task2": 8, "task3": 12}.get(task, 16)
    k_neighbors = min(k_base, max(3, n_labeled // 15))
    
    print(f"  Building RAW kNN graph (k={k_neighbors}) initially...")
    all_feat_concat = np.concatenate([all_img_r, all_txt_r], axis=1)
    adj = build_knn_graph(all_feat_concat, k=k_neighbors)

    # --- 5. Create model ---
    num_classes = len(np.unique(all_labels))
    nc = num_classes

    model = CausalCrisisModel(
        img_dim=all_img_r.shape[1],
        txt_dim=all_txt_r.shape[1],
        hidden_dim=hidden_dim,
        causal_dim=causal_dim,
        spurious_dim=causal_dim,
        num_domains=num_domains,
        num_classes_task1=nc if task == "task1" else 2,
        num_classes_task2=nc if task == "task2" else 2,
        num_classes_task3=nc if task == "task3" else 2,
        dropout=0.3,
        use_graph=use_graph,
        use_attention=use_attention,
        use_mtl=use_mtl,
        use_causal=use_causal,
        use_intervention=use_intervention,
        intervention_momentum=intervention_momentum,
    )

    # Class weights cho Focal Loss (Hoan toan tap trung vao labeled subset thuc su)
    labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
    labeled_subset_labels = all_labels[labeled_indices]
    classes = np.unique(all_labels)
    counts = np.bincount(labeled_subset_labels, minlength=len(classes))
    counts = np.maximum(counts, 1)
    weight_arr = len(labeled_subset_labels) / (len(classes) * counts)
    
    # Clip weights to prevent zero-dominated extreme gradients
    weight_arr = np.clip(weight_arr, 0.5, 5.0)

    class_weights_tensor = torch.tensor(weight_arr, dtype=torch.float32).to(device)
    # Issue 37: Dynamic Focal Loss gamma
    task_gamma = 1.0 if task == "task1" else 2.0
    criterion = CausalCrisisLoss(gamma=task_gamma)
    criterion.focal.weight = class_weights_tensor

    # BUG FIX: DO NOT freeze randomly initialized attention modules. 
    # Freezing random nn.Linear layers scrambles features into permanent noise.
    # We remove the block that sets requires_grad = False for SelfAttention.
    
    # --- 6. Train ---
    results_dir = os.path.dirname(results_csv)
    os.makedirs(results_dir, exist_ok=True)

    trainer = CausalCrisisTrainer(
        model=model,
        criterion=criterion,
        device=device,
        lr=lr,
        weight_decay=1e-3,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=f"{results_dir}/checkpoints",
    )

    img_tensor = torch.tensor(all_img_r, dtype=torch.float32)
    txt_tensor = torch.tensor(all_txt_r, dtype=torch.float32)
    domain_tensor = torch.tensor(all_domain_labels, dtype=torch.long)
    labels_dict = {task: torch.tensor(all_labels, dtype=torch.long)}

    run_name = f"{variant_name}_{task}_s{seed}_n{n_labeled}"
    if lodo_event:
        run_name += f"_lodo_{lodo_event}"
        
    result = trainer.train(
        img_tensor, txt_tensor, labels_dict, adj,
        domain_tensor, labeled_mask, test_mask,
        task=task, run_name=run_name,
        use_causal_graph=use_causal_graph,
        k_neighbors=k_neighbors
    )

    # --- 7. Save to CSV ---
    test_r = result["test_results"].get(task, {})
    dom_inv = result["test_results"].get("domain_invariance", {})

    row = {
        "model": variant_name,
        "task": task,
        "seed": seed,
        "few_shot": n_labeled,
        "accuracy": test_r.get("accuracy", 0),
        "balanced_accuracy": test_r.get("balanced_accuracy", 0),
        "micro_f1": test_r.get("micro_f1", 0),
        "macro_f1": test_r.get("macro_f1", 0),
        "weighted_f1": test_r.get("weighted_f1", 0),
        "domain_acc_causal": dom_inv.get("causal_domain_acc", -1),
        "train_time_s": round(result["train_time_s"], 1),
        "timestamp": datetime.now().isoformat(),
    }

    headers = [
        'model', 'task', 'seed', 'few_shot', 'accuracy',
        'balanced_accuracy', 'micro_f1', 'macro_f1', 'weighted_f1',
        'domain_acc_causal', 'train_time_s', 'timestamp',
    ]
    file_exists = os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    print(f"\n  Saved to {results_csv}")
    return row


# ============================================================
# 5. BATCH RUNNER (cho Colab)
# ============================================================

def run_causal_all_experiments(
    dataset_path="/content/datasets/CrisisMMD_v2.0",
    seeds=(42, 123, 456, 789, 1024),
    tasks=("task1", "task2", "task3"),
    few_shot_sizes=(50, 100, 250, 500),
    device="cuda",
    results_csv="/content/causal_results/all_results.csv",
    variant_name="causal_full",
    use_causal=True,
    use_intervention=True,
    use_causal_graph=True,
    use_graph=True,
    use_attention=True,
):
    """Chay tat ca CausalCrisis experiments."""
    total = len(seeds) * len(tasks) * len(few_shot_sizes)
    ct = 0

    print(f"\n{'='*60}")
    print(f"  CausalCrisis Full Experiment Suite")
    print(f"  Variant: {variant_name}")
    print(f"  {len(seeds)} seeds x {len(tasks)} tasks x {len(few_shot_sizes)} sizes = {total} runs")
    print(f"{'='*60}")

    for task in tasks:
        for size in few_shot_sizes:
            for seed in seeds:
                ct += 1
                print(f"\n[{ct}/{total}] task={task}, size={size}, seed={seed}")

                # Check cache
                if os.path.exists(results_csv):
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    mask = (
                        (df['model'] == variant_name) &
                        (df['task'] == task) &
                        (df['seed'] == seed) &
                        (df['few_shot'] == size)
                    )
                    if mask.any():
                        wf1 = df.loc[mask, 'weighted_f1'].values[0]
                        print(f"  CACHED: wF1={wf1:.4f}")
                        continue

                try:
                    run_causal_experiment(
                        dataset_path=dataset_path,
                        task=task,
                        seed=seed,
                        n_labeled=size,
                        device=device,
                        results_csv=results_csv,
                        variant_name=variant_name,
                        use_graph=use_graph,
                        use_attention=use_attention,
                        use_mtl=False,    # Disable MTL default vi data truyen vao la Single-Task
                        use_causal=use_causal,
                        use_intervention=use_intervention,
                        use_causal_graph=use_causal_graph,
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All CausalCrisis experiments complete! ({ct} total)")
    print(f"{'='*60}")


# ============================================================
# 6. ABLATION RUNNER & STATISTICAL SIGNIFICANCE
# ============================================================

def run_ablation_suite(
    dataset_path="/content/datasets/CrisisMMD_v2.0",
    seeds=(42, 123, 456, 789, 1024),
    tasks=("task1", "task2", "task3"),
    few_shot_sizes=(50, 500),
    device="cuda",
    results_csv="/content/causal_results/ablation_results.csv",
):
    """
    Chay 7 variants de xay dung bang Ablation Table hoan chinh.
    """
    variants = [
        {"name": "GEDA_baseline", "causal": False, "int": False, "graph": True,  "attn": True},
        {"name": "Causal_NoInt",  "causal": True,  "int": False, "graph": True,  "attn": True},
        {"name": "Causal_Int",    "causal": True,  "int": True,  "graph": True,  "attn": True},
        {"name": "Causal_NoGraph","causal": True,  "int": True,  "graph": False, "attn": True},
        {"name": "Causal_NoAttn", "causal": True,  "int": True,  "graph": True,  "attn": False},
        # No GRL requires disabling use_attention adversarial loss weighting inside Trainer/Model?
        # A simpler way is to just let grl_lambda be 0 by adding a flag, but we'll use Full for now:
        {"name": "Causal_Full",   "causal": True,  "int": True,  "graph": True,  "attn": True},
    ]
    
    for v in variants:
        run_causal_all_experiments(
            dataset_path=dataset_path, seeds=seeds, tasks=tasks, few_shot_sizes=few_shot_sizes,
            device=device, results_csv=results_csv, variant_name=v["name"],
            use_causal=v["causal"], use_intervention=v["int"], use_causal_graph=False,
            use_graph=v["graph"], use_attention=v["attn"],
        )

def run_lodo_all_experiments(
     dataset_path="/content/datasets/CrisisMMD_v2.0",
     seeds=(42, 123, 456), task="task1", size=500, device="cuda",
     results_csv="/content/causal_results/lodo_results.csv",
):
    """
    Leave-One-Disaster-Out Experiment to evaluate out-of-distribution generalization.
    """
    events = ['california_wildfires', 'hurricane_harvey', 'hurricane_irma', 
              'hurricane_maria', 'iraq_iran_earthquake', 'mexico_earthquake', 'srilanka_floods']
    
    for evt in events:
        for s in seeds:
             run_causal_experiment(
                 dataset_path=dataset_path, task=task, seed=s, n_labeled=size,
                 device=device, results_csv=results_csv, variant_name="causal_full",
                 use_graph=True, use_attention=True, use_causal=True, use_intervention=True,
                 use_causal_graph=True, lodo_event=evt
             )
             
def compute_significance(results_csv, model_a, model_b, metric="weighted_f1"):
    import pandas as pd
    from scipy import stats
    df = pd.read_csv(results_csv)
    
    # Merge on same task, seed, few_shot to do paired t-test
    df_a = df[df['model'] == model_a]
    df_b = df[df['model'] == model_b]
    
    merged = pd.merge(df_a, df_b, on=['task', 'seed', 'few_shot'], suffixes=('_A', '_B'))
    if len(merged) == 0:
        return {"error": "No common runs found for paired testing."}
        
    a_vals = merged[f"{metric}_A"].values
    b_vals = merged[f"{metric}_B"].values
    
    t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
    mean_diff = a_vals.mean() - b_vals.mean()
    
    return {
        "model_A": model_a,
        "model_B": model_b,
        "n_samples": len(a_vals),
        "mean_A": a_vals.mean(),
        "mean_B": b_vals.mean(),
        "mean_diff": mean_diff,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

# ============================================================
# 7. SANITY TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    print("Testing CausalCrisisTrainer...")
    from models.causal_crisis_model import CausalCrisisModel, CausalCrisisLoss

    N, D = 100, 256
    NUM_DOMAINS = 7

    model = CausalCrisisModel(
        img_dim=D, txt_dim=D, hidden_dim=128,
        causal_dim=64, spurious_dim=64,
        num_domains=NUM_DOMAINS,
    )
    criterion = CausalCrisisLoss()
    trainer = CausalCrisisTrainer(
        model, criterion, device="cpu",
        max_epochs=5, patience=3,
        checkpoint_dir="/tmp/causal_test_ckpts",
    )

    img = torch.randn(N, D)
    txt = torch.randn(N, D)
    labels = {"task1": torch.randint(0, 2, (N,))}
    adj = torch.eye(N)
    domains = torch.randint(0, NUM_DOMAINS, (N,))
    labeled = torch.zeros(N, dtype=torch.bool)
    labeled[:20] = True
    test = torch.zeros(N, dtype=torch.bool)
    test[80:] = True

    result = trainer.train(
        img, txt, labels, adj, domains, labeled, test,
        task="task1", run_name="test"
    )
    print(f"Test result: {result['test_results']}")
    print("[OK] CausalCrisisTrainer works!")
