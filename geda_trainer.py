"""
GEDA Trainer -- Training loop cho Graph-Enhanced Differential Attention
=======================================================================
Pipeline: Extract CLIP features -> PCA -> Build kNN graph -> Train GEDA
Designed for Google Colab with crash-safe checkpointing.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ============================================================
# 1. DATASET: CrisisMMD with pre-extracted CLIP features
# ============================================================

class CrisisMMDFeatureDataset(Dataset):
    """Dataset that loads pre-extracted CLIP features from .npy files."""

    def __init__(self, img_feat, txt_feat, labels, task="task1"):
        self.img_feat = torch.tensor(img_feat, dtype=torch.float32)
        self.txt_feat = torch.tensor(txt_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.task = task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img": self.img_feat[idx],
            "txt": self.txt_feat[idx],
            "label": self.labels[idx],
            "idx": idx,
        }


# ============================================================
# 2. FEATURE EXTRACTION (CLIP + PCA)
# ============================================================

def extract_clip_features(dataset_path, task="task1", split="train", device="cuda"):
    """
    Extract CLIP features cho CrisisMMD.
    Luu cache de khong phai extract lai.
    """
    import open_clip
    from PIL import Image

    cache_dir = f"{dataset_path}/.cache/clip_features"
    os.makedirs(cache_dir, exist_ok=True)

    cache_img = f"{cache_dir}/{task}_{split}_img.npy"
    cache_txt = f"{cache_dir}/{task}_{split}_txt.npy"
    cache_lbl = f"{cache_dir}/{task}_{split}_labels.npy"

    if os.path.exists(cache_img) and os.path.exists(cache_txt):
        print(f"  Loading cached features: {task}/{split}")
        return np.load(cache_img), np.load(cache_txt), np.load(cache_lbl)

    print(f"  Extracting CLIP features: {task}/{split}")

    # Map task to TSV file
    task_map = {
        "task1": "task_informative_text_img",
        "task2": "task_humanitarian_text_img",
        "task3": "task_damage_text_img",
    }
    tsv_name = f"{task_map[task]}_{split}.tsv"
    # Try multiple locations for TSV files
    candidates = [
        f"{dataset_path}/{tsv_name}",                          # root
        f"{dataset_path}/crisismmd_datasplit_all/{tsv_name}",  # subdir
    ]
    tsv_path = None
    for c in candidates:
        if os.path.exists(c):
            tsv_path = c
            break
    if tsv_path is None:
        raise FileNotFoundError(f"TSV not found in: {candidates}")

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Parse TSV
    import pandas as pd
    df = pd.read_csv(tsv_path, sep='\t')

    # Columns: event_name, tweet_id, image_path, tweet_text, label, ...
    img_col = [c for c in df.columns if 'image' in c.lower() and 'label' not in c.lower()][0]
    txt_col = [c for c in df.columns if 'text' in c.lower() and 'label' not in c.lower() and 'llava' not in c.lower()][0]
    # Use 'label' column (NOT 'label_text_image' which is all same value)
    lbl_col = 'label' if 'label' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]

    # Label encoding
    label_map = {lbl: i for i, lbl in enumerate(sorted(df[lbl_col].unique()))}
    labels = df[lbl_col].map(label_map).values

    img_features = []
    txt_features = []

    with torch.no_grad():
        # Batch text
        batch_size = 64
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]

            # Text features
            texts = batch_df[txt_col].tolist()
            tokens = tokenizer(texts).to(device)
            t_feat = model.encode_text(tokens)
            t_feat = F.normalize(t_feat, dim=-1)
            txt_features.append(t_feat.cpu().numpy())

            # Image features
            batch_imgs = []
            for _, row in batch_df.iterrows():
                img_path = f"{dataset_path}/{row[img_col]}"
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = preprocess(img)
                except Exception:
                    img = torch.zeros(3, 224, 224)
                batch_imgs.append(img)

            img_batch = torch.stack(batch_imgs).to(device)
            i_feat = model.encode_image(img_batch)
            i_feat = F.normalize(i_feat, dim=-1)
            img_features.append(i_feat.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"    {i}/{len(df)}...")

    img_features = np.concatenate(img_features, axis=0)
    txt_features = np.concatenate(txt_features, axis=0)

    np.save(cache_img, img_features)
    np.save(cache_txt, txt_features)
    np.save(cache_lbl, labels)

    print(f"  Extracted {len(labels)} samples, {len(label_map)} classes")
    return img_features, txt_features, labels


def apply_pca(train_img, train_txt, test_img, test_txt, n_components=256):
    """PCA reduction giong Paper 1."""
    pca_img = PCA(n_components=n_components, random_state=42)
    pca_txt = PCA(n_components=n_components, random_state=42)

    train_img_r = pca_img.fit_transform(train_img)
    test_img_r = pca_img.transform(test_img)

    train_txt_r = pca_txt.fit_transform(train_txt)
    test_txt_r = pca_txt.transform(test_txt)

    return train_img_r, train_txt_r, test_img_r, test_txt_r


# ============================================================
# 3. GRAPH CONSTRUCTION (kNN + FAISS optional)
# ============================================================

def build_knn_graph(features, k=16, use_faiss=True):
    """
    Build k-NN adjacency matrix.
    Returns normalized adjacency: D^{-1}A.
    """
    n = features.shape[0]

    if use_faiss:
        try:
            import faiss
            d = features.shape[1]
            index = faiss.IndexFlatIP(d)  # inner product = cosine on normalized
            feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            index.add(feats_norm.astype(np.float32))
            _, indices = index.search(feats_norm.astype(np.float32), k + 1)
            indices = indices[:, 1:]  # remove self
        except ImportError:
            use_faiss = False

    if not use_faiss:
        nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        nn_model.fit(features)
        _, indices = nn_model.kneighbors(features)

    # Build sparse-ish adjacency
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in indices[i]:
            if j < n:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # undirected

    # Add self-loops
    adj += np.eye(n, dtype=np.float32)

    # Row-normalize: D^{-1}A
    row_sum = adj.sum(axis=1, keepdims=True)
    adj = adj / (row_sum + 1e-8)

    return torch.tensor(adj, dtype=torch.float32)


# ============================================================
# 4. TRAINING LOOP
# ============================================================

class GEDATrainer:
    """
    Training loop cho GEDA model.

    Features:
    - Full-graph training (nhu Paper 1)
    - Shuffle split (40% train / 60% val tu labeled data)
    - Early stopping voi harmonic mean F1
    - Crash-safe checkpointing
    - Multi-seed evaluation
    """

    def __init__(
        self,
        model,
        criterion,
        device="cuda",
        lr=1e-4,
        weight_decay=1e-3,
        max_epochs=500,
        patience=100,
        checkpoint_dir="/content/geda_results/checkpoints",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )

    def _shuffle_split(self, n_labeled, train_frac=0.4):
        """Paper 1 style: shuffle labeled indices moi epoch."""
        perm = torch.randperm(n_labeled)
        n_train = int(n_labeled * train_frac)
        return perm[:n_train], perm[n_train:]

    def train_epoch(self, img_feat, txt_feat, labels, adj_img, adj_txt,
                     labeled_mask, task="task1"):
        """Train 1 epoch voi shuffle split."""
        self.model.train()

        # Shuffle split trên labeled data
        labeled_idx = torch.where(labeled_mask)[0]
        n_labeled = len(labeled_idx)
        train_idx, val_idx = self._shuffle_split(n_labeled)
        train_idx = labeled_idx[train_idx]
        val_idx = labeled_idx[val_idx]

        # Forward pass (full graph)
        outputs = self.model(img_feat, txt_feat, adj_img, adj_txt, task=task)

        # Train loss (only on shuffle-train subset)
        train_losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for t, logits in outputs.items():
            if t in labels:
                loss = F.cross_entropy(logits[train_idx], labels[t][train_idx])
                total_loss = total_loss + loss
                train_losses[t] = loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Evaluate on val subset
        self.model.eval()
        with torch.no_grad():
            outputs_val = self.model(img_feat, txt_feat, adj_img, adj_txt, task=task)

        val_f1s = {}
        train_f1s = {}
        for t, logits in outputs_val.items():
            if t in labels:
                # Val F1
                pred_val = logits[val_idx].argmax(dim=-1).cpu().numpy()
                true_val = labels[t][val_idx].cpu().numpy()
                val_f1s[t] = f1_score(true_val, pred_val, average='weighted', zero_division=0)

                # Train F1
                pred_tr = logits[train_idx].argmax(dim=-1).cpu().numpy()
                true_tr = labels[t][train_idx].cpu().numpy()
                train_f1s[t] = f1_score(true_tr, pred_tr, average='weighted', zero_division=0)

        # Harmonic mean (Paper 1 style)
        avg_val_f1 = np.mean(list(val_f1s.values())) if val_f1s else 0
        avg_train_f1 = np.mean(list(train_f1s.values())) if train_f1s else 0
        if avg_val_f1 + avg_train_f1 > 0:
            hm_f1 = 2 * avg_val_f1 * avg_train_f1 / (avg_val_f1 + avg_train_f1)
        else:
            hm_f1 = 0

        return {
            "train_loss": total_loss.item(),
            "val_f1": avg_val_f1,
            "train_f1": avg_train_f1,
            "hm_f1": hm_f1,
            "val_f1s": val_f1s,
        }

    @torch.no_grad()
    def evaluate(self, img_feat, txt_feat, labels, adj_img, adj_txt,
                  test_mask, task="task1"):
        """Evaluate on test set."""
        self.model.eval()
        outputs = self.model(img_feat, txt_feat, adj_img, adj_txt, task=task)

        test_idx = torch.where(test_mask)[0]
        results = {}

        for t, logits in outputs.items():
            if t in labels:
                preds = logits[test_idx].argmax(dim=-1).cpu().numpy()
                trues = labels[t][test_idx].cpu().numpy()

                results[t] = {
                    "accuracy": float(accuracy_score(trues, preds)),
                    "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
                    "weighted_f1": float(f1_score(trues, preds, average='weighted', zero_division=0)),
                    "macro_f1": float(f1_score(trues, preds, average='macro', zero_division=0)),
                    "micro_f1": float(f1_score(trues, preds, average='micro', zero_division=0)),
                }

        return results

    def train(self, img_feat, txt_feat, labels, adj_img, adj_txt,
              labeled_mask, test_mask, task="task1", run_name="geda"):
        """
        Full training loop voi early stopping.

        Args:
            img_feat: (N, D) all features (labeled + unlabeled + test)
            txt_feat: (N, D) text features
            labels: dict {"task1": (N,), "task2": ...}
            adj_img, adj_txt: (N, N) adjacency matrices
            labeled_mask: (N,) bool, True for labeled samples
            test_mask: (N,) bool, True for test samples
            task: which task(s) to train
            run_name: for checkpoint naming
        """
        # Move to device
        img_feat = img_feat.to(self.device)
        txt_feat = txt_feat.to(self.device)
        adj_img = adj_img.to(self.device)
        adj_txt = adj_txt.to(self.device)
        labels = {t: l.to(self.device) for t, l in labels.items()}
        labeled_mask = labeled_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        best_hm_f1 = 0
        patience_counter = 0
        best_state = None
        history = []

        print(f"\n{'='*50}")
        print(f"  Training: {run_name}")
        print(f"  Labeled: {labeled_mask.sum().item()}, Test: {test_mask.sum().item()}")
        print(f"  Max epochs: {self.max_epochs}, Patience: {self.patience}")
        print(f"{'='*50}")

        t0 = time.time()

        for epoch in range(self.max_epochs):
            metrics = self.train_epoch(
                img_feat, txt_feat, labels, adj_img, adj_txt,
                labeled_mask, task=task
            )
            history.append(metrics)

            # Early stopping on harmonic mean F1
            if metrics["hm_f1"] > best_hm_f1:
                best_hm_f1 = metrics["hm_f1"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:4d}: loss={metrics['train_loss']:.4f} "
                      f"val_f1={metrics['val_f1']:.4f} hm_f1={metrics['hm_f1']:.4f} "
                      f"patience={patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        elapsed = time.time() - t0

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation on test set
        test_results = self.evaluate(
            img_feat, txt_feat, labels, adj_img, adj_txt,
            test_mask, task=task
        )

        print(f"\n  Training time: {elapsed:.1f}s ({epoch+1} epochs)")
        for t, r in test_results.items():
            print(f"  Test {t}: wF1={r['weighted_f1']:.4f} bAcc={r['balanced_accuracy']:.4f}")

        # Save checkpoint
        ckpt_path = f"{self.checkpoint_dir}/{run_name}.pt"
        torch.save({
            "model_state": self.model.state_dict(),
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
# 5. MULTI-SEED RUNNER
# ============================================================

def set_seed(seed):
    """Reproducibility."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_geda_experiment(
    dataset_path,
    task="task1",
    seed=42,
    n_labeled=500,
    k=16,
    hidden_dim=512,
    pca_dim=256,
    max_epochs=500,
    patience=100,
    lr=1e-4,
    device="cuda",
    results_csv="/content/geda_results/all_results.csv",
    use_graph=True,
    use_attention=True,
    use_mtl=False,
    variant_name="geda_full",
):
    """
    Chay 1 GEDA experiment end-to-end.

    Returns dict voi metrics.
    """
    from geda_model import GEDAModel, GEDALoss

    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  GEDA Experiment: {variant_name} | task={task} | seed={seed} | N={n_labeled}")
    print(f"{'='*60}")

    # --- 1. Extract features ---
    train_img, train_txt, train_labels = extract_clip_features(dataset_path, task, "train", device)
    test_img, test_txt, test_labels = extract_clip_features(dataset_path, task, "test", device)

    # --- 2. PCA ---
    all_img = np.concatenate([train_img, test_img], axis=0)
    all_txt = np.concatenate([train_txt, test_txt], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    pca_img = PCA(n_components=min(pca_dim, all_img.shape[1]), random_state=seed)
    pca_txt = PCA(n_components=min(pca_dim, all_txt.shape[1]), random_state=seed)

    all_img_r = pca_img.fit_transform(all_img)
    all_txt_r = pca_txt.fit_transform(all_txt)

    n_train = len(train_labels)
    n_total = len(all_labels)

    # --- 3. Create labeled mask (few-shot) ---
    labeled_mask = torch.zeros(n_total, dtype=torch.bool)
    # Select n_labeled from train set
    n_select = min(n_labeled, n_train)
    perm = torch.randperm(n_train)[:n_select]
    labeled_mask[perm] = True

    test_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask[n_train:] = True

    # --- 4. Build graph ---
    print(f"  Building kNN graph (k={k})...")
    all_feat_concat = np.concatenate([all_img_r, all_txt_r], axis=1)
    adj = build_knn_graph(all_feat_concat, k=k)

    # --- 5. Create model ---
    # Num classes
    num_classes = len(np.unique(all_labels))
    task_num_classes = {"task1": 2, "task2": 6, "task3": 3}
    nc = task_num_classes.get(task, num_classes)

    model = GEDAModel(
        img_dim=all_img_r.shape[1],
        txt_dim=all_txt_r.shape[1],
        hidden_dim=hidden_dim,
        num_classes_task1=nc if task == "task1" else 2,
        num_classes_task2=nc if task == "task2" else 6,
        num_classes_task3=nc if task == "task3" else 3,
        dropout=0.3,
        use_graph=use_graph,
        use_attention=use_attention,
        use_mtl=use_mtl,
    )

    criterion = GEDALoss()

    # --- 6. Train ---
    trainer = GEDATrainer(
        model=model,
        criterion=criterion,
        device=device,
        lr=lr,
        weight_decay=1e-3,
        max_epochs=max_epochs,
        patience=patience,
    )

    img_tensor = torch.tensor(all_img_r, dtype=torch.float32)
    txt_tensor = torch.tensor(all_txt_r, dtype=torch.float32)
    labels_dict = {task: torch.tensor(all_labels, dtype=torch.long)}

    run_name = f"{variant_name}_{task}_s{seed}_n{n_labeled}"
    result = trainer.train(
        img_tensor, txt_tensor, labels_dict, adj, adj,
        labeled_mask, test_mask, task=task, run_name=run_name
    )

    # --- 7. Save to CSV ---
    test_r = result["test_results"].get(task, {})
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
        "train_time_s": round(result["train_time_s"], 1),
        "timestamp": datetime.now().isoformat(),
    }

    headers = ['model', 'task', 'seed', 'few_shot', 'accuracy', 'balanced_accuracy',
               'micro_f1', 'macro_f1', 'weighted_f1', 'train_time_s', 'timestamp']
    file_exists = os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    print(f"\n  Saved to {results_csv}")
    return row


# ============================================================
# 6. BATCH RUNNER (cho Colab notebook)
# ============================================================

def run_geda_all_experiments(
    dataset_path="/content/datasets/CrisisMMD_v2.0",
    seeds=(42, 123, 456, 789, 1024),
    tasks=("task1", "task2", "task3"),
    few_shot_sizes=(50, 100, 250, 500),
    device="cuda",
    results_csv="/content/geda_results/all_results.csv",
):
    """Chay tat ca GEDA experiments."""
    total = len(seeds) * len(tasks) * len(few_shot_sizes)
    ct = 0

    print(f"\n{'='*60}")
    print(f"  GEDA Full Experiment Suite")
    print(f"  {len(seeds)} seeds x {len(tasks)} tasks x {len(few_shot_sizes)} sizes = {total} runs")
    print(f"{'='*60}")

    for task in tasks:
        for size in few_shot_sizes:
            for seed in seeds:
                ct += 1
                print(f"\n[{ct}/{total}] task={task}, size={size}, seed={seed}")

                # Check if already done
                if os.path.exists(results_csv):
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    mask = (
                        (df['model'] == 'geda_full') &
                        (df['task'] == task) &
                        (df['seed'] == seed) &
                        (df['few_shot'] == size)
                    )
                    if mask.any():
                        wf1 = df.loc[mask, 'weighted_f1'].values[0]
                        print(f"  CACHED: wF1={wf1:.4f}")
                        continue

                try:
                    run_geda_experiment(
                        dataset_path=dataset_path,
                        task=task,
                        seed=seed,
                        n_labeled=size,
                        device=device,
                        results_csv=results_csv,
                        variant_name="geda_full",
                        use_graph=True,
                        use_attention=True,
                        use_mtl=True,
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"  All GEDA experiments complete! ({ct} total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Quick sanity test with dummy data
    print("Testing GEDATrainer...")
    from geda_model import GEDAModel, GEDALoss

    N, D = 100, 256
    model = GEDAModel(img_dim=D, txt_dim=D, hidden_dim=128)
    criterion = GEDALoss()
    trainer = GEDATrainer(model, criterion, device="cpu", max_epochs=5, patience=3)

    img = torch.randn(N, D)
    txt = torch.randn(N, D)
    labels = {"task1": torch.randint(0, 2, (N,))}
    adj = torch.eye(N)
    labeled = torch.zeros(N, dtype=torch.bool)
    labeled[:20] = True
    test = torch.zeros(N, dtype=torch.bool)
    test[80:] = True

    result = trainer.train(img, txt, labels, adj, adj, labeled, test,
                           task="task1", run_name="test")
    print(f"Test result: {result['test_results']}")
    print("[OK] Trainer works!")
