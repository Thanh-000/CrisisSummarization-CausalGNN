import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.causalcrisis_v2 import CausalCrisisV2Model
from trainers.causal_crisis_trainer import extract_clip_features_with_domain
from trainers.causalcrisis_v2_trainer import Phase1Trainer, Phase2Trainer

def run_lodo_experiment(task="task2", data_path="/content/CrisisMMD_v2.0", epochs_p1=100, epochs_p2=40, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n============================================================")
    print(f"  CausalCrisis v2 LODO OOD (Phase 3a) | task={task} | seed={seed}")
    print(f"============================================================")

    # 1. Load All Data (Train + Test splits from IID are combined for LODO)
    train_img, train_txt, train_labels_str, train_events = extract_clip_features_with_domain(data_path, task, "train", device)
    test_img, test_txt, test_labels_str, test_events = extract_clip_features_with_domain(data_path, task, "test", device)

    all_img = np.concatenate([train_img, test_img], axis=0)
    all_txt = np.concatenate([train_txt, test_txt], axis=0)
    all_labels_str = np.concatenate([train_labels_str, test_labels_str], axis=0)
    all_events = np.concatenate([train_events, test_events], axis=0)

    # Label Encoding
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels_str)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    
    event_le = LabelEncoder()
    all_event_ids = event_le.fit_transform(all_events)
    domain_names = list(event_le.classes_)
    num_domains = len(domain_names)
    
    print(f"  Domains ({num_domains}): {domain_names}")
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Total samples: {len(all_labels)}")
    print(f"============================================================\n")

    # Metrics storage
    p1_f1_scores, p1_bacc_scores = [], []
    p2_f1_scores, p2_bacc_scores = [], []

    img_dim = all_img.shape[1]
    txt_dim = all_txt.shape[1]
    batch_size = 256

    # 2. Leave-One-Disaster-Out Loop
    for test_domain_idx, test_domain_name in enumerate(domain_names):
        print(f"\n>>> FOLD {test_domain_idx+1}/{num_domains}: Testing on [{test_domain_name}] (Hold-out Domain) <<<")
        
        # Split Data
        test_mask = (all_event_ids == test_domain_idx)
        train_mask = ~test_mask
        
        X_img_train, X_txt_train, y_train, d_train = all_img[train_mask], all_txt[train_mask], all_labels[train_mask], all_event_ids[train_mask]
        X_img_test, X_txt_test, y_test, d_test = all_img[test_mask], all_txt[test_mask], all_labels[test_mask], all_event_ids[test_mask]
        
        print(f"    Train samples: {len(y_train)} | Test samples: {len(y_test)}")
        if len(y_test) == 0:
            print("    Skipping: No test samples.")
            continue

        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_img_train, dtype=torch.float32),
            torch.tensor(X_txt_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(d_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_img_test, dtype=torch.float32),
            torch.tensor(X_txt_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
            torch.tensor(d_test, dtype=torch.long)
        )
        
        # Drop last to avoid batch of 1 ruining BN if any
        drop_last = len(y_train) % batch_size == 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ---------------------------------------------------------
        #  A. PHASE 1: MLP Baseline
        # ---------------------------------------------------------
        print(f"  --- Phase 1: MLP Baseline Training ---")
        model_p1 = CausalCrisisV2Model(
            img_dim=img_dim, txt_dim=txt_dim,
            hidden_dim=256, causal_dim=256, spurious_dim=256,
            num_domains=num_domains, num_classes=num_classes,
            dropout=0.3
        ).to(device)
        
        optimizer_p1 = torch.optim.AdamW(model_p1.parameters(), lr=1e-3, weight_decay=1e-4)
        trainer_p1 = Phase1Trainer(model_p1, optimizer_p1, device, max_epochs=epochs_p1)
        
        best_p1_bacc, best_p1_f1 = 0, 0
        patience_counter = 0
        patience_limit_p1 = 15
        
        for epoch in range(1, epochs_p1 + 1):
            # Tắt tqdm output cho lodo để đỡ rối log
            _, _ = trainer_p1.train_epoch(train_loader, epoch, use_mixup=True)
            _, test_f1, test_acc = trainer_p1.evaluate(test_loader)
            
            if test_acc > best_p1_bacc:
                best_p1_bacc = test_acc
                best_p1_f1 = test_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit_p1:
                break
                
        p1_f1_scores.append(best_p1_f1)
        p1_bacc_scores.append(best_p1_bacc)
        print(f"    -> Phase 1 Fold Best: bAcc={best_p1_bacc:.4f}, F1={best_p1_f1:.4f}")

        # ---------------------------------------------------------
        #  B. PHASE 2: Causal GNN (GAT + BA)
        # ---------------------------------------------------------
        print(f"  --- Phase 2: Causal GNN (GAT + BA) Training ---")
        model_p2 = CausalCrisisV2Model(
            img_dim=img_dim, txt_dim=txt_dim,
            hidden_dim=256, causal_dim=256, spurious_dim=256,
            num_domains=num_domains, num_classes=num_classes,
            dropout=0.5
        ).to(device)
        
        gnn_params, phase1_params = [], []
        for name, param in model_p2.named_parameters():
            if 'gnn' in name: gnn_params.append(param)
            else: phase1_params.append(param)
                
        optimizer_p2 = torch.optim.AdamW([
            {'params': phase1_params, 'weight_decay': 1e-5},
            {'params': gnn_params, 'weight_decay': 1e-3}
        ], lr=5e-4)
        
        trainer_p2 = Phase2Trainer(model_p2, optimizer_p2, device, max_epochs=epochs_p2, k_neighbors=5, memory_size=256, m_samples=4)
        trainer_p2.config_mode = "REVAMP"
        
        best_p2_bacc, best_p2_f1 = 0, 0
        patience_counter = 0
        patience_limit_p2 = 12
        
        for epoch in range(1, epochs_p2 + 1):
            _, _ = trainer_p2.train_epoch(train_loader, epoch, use_mixup=False)
            _, test_f1, test_acc = trainer_p2.evaluate(test_loader)
            
            if test_acc > best_p2_bacc:
                best_p2_bacc = test_acc
                best_p2_f1 = test_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit_p2:
                break
                
        p2_f1_scores.append(best_p2_f1)
        p2_bacc_scores.append(best_p2_bacc)
        print(f"    -> Phase 2 Fold Best: bAcc={best_p2_bacc:.4f}, F1={best_p2_f1:.4f}")

    # 3. Final LODO Report
    print(f"\n============================================================")
    print(f"  FINAL LODO PROTOCOL RESULTS ({num_domains} Folds)")
    print(f"============================================================")
    
    p1_f1_mean, p1_f1_std = np.mean(p1_f1_scores), np.std(p1_f1_scores)
    p1_bacc_mean, p1_bacc_std = np.mean(p1_bacc_scores), np.std(p1_bacc_scores)
    
    p2_f1_mean, p2_f1_std = np.mean(p2_f1_scores), np.std(p2_f1_scores)
    p2_bacc_mean, p2_bacc_std = np.mean(p2_bacc_scores), np.std(p2_bacc_scores)
    
    print(f"  [Phase 1 - MLP Baseline] (IID Backbone)")
    print(f"    Average bAcc: {p1_bacc_mean:.4f} ± {p1_bacc_std:.4f}")
    print(f"    Average F1:   {p1_f1_mean:.4f} ± {p1_f1_std:.4f}")
    print(f"")
    print(f"  [Phase 2 - Causal GNN] (Soft-attention GAT + Vector BA)")
    print(f"    Average bAcc: {p2_bacc_mean:.4f} ± {p2_bacc_std:.4f}")
    print(f"    Average F1:   {p2_f1_mean:.4f} ± {p2_f1_std:.4f}")
    print(f"============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-Disaster-Out (LODO) Evaluation Protocol")
    parser.add_argument("--task", type=str, default="task2", help="Task name")
    parser.add_argument("--data_path", type=str, default="/content/CrisisMMD_v2.0", help="Path to CrisisMMD dataset")
    parser.add_argument("--epochs_p1", type=int, default=100, help="Max epochs for Phase 1")
    parser.add_argument("--epochs_p2", type=int, default=40, help="Max epochs for Phase 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_lodo_experiment(
        task=args.task,
        data_path=args.data_path,
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        seed=args.seed
    )
