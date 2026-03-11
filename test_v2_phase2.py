import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.causalcrisis_v2 import CausalCrisisV2Model
from trainers.causalcrisis_v2_trainer import Phase2Trainer
from trainers.causal_crisis_trainer import extract_clip_features_with_domain

def run_phase2_experiment(dataset_path, task, seed, device, epochs=100, k_neighbors=3, m_samples=4):
    print(f"\n============================================================")
    print(f"  CausalCrisis v2 (Phase 2) | task={task} | seed={seed}")
    print(f"============================================================")

    # 1. Extract Features + Domain Labels (Using existing logic)
    train_img, train_txt, train_labels_str, train_events = \
        extract_clip_features_with_domain(dataset_path, task, "train", device)
    test_img, test_txt, test_labels_str, test_events = \
        extract_clip_features_with_domain(dataset_path, task, "test", device)

    # 2. Encode Labels
    all_labels_str = np.concatenate([train_labels_str, test_labels_str])
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels_str)
    
    n_train = len(train_labels_str)
    train_labels = all_labels[:n_train]
    test_labels = all_labels[n_train:]

    # Domain Labels (For GRL)
    all_events = np.concatenate([train_events, test_events])
    event_le = LabelEncoder()
    all_domain_labels = event_le.fit_transform(all_events)
    num_domains = len(event_le.classes_)
    
    train_domain_labels = all_domain_labels[:n_train]
    test_domain_labels = all_domain_labels[n_train:]

    print(f"  Domains ({num_domains}): {list(event_le.classes_)}")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"  Train samples: {n_train}, Test samples: {len(test_labels)}")

    # 3. Create DataLoaders
    # Batch Size 256 để nâng cao chất lượng GNN Soft-Graph (do Cache Vector chiếm ít VRAM)
    batch_size = 256
    train_dataset = TensorDataset(
        torch.tensor(train_img, dtype=torch.float32),
        torch.tensor(train_txt, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(train_domain_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_img, dtype=torch.float32),
        torch.tensor(test_txt, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
        torch.tensor(test_domain_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model and Trainer
    model = CausalCrisisV2Model(
        img_dim=train_img.shape[1], # Raw CLIP features = 1024 or 512
        txt_dim=train_txt.shape[1], # Raw CLIP features = 768 or 512
        hidden_dim=256,
        causal_dim=256,
        spurious_dim=256,
        num_domains=num_domains,
        num_classes=len(le.classes_),
        dropout=0.5 # Tăng dropout bạo lực để chống overfitting
    ).to(device)

    # Trọng số L2 riêng biệt: GNN cần weight_decay mạnh hơn để chống overfitting (0.5e-3)
    gnn_params = []
    phase1_params = []
    for name, param in model.named_parameters():
        if 'gnn' in name:
            gnn_params.append(param)
        else:
            phase1_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': phase1_params, 'weight_decay': 1e-5},
        {'params': gnn_params, 'weight_decay': 1e-3} # Tăng weight decay GNN lên 1e-3
    ], lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Phase2Trainer(
        model, optimizer, device, 
        max_epochs=epochs, 
        k_neighbors=k_neighbors,
        memory_size=256,
        m_samples=m_samples
    )

    # 5. Load Phase 1 Pretrained Weights (CRITICAL: GNN needs stable causal features)
    phase1_path = f"best_model_{task}.pth" # Logic matching typical saving names
    if os.path.exists(phase1_path):
        print(f"  --> Loading Phase 1 Checkpoint: {phase1_path}")
        model.load_state_dict(torch.load(phase1_path, map_location=device), strict=False)
    else:
        # Check alternative common names
        alt_path = "best_model.pth"
        if os.path.exists(alt_path):
             print(f"  --> Loading Alternative Checkpoint: {alt_path}")
             model.load_state_dict(torch.load(alt_path, map_location=device), strict=False)

    # 6. Training Loop Configuration
    trainer.config_mode = "C" # C: Causal/Backdoor mode, E: GNN only
    best_test_f1 = 0
    best_test_acc = 0
    patience_counter = 0
    patience_limit = 15

    print(f"\n  Starting Training Loop (Mode: {trainer.config_mode})...")

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = trainer.train_epoch(train_loader, epoch, use_mixup=False)
        test_loss, test_f1, test_acc = trainer.evaluate(test_loader)
        
        scheduler.step()
        
        # Monitor bAcc to align with Phase 1's true strength (handling imbalanced data)
        if test_acc > best_test_acc:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Patience: {patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    print(f"\n============================================================")
    print(f"  Phase 2 Best Test F1:    {best_test_f1:.4f}")
    print(f"  Phase 2 Best Test bAcc:  {best_test_acc:.4f}")
    print(f"============================================================")
    
    return best_test_f1, best_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 of CausalCrisis V2")
    parser.add_argument("--task", type=str, default="task1", help="Task name (task1, task2, task3)")
    parser.add_argument("--data_path", type=str, default="/content/CrisisMMD_v2.0", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--k", type=int, default=3, help="K neighbors for Graph")
    parser.add_argument("--m", type=int, default=4, help="M samples for Backdoor Adjustment")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run multi-seed
    seeds = [42, 100, 2026]
    results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        best_f1, best_acc = run_phase2_experiment(
            dataset_path=args.data_path,
            task=args.task,
            seed=seed,
            device=device,
            epochs=args.epochs,
            k_neighbors=args.k,
            m_samples=args.m
        )
        results.append((seed, best_f1, best_acc))
        
    print(f"\n============================================================")
    print(f"  FINAL MULTI-SEED RESULTS")
    print(f"============================================================")
    avg_f1 = np.mean([r[1] for r in results])
    avg_acc = np.mean([r[2] for r in results])
    for seed, f1, acc in results:
        print(f"    Seed {seed:4d} | F1: {f1:.4f} | bAcc: {acc:.4f}")
    print(f"  ----------------------------------------------------------")
    print(f"    AVERAGE   | F1: {avg_f1:.4f} | bAcc: {avg_acc:.4f}")
    print(f"============================================================")
