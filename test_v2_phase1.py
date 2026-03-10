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
from trainers.causalcrisis_v2_trainer import Phase1Trainer
from trainers.causal_crisis_trainer import extract_clip_features_with_domain

def run_phase1_experiment(dataset_path, task, seed, device, epochs=100):
    print(f"\n============================================================")
    print(f"  CausalCrisis v2 (Phase 1) | task={task} | seed={seed}")
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Initialize Model and Trainer
    model = CausalCrisisV2Model(
        img_dim=train_img.shape[1], # Raw CLIP features = 1024 or 512
        txt_dim=train_txt.shape[1], # Raw CLIP features = 768 or 512
        hidden_dim=256,
        causal_dim=256,
        spurious_dim=256,
        num_domains=num_domains,
        num_classes=len(le.classes_),
        dropout=0.3
    ).to(device)

    # Dùng optimizer AdamW với Weight Decay (giống CAMO/GEDA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # Slightly higher base LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Phase1Trainer(model, optimizer, device, max_epochs=epochs)

    # 5. Training Loop
    best_test_f1 = 0
    best_test_acc = 0
    patience_counter = 0
    patience_limit = 20

    print("\n  Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = trainer.train_epoch(train_loader, epoch, use_mixup=True)
        test_loss, test_f1, test_acc = trainer.evaluate(test_loader)
        
        scheduler.step() # Cập nhật Learning Rate
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            patience_counter = 0
            # Có thể lưu checkpoint model ở đây
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
    print(f"  Phase 1 Best Test F1:    {best_test_f1:.4f}")
    print(f"  Phase 1 Best Test bAcc:  {best_test_acc:.4f}")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 of CausalCrisis V2")
    parser.add_argument("--task", type=str, default="task1", help="Task name (task1, task2, task3)")
    parser.add_argument("--data_path", type=str, default="/content/CrisisMMD_v2.0", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_phase1_experiment(
        dataset_path=args.data_path,
        task=args.task,
        seed=seed,
        device=device,
        epochs=args.epochs
    )
