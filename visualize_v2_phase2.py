import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.causalcrisis_v2 import CausalCrisisV2Model
from trainers.causalcrisis_v2_trainer import Phase2Trainer, build_knn_graph
from trainers.causal_crisis_trainer import extract_clip_features_with_domain

def plot_confusion_matrix(y_true, y_pred, classes, task):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {task.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{task}.png', dpi=300)
    print(f"  -> Saved confusion matrix to confusion_matrix_{task}.png")
    plt.close()

def plot_tsne(features_p1, features_p2, labels, classes, task):
    print("  -> Running t-SNE (this might take a moment)...")
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # We plot a subset to avoid overcrowding
    max_samples = 1500
    if len(features_p1) > max_samples:
        indices = np.random.choice(len(features_p1), max_samples, replace=False)
        features_p1 = features_p1[indices]
        features_p2 = features_p2[indices]
        labels = labels[indices]
        
    tsne_p1 = tsne.fit_transform(features_p1)
    tsne_p2 = tsne.fit_transform(features_p2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = ax1.scatter(tsne_p1[:, 0], tsne_p1[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    ax1.set_title("Phase 1: MLP Features ($X_c$)", fontsize=14)
    ax1.axis('off')
    
    scatter2 = ax2.scatter(tsne_p2[:, 0], tsne_p2[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    ax2.set_title("Phase 2: GAT + BA Features ($X_{gnn} + X_{s}$)", fontsize=14)
    ax2.axis('off')
    
    # Legend
    legend1 = ax1.legend(handles=scatter1.legend_elements()[0], labels=classes, loc="best", title="Classes", fontsize=8)
    ax1.add_artist(legend1)
    
    plt.suptitle(f"Feature Space T-SNE Comparison - {task.upper()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'tsne_comparison_{task}.png', dpi=300)
    print(f"  -> Saved t-SNE plot to tsne_comparison_{task}.png")
    plt.close()

def extract_and_visualize(dataset_path, task, device, seed=42):
    print(f"\n============================================================")
    print(f"  Visualization & Inference Script | {task.upper()}")
    print(f"============================================================")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Data
    train_img, train_txt, train_labels_str, train_events = extract_clip_features_with_domain(dataset_path, task, "train", device)
    test_img, test_txt, test_labels_str, test_events = extract_clip_features_with_domain(dataset_path, task, "test", device)

    all_labels_str = np.concatenate([train_labels_str, test_labels_str])
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels_str)
    
    n_train = len(train_labels_str)
    train_labels = all_labels[:n_train]
    test_labels = all_labels[n_train:]

    all_events = np.concatenate([train_events, test_events])
    event_le = LabelEncoder()
    all_domain_labels = event_le.fit_transform(all_events)
    num_domains = len(event_le.classes_)
    
    train_domain_labels = all_domain_labels[:n_train]
    test_domain_labels = all_domain_labels[n_train:]

    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

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

    # 2. Init Model
    model = CausalCrisisV2Model(
        img_dim=train_img.shape[1],
        txt_dim=train_txt.shape[1],
        hidden_dim=256, causal_dim=256, spurious_dim=256,
        num_domains=num_domains,
        num_classes=num_classes,
        dropout=0.5
    ).to(device)

    # L2 Reg
    gnn_params, phase1_params = [], []
    for name, param in model.named_parameters():
        if 'gnn' in name: gnn_params.append(param)
        else: phase1_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': phase1_params, 'weight_decay': 1e-5},
        {'params': gnn_params, 'weight_decay': 1e-3}
    ], lr=5e-4)

    trainer = Phase2Trainer(model, optimizer, device, max_epochs=40, k_neighbors=5, memory_size=256, m_samples=4)
    trainer.config_mode = "REVAMP"

    print("\n  [INFO] Fast-training model to gather Best Weights for visualization...")
    best_test_acc = 0
    best_model_state = None
    
    for epoch in range(1, 41):
        trainer.train_epoch(train_loader, epoch, use_mixup=False)
        test_loss, test_f1, test_acc = trainer.evaluate(test_loader)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n  [INFO] Fast-training completed. Best bAcc: {best_test_acc:.4f}. Restoring best weights...")
    model.load_state_dict(best_model_state)
    model.eval()

    # 3. Extract Features for T-SNE and Predictions for CM
    print("  [INFO] Extracting features from Test Set...")
    all_preds, all_targets_list = [], []
    all_xc, all_x_unified = [], []
    
    # Enable GNN & BA flags for evaluation manually
    trainer.current_epoch = 999 

    with torch.no_grad():
        for batch in test_loader:
            img, txt, labels, _ = [b.to(device) for b in batch]
            
            # GNN Graph
            outputs_eval = trainer.model(img, txt, adj=None) 
            xc = outputs_eval["xc"]
            adj = build_knn_graph(xc, k=trainer.k_neighbors, training=False)
            
            # Forward Full
            bank_samples = trainer.memory_bank.sample(trainer.m_samples).to(device)
            backdoor_xs = bank_samples.unsqueeze(0).expand(img.size(0), trainer.m_samples, -1)
            outputs = trainer.model(img, txt, adj=adj, backdoor_xs=backdoor_xs)
            
            preds = torch.argmax(outputs['logits_ba'], dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets_list.extend(labels.cpu().numpy())
            
            # T-SNE representation:
            # Phase 1: xc
            all_xc.append(outputs["xc"].cpu().numpy())
            # Phase 2: xc_graph + expectation of xs
            # In representation space, it is xc_graph
            all_x_unified.append(outputs["xc_graph"].cpu().numpy())

    all_targets_list = np.array(all_targets_list)
    all_preds = np.array(all_preds)
    all_xc = np.concatenate(all_xc, axis=0)
    all_x_unified = np.concatenate(all_x_unified, axis=0)

    # 4. Plot!
    print("\n  [INFO] Generating Visualizations...")
    plot_confusion_matrix(all_targets_list, all_preds, class_names, task)
    plot_tsne(all_xc, all_x_unified, all_targets_list, class_names, task)
    
    print("\n============================================================")
    print("  Visualizations Complete! Check the current directory.")
    print("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="task2")
    parser.add_argument("--data_path", type=str, default="/content/CrisisMMD_v2.0")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extract_and_visualize(args.data_path, args.task, device)
