"""
============================================================================
CausalCrisis V3 — Google Colab Notebook Script
============================================================================

📋 Cách sử dụng (trên Google Colab):
1. Upload toàn bộ thư mục `src/` lên Colab hoặc clone repo
2. Mount Google Drive (nếu cần)
3. Chạy từng section theo thứ tự

📌 Requirements:
   pip install torch torchvision open_clip_torch scikit-learn pandas matplotlib seaborn

============================================================================
"""

# %%
# ============================================================================
# 📦 CELL 1: Setup & Installation
# ============================================================================
print("=" * 60)
print("🚀 CausalCrisis V3 — Setup")
print("=" * 60)

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install dependencies
packages = [
    "open_clip_torch",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
]

for pkg in packages:
    try:
        __import__(pkg.replace("-", "_").split("_")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import torch
import numpy as np
import os

print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Mount Google Drive (Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Google Drive mounted")
except ImportError:
    IN_COLAB = False
    print("ℹ️ Not in Colab — running locally")

# %%
# ============================================================================
# 📦 CELL 2: Clone Repository & Setup Paths
# ============================================================================
print("\n" + "=" * 60)
print("📂 Setting up paths")
print("=" * 60)

# === CẤU HÌNH ĐƯỜNG DẪN ===
# Thay đổi theo setup của bạn
REPO_URL = "https://github.com/Thanh-000/CrisisSummarization-CausalGNN.git"
DATASET_DIR = "/content/CrisisMMD_v2.0"  # Đường dẫn dataset CrisisMMD
PROJECT_DIR = "/content/CausalCrisisV3"
CACHE_DIR = "/content/cached_features"

# Clone repo nếu chưa có
if not os.path.exists(os.path.join(PROJECT_DIR, "src")):
    print("📥 Cloning repository...")
    os.system(f"git clone {REPO_URL} {PROJECT_DIR}")
else:
    print("✅ Repository already exists")

# Thêm src vào Python path
sys.path.insert(0, PROJECT_DIR)

# Import modules
from src.config import get_config, CausalCrisisConfig
from src.models import CausalCrisisV3, CLIPMLPBaseline
from src.losses import CausalCrisisLoss, FocalLoss
from src.data import (
    load_crisismmd_annotations,
    extract_and_cache_clip_features,
    create_stratified_splits,
    create_dataloaders,
    compute_class_weights,
)
from src.trainer import CausalCrisisTrainer, BaselineTrainer
from src.evaluate import (
    compute_metrics,
    plot_training_curves,
    plot_tsne_causal_features,
)

print("✅ All modules imported")

# %%
# ============================================================================
# 📦 CELL 3: Load CrisisMMD Dataset
# ============================================================================
print("\n" + "=" * 60)
print("📊 Loading CrisisMMD v2.0")
print("=" * 60)

# Kiểm tra dataset
assert os.path.exists(DATASET_DIR), (
    f"❌ Dataset not found at {DATASET_DIR}\n"
    f"   Download from: https://crisisnlp.qcri.org/crisismmd"
)

# Load annotations
data = load_crisismmd_annotations(DATASET_DIR, task="task1")
print(f"\nDataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# %%
# ============================================================================
# 📦 CELL 4: Extract & Cache CLIP Features
# ============================================================================
print("\n" + "=" * 60)
print("🔄 CLIP Feature Extraction (ViT-L/14)")
print("=" * 60)

config = get_config("task1")

image_features, text_features = extract_and_cache_clip_features(
    data=data,
    cache_dir=CACHE_DIR,
    model_name=config.clip.model_name,
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Labels & domains
labels = data["label"].values
domain_ids = data.get("domain_id", np.zeros(len(labels))).values.astype(int)

print(f"\n📐 Feature shapes:")
print(f"   Image: {image_features.shape}")
print(f"   Text:  {text_features.shape}")
print(f"   Labels: {labels.shape} (classes: {np.unique(labels)})")
print(f"   Domains: {np.unique(domain_ids)}")

# %%
# ============================================================================
# 📦 CELL 5: Create Data Splits
# ============================================================================
print("\n" + "=" * 60)
print("📊 Creating Stratified Splits")
print("=" * 60)

SEED = 42
train_idx, val_idx, test_idx = create_stratified_splits(
    labels, domain_ids,
    test_ratio=config.eval.test_split,
    val_ratio=config.eval.val_split,
    seed=SEED,
)

# Class weights cho Focal Loss
train_labels = labels[train_idx]
class_weights = compute_class_weights(train_labels)

# DataLoaders
loaders = create_dataloaders(
    image_features, text_features, labels, domain_ids,
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    batch_size=config.training.batch_size,
    num_workers=0,  # Colab tốt nhất dùng 0
)

print(f"✅ Train: {len(loaders['train'].dataset)}")
print(f"✅ Val:   {len(loaders['val'].dataset)}")
print(f"✅ Test:  {len(loaders['test'].dataset)}")

# %%
# ============================================================================
# 🧪 CELL 6: EXPERIMENT H1 — CLIP MLP Baseline
# ============================================================================
print("\n" + "=" * 60)
print("🧪 H1: CLIP ViT-L/14 + MLP Baseline")
print("   Prediction: F1 > 88%")
print("=" * 60)

from src.losses import FocalLoss

baseline_results = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for seed in config.training.seeds:
    print(f"\n--- Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Model
    baseline_model = CLIPMLPBaseline(
        input_dim=768*2,
        num_classes=config.classifier.num_classes,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )
    
    # Loss
    loss_fn = FocalLoss(alpha=class_weights, gamma=config.training.focal_gamma)
    
    # Train
    trainer = BaselineTrainer(
        baseline_model, optimizer, scheduler,
        device=DEVICE, save_dir="checkpoints"
    )
    history = trainer.train(
        loaders["train"], loaders["val"],
        epochs=100, patience=15,
        loss_fn=loss_fn,
    )
    
    # Eval on test
    baseline_model.load_state_dict(
        torch.load("checkpoints/baseline_best.pt", map_location=DEVICE)
    )
    baseline_model.eval().to(DEVICE)
    
    all_preds, all_labels_test, all_probs = [], [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            f_v = batch["image_features"].to(DEVICE)
            f_t = batch["text_features"].to(DEVICE)
            logits = baseline_model(f_v, f_t)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels_test.extend(batch["label"].numpy())
            all_probs.extend(torch.softmax(logits, -1).cpu().numpy())
    
    metrics = compute_metrics(np.array(all_labels_test), np.array(all_preds))
    baseline_results.append(metrics)
    print(f"   Test F1 = {metrics['f1_weighted']:.4f}, Acc = {metrics['accuracy']:.4f}")

# Summary
f1_scores = [r['f1_weighted'] for r in baseline_results]
print(f"\n{'='*60}")
print(f"📊 H1 Results: CLIP MLP Baseline")
print(f"   F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"   Best F1 = {np.max(f1_scores):.4f}")
h1_supported = np.mean(f1_scores) > 0.88
print(f"   H1 supported (>88%): {'✅ YES' if h1_supported else '❌ NO'}")
print(f"{'='*60}")

# %%
# ============================================================================
# 🧪 CELL 7: EXPERIMENT H2 — CausalCrisis V3 Full Model
# ============================================================================
print("\n" + "=" * 60)
print("🧪 H2/H3/H5: CausalCrisis V3 Enhanced")
print("   Target: >90% F1 (surpass CrisisSpot 90.9%)")
print("=" * 60)

v3_results = []

for seed in config.training.seeds:
    print(f"\n{'='*40}")
    print(f"Seed {seed}")
    print(f"{'='*40}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CausalCrisis V3 model
    model = CausalCrisisV3(
        input_dim=config.clip.image_dim,
        causal_dim=config.disentangle.causal_dim,
        spurious_dim=config.disentangle.spurious_dim,
        num_classes=config.classifier.num_classes,
        num_domains=config.training.num_domains,
        nhead=config.fusion.nhead,
        dropout=config.disentangle.dropout,
        use_ica_init=config.disentangle.use_ica_init,
        fusion_type=config.fusion.fusion_type,
        grl_lambda_max=config.training.grl_lambda_max,
        grl_warmup_epochs=config.training.grl_warmup_epochs,
    )
    
    print(f"📐 Model parameters: {model.get_trainable_params():,}")
    
    # Loss function (với adaptive weighting)
    loss_fn = CausalCrisisLoss(
        num_classes=config.classifier.num_classes,
        focal_gamma=config.training.focal_gamma,
        alpha_adv=config.training.alpha_adv,
        alpha_ortho=config.training.alpha_ortho,
        alpha_supcon=config.training.alpha_supcon,
        use_adaptive=config.training.use_adaptive_weights,
        class_weights=class_weights,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler (Cosine Warm Restarts)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.T_0,
        T_mult=config.training.T_mult,
        eta_min=config.training.eta_min,
    )
    
    # Trainer
    trainer = CausalCrisisTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        warmup_epochs=config.training.warmup_epochs,
        ba_start_epoch=config.training.ba_start_epoch,
        experiment_name=f"v3_seed{seed}",
    )
    
    # Train
    history = trainer.train(
        loaders["train"], loaders["val"],
        epochs=config.training.epochs,
        patience=config.training.early_stop_patience,
    )
    
    # Evaluate on test (with BA)
    test_metrics = trainer.evaluate(loaders["test"], use_ba=True)
    v3_results.append(test_metrics)
    
    print(f"\n📊 Test Results (seed {seed}):")
    print(f"   F1 = {test_metrics['f1']:.4f}")
    print(f"   Accuracy = {test_metrics['accuracy']:.4f}")
    print(f"   Precision = {test_metrics['precision']:.4f}")
    print(f"   Recall = {test_metrics['recall']:.4f}")

# Summary
f1_v3 = [r['f1'] for r in v3_results]
print(f"\n{'='*60}")
print(f"📊 CausalCrisis V3 Full Results")
print(f"   F1 = {np.mean(f1_v3):.4f} ± {np.std(f1_v3):.4f}")
print(f"   Best F1 = {np.max(f1_v3):.4f}")
print(f"   vs Baseline: Δ = {np.mean(f1_v3) - np.mean(f1_scores):+.4f}")
beat_crisisspot = np.mean(f1_v3) > 0.909
print(f"   Surpass CrisisSpot (90.9%): {'✅ YES' if beat_crisisspot else '❌ NO'}")
print(f"{'='*60}")

# %%
# ============================================================================
# 📊 CELL 8: Training Visualization
# ============================================================================
print("\n📊 Generating Visualizations...")

# Plot cuối cùng
if history:
    plot_training_curves(history, save_path="training_curves.png")

# %%
# ============================================================================
# 📊 CELL 9: t-SNE Visualization of Causal Features
# ============================================================================
print("\n📊 t-SNE Visualization of Causal Features...")

# Extract causal features từ test set
model.eval()
all_C_v, all_C_t, all_labels_viz, all_domains_viz = [], [], [], []

with torch.no_grad():
    for batch in loaders["test"]:
        f_v = batch["image_features"].to(DEVICE)
        f_t = batch["text_features"].to(DEVICE)
        
        output = model(f_v, f_t)
        all_C_v.append(output["C_v"].cpu().numpy())
        all_C_t.append(output["C_t"].cpu().numpy())
        all_labels_viz.extend(batch["label"].numpy())
        if "domain_id" in batch:
            all_domains_viz.extend(batch["domain_id"].numpy())

C_v_all = np.concatenate(all_C_v)
C_t_all = np.concatenate(all_C_t)
labels_viz = np.array(all_labels_viz)
domains_viz = np.array(all_domains_viz) if all_domains_viz else np.zeros(len(labels_viz))

plot_tsne_causal_features(
    C_v_all, C_t_all, labels_viz, domains_viz,
    save_path="tsne_causal_features.png"
)

# %%
# ============================================================================
# 📊 CELL 10: Statistical Comparison — V3 vs Baseline
# ============================================================================
from src.evaluate import paired_bootstrap_test

print("\n📊 Statistical Significance Test (Bootstrap)")
print("=" * 60)

# Lấy predictions từ best run
baseline_preds = baseline_results[0]["predictions"] if hasattr(baseline_results[0], 'get') else None
v3_preds = v3_results[0]["predictions"]
test_labels = v3_results[0]["labels"]

if baseline_preds is not None and len(baseline_preds) == len(v3_preds):
    sig_test = paired_bootstrap_test(
        test_labels, v3_preds, baseline_preds,
        n_bootstrap=10000,
    )
    
    print(f"   V3 F1:       {sig_test['score_a']:.4f}")
    print(f"   Baseline F1: {sig_test['score_b']:.4f}")
    print(f"   Difference:  {sig_test['observed_diff']:+.4f}")
    print(f"   p-value:     {sig_test['p_value']:.4f}")
    print(f"   95% CI:      [{sig_test['ci_lower']:+.4f}, {sig_test['ci_upper']:+.4f}]")
    print(f"   Significant: {'✅ YES (p<0.05)' if sig_test['significant'] else '❌ NO'}")

print("\n🎉 Experiment complete!")
