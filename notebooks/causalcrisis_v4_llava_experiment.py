"""
CausalCrisis V4 — LLaVA + Guided Cross-Attention Experiment
Google Colab Notebook Script

Hypothesis Testing:
  H1: LLaVA captions → +10% F1 over frozen CLIP baseline
  H2: Guided CA > standard CrossAttention
  H6: Full pipeline → 93%+ F1w

Based on: Munia et al. (CVPRw 2025) — 92.89% F1w SOTA
"""

# %%
# ============================================================================
# CELL 1: Setup & Installation
# ============================================================================
print("=" * 60)
print("CausalCrisis V4 — LLaVA + Guided CA Experiment")
print("=" * 60)

import subprocess
import sys
import time


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


# Cài packages cần thiết
packages_to_install = [
    "open_clip_torch",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "transformers",
    "accelerate",
    "bitsandbytes",
]

for pkg in packages_to_install:
    try:
        mod = pkg.replace("-", "_").replace("_torch", "").split("==")[0]
        if pkg == "open_clip_torch":
            mod = "open_clip"
        elif pkg == "scikit-learn":
            mod = "sklearn"
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# LLaVA cần install riêng
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    print("✅ LLaVA-Next already available")
except ImportError:
    print("Installing LLaVA dependencies...")
    install("transformers>=4.40.0")
    install("accelerate")

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report
)

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {total_memory / 1e9:.1f} GB")

try:
    import google.colab
    IN_COLAB = True
    print("✅ Colab detected")
except ImportError:
    IN_COLAB = False
    print("Not in Colab — running locally")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ============================================================================
# CELL 2: Clone repo & Setup paths
# ============================================================================
print("\n" + "=" * 60)
print("📂 Setting up project")
print("=" * 60)

REPO_URL = "https://github.com/Thanh-000/CrisisSummarization-CausalGNN.git"
BRANCH = "v3-causalcrisis-enhanced"
PROJECT_DIR = "/content/CrisisSummarization" if IN_COLAB else "."

if IN_COLAB and not os.path.exists(PROJECT_DIR):
    os.system(f"git clone -b {BRANCH} {REPO_URL} {PROJECT_DIR}")
elif IN_COLAB:
    os.system(f"cd {PROJECT_DIR} && git pull origin {BRANCH}")

if IN_COLAB:
    sys.path.insert(0, PROJECT_DIR)

# Import project modules (V4 refactored)
from src.config import V4Config, get_config
from src.data import (
    load_crisismmd_annotations,
    extract_and_cache_clip_features,
    CrisisMMDDataset,
    CrisisMMD3ModalDataset,
    create_dataloaders,
    create_3modal_loaders,
    compute_class_weights,
)
from src.models import (
    GuidedCrossAttention,
    ThreeModalityClassifier,
    CLIPMLPBaseline,
)
from src.losses import FocalLoss
from src.trainer import GenericTrainer

config = get_config("task1", version="v4")
print(f"✅ Config loaded: {config.experiment_name} (v4)")

# %%
# ============================================================================
# CELL 3: Load CrisisMMD Data (Official Splits matching Munia et al.)
# ============================================================================
print("\n" + "=" * 60)
print("📊 Loading CrisisMMD with OFFICIAL splits")
print("=" * 60)

# Đường dẫn dataset  
DATA_DIR = os.path.join(PROJECT_DIR, "CrisisMMD_v2.0")

if not os.path.exists(DATA_DIR):
    print("⚠️ CrisisMMD not found. Downloading...")
    # TODO: Thêm download script hoặc manual upload
    print("Please upload CrisisMMD_v2.0 to:", DATA_DIR)
else:
    print(f"✅ Found CrisisMMD at {DATA_DIR}")

# Load annotations — giữ original train/dev/test splits
data = load_crisismmd_annotations(DATA_DIR, task="task1")

# === CRITICAL: Sử dụng official splits thay vì random split ===
# Munia et al. splits: 9599 train / 1573 val / 1534 test
# CrisisMMD official splits theo 'split' column
if "split" in data.columns:
    train_mask = data["split"] == "train"
    val_mask = data["split"] == "dev"
    test_mask = data["split"] == "test"
    
    train_indices = np.where(train_mask.values)[0]
    val_indices = np.where(val_mask.values)[0]
    test_indices = np.where(test_mask.values)[0]
    
    print(f"\n📊 Using OFFICIAL splits:")
    print(f"   Train: {len(train_indices)}")
    print(f"   Val:   {len(val_indices)}")
    print(f"   Test:  {len(test_indices)}")
else:
    # Fallback — random stratified split
    from src.data import create_stratified_splits
    train_indices, val_indices, test_indices = create_stratified_splits(
        data["label"].values,
        data.get("domain_id", pd.Series(dtype=int)).values if "domain_id" in data.columns else None,
        seed=42,
    )
    print("⚠️ Using random stratified splits (official splits not available)")

labels = data["label"].values
domain_ids = data["domain_id"].values if "domain_id" in data.columns else None
class_weights = compute_class_weights(labels[train_indices])

print(f"\n📊 Class distribution (train):")
for cls, count in Counter(labels[train_indices]).items():
    print(f"   Class {cls}: {count} ({count/len(train_indices)*100:.1f}%)")

# %%
# ============================================================================
# CELL 4: CLIP Feature Extraction (Image + Text)
# ============================================================================
print("\n" + "=" * 60)
print("🔄 CLIP Feature Extraction")
print("=" * 60)

CACHE_DIR = os.path.join(PROJECT_DIR, "cached_features")
os.makedirs(CACHE_DIR, exist_ok=True)

image_features, text_features = extract_and_cache_clip_features(
    data,
    cache_dir=CACHE_DIR,
    model_name="ViT-L/14",
    batch_size=64,
    device=DEVICE,
)

print(f"✅ Image features: {image_features.shape}")
print(f"✅ Text features:  {text_features.shape}")

# %%
# ============================================================================
# CELL 5: LLaVA Caption Generation 🆕
# ============================================================================
print("\n" + "=" * 60)
print("🤖 LLaVA Caption Generation")
print("=" * 60)

LLAVA_CACHE = os.path.join(CACHE_DIR, "llava_captions.json")
LLAVA_CHECKPOINT = os.path.join(CACHE_DIR, "llava_captions_checkpoint.json")

if os.path.exists(LLAVA_CACHE):
    print(f"Loading cached LLaVA captions from {LLAVA_CACHE}")
    with open(LLAVA_CACHE, "r", encoding="utf-8") as f:
        llava_captions = json.load(f)
    print(f"   Loaded {len(llava_captions)} captions")
else:
    print("Generating LLaVA captions (this takes 2-4 hours on A100)...")
    
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from PIL import Image
    
    # Dùng LLaVA-Next (LLaVA-1.6) — nhe hon va chat luong tot
    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    print(f"   Loading model: {MODEL_ID}")
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,  # 4-bit quantization de tiet kiem VRAM
    )
    
    # Prompt toi uu cho crisis classification
    LLAVA_PROMPT = (
        "[INST] <image>\n"
        "Describe this social media image in detail. "
        "Focus on: (1) what objects, people, or structures are visible, "
        "(2) any signs of damage, disaster, or emergency, "
        "(3) the overall scene and context. "
        "Be specific and factual. [/INST]"
    )
    
    # Resume from checkpoint neu co
    if os.path.exists(LLAVA_CHECKPOINT):
        with open(LLAVA_CHECKPOINT, "r", encoding="utf-8") as f:
            llava_captions = json.load(f)
        print(f"   Resuming from checkpoint: {len(llava_captions)} captions done")
    else:
        llava_captions = {}
    
    total = len(data)
    batch_errors = 0
    start_time = time.time()
    
    for idx in range(total):
        # Skip neu da co trong checkpoint
        if str(idx) in llava_captions:
            continue
        
        image_path = data.iloc[idx]["image_path"]
        
        try:
            img = Image.open(image_path).convert("RGB")
            
            inputs = processor(
                text=LLAVA_PROMPT,
                images=img,
                return_tensors="pt",
            ).to(llava_model.device)
            
            with torch.no_grad():
                output_ids = llava_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,  # Deterministic
                    temperature=1.0,
                )
            
            # Decode — chi lay phan generated (bo prompt)
            generated = processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            llava_captions[str(idx)] = generated
            
        except Exception as e:
            # Fallback: dung tweet text goc neu LLaVA loi
            llava_captions[str(idx)] = str(data.iloc[idx].get("text", ""))
            batch_errors += 1
        
        # Checkpoint moi 500 anh
        if (idx + 1) % 500 == 0:
            with open(LLAVA_CHECKPOINT, "w", encoding="utf-8") as f:
                json.dump(llava_captions, f, ensure_ascii=False)
            elapsed = time.time() - start_time
            done = sum(1 for k in llava_captions if k.isdigit())
            rate = max(done, 1) / max(elapsed, 1)
            remaining = (total - done) / max(rate, 0.01) / 60
            print(f"   [{done}/{total}] {rate:.1f} img/s, ~{remaining:.0f}min remaining, errors={batch_errors}")
    
    # Save final cache
    with open(LLAVA_CACHE, "w", encoding="utf-8") as f:
        json.dump(llava_captions, f, ensure_ascii=False, indent=2)
    
    # Remove checkpoint file
    if os.path.exists(LLAVA_CHECKPOINT):
        os.remove(LLAVA_CHECKPOINT)
    
    elapsed = time.time() - start_time
    print(f"\nGenerated {len(llava_captions)} captions in {elapsed/60:.1f} min")
    print(f"   Errors: {batch_errors}/{total}")
    
    # Giai phong model
    del llava_model, processor
    torch.cuda.empty_cache()

# Spot-check
print("\nCaption examples:")
for i in range(min(3, len(llava_captions))):
    idx_str = str(i)
    caption = llava_captions.get(idx_str, "N/A")
    tweet = str(data.iloc[i].get("text", ""))[:80]
    print(f"\n  [{i}] Tweet: {tweet}...")
    print(f"       Caption: {caption[:120]}...")

# %%
# ============================================================================
# CELL 6: Encode LLaVA Captions with CLIP Text Encoder 🆕
# ============================================================================
print("\n" + "=" * 60)
print("🔄 Encoding LLaVA captions with CLIP")
print("=" * 60)

# Two encoding strategies:
# Strategy A (Munia-style): concat tweet + caption -> CLIP text encode
# Strategy B (Separate):    caption only -> CLIP text encode (3rd independent modality)
# We cache both for ablation.

LLAVA_FEAT_COMBINED = os.path.join(CACHE_DIR, "clip_ViT-L_14_llava_combined.npy")
LLAVA_FEAT_SEPARATE = os.path.join(CACHE_DIR, "clip_ViT-L_14_llava_only.npy")

need_encode = not os.path.exists(LLAVA_FEAT_COMBINED) or not os.path.exists(LLAVA_FEAT_SEPARATE)

if need_encode:
    import open_clip
    
    print("Encoding LLaVA captions with CLIP ViT-L/14 text encoder...")
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model = clip_model.to(DEVICE).eval()
    
    all_combined_feats = []
    all_separate_feats = []
    batch_size = 64
    total = len(data)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        
        combined_texts = []
        caption_only_texts = []
        for idx in range(start, end):
            tweet = str(data.iloc[idx].get("text", ""))
            caption = llava_captions.get(str(idx), "")
            # Strategy A: Munia-style concat
            combined_texts.append(f"{tweet} [SEP] {caption}")
            # Strategy B: caption only
            caption_only_texts.append(caption if caption else tweet)
        
        # Encode combined
        tokens_combined = tokenizer(combined_texts).to(DEVICE)
        tokens_separate = tokenizer(caption_only_texts).to(DEVICE)
        
        with torch.no_grad():
            feats_c = clip_model.encode_text(tokens_combined)
            feats_c = feats_c / feats_c.norm(dim=-1, keepdim=True)
            feats_s = clip_model.encode_text(tokens_separate)
            feats_s = feats_s / feats_s.norm(dim=-1, keepdim=True)
        
        all_combined_feats.append(feats_c.cpu().numpy())
        all_separate_feats.append(feats_s.cpu().numpy())
        
        if (start // batch_size + 1) % 50 == 0:
            print(f"   Batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
    
    llava_combined = np.concatenate(all_combined_feats)
    llava_separate = np.concatenate(all_separate_feats)
    np.save(LLAVA_FEAT_COMBINED, llava_combined)
    np.save(LLAVA_FEAT_SEPARATE, llava_separate)
    
    del clip_model
    torch.cuda.empty_cache()
else:
    print("Loading cached LLaVA CLIP features")
    llava_combined = np.load(LLAVA_FEAT_COMBINED)
    llava_separate = np.load(LLAVA_FEAT_SEPARATE)

# Default: use combined (Munia-style) for main experiments
llava_features = llava_combined

print(f"LLaVA combined features: {llava_combined.shape}")
print(f"LLaVA separate features: {llava_separate.shape}")
print(f"   Combined L2 norms: mean={np.linalg.norm(llava_combined, axis=1).mean():.4f}")
print(f"   Separate L2 norms: mean={np.linalg.norm(llava_separate, axis=1).mean():.4f}")

# Cosine similarity giua combined vs original text features
cos_sim = np.sum(llava_combined * text_features, axis=1).mean()
print(f"   Cosine sim (combined vs original text): {cos_sim:.4f}")

# %%
# ============================================================================
# CELL 7: Model & Training Setup (imported from src/) ✅
# ============================================================================
print("\n" + "=" * 60)
print("🏗️ Models & Training — imported from src/")
print("=" * 60)

# Models, losses, datasets, trainer — all imported from src/ (no duplicates!)
# GuidedCrossAttention, ThreeModalityClassifier → src.models
# FocalLoss → src.losses
# CrisisMMD3ModalDataset, create_3modal_loaders → src.data
# GenericTrainer → src.trainer

print("   ✅ GuidedCrossAttention (Munia et al. CVPRw 2025)")
print("   ✅ ThreeModalityClassifier (2/3-modal, concat/GCA)")
print("   ✅ FocalLoss (with label_smoothing)")
print("   ✅ GenericTrainer (run_experiment)")

# Tạo loaders
BATCH_SIZE = 32  # Munia et al. dùng 32

loaders_3modal = create_3modal_loaders(
    image_features, text_features, llava_features,
    labels, train_indices, val_indices, test_indices,
    batch_size=BATCH_SIZE,
)

for name, loader in loaders_3modal.items():
    print(f"   {name}: {len(loader.dataset)} samples, {len(loader)} batches")

# Khởi tạo trainer và loss
trainer = GenericTrainer(device=DEVICE)
loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)

print("\n✅ Training setup ready")

# %%
# ============================================================================
# CELL 10: Experiment A — Baselines (2-modal) 🧪
# ============================================================================
print("\n" + "=" * 70)
print("🧪 Experiment A: 2-Modal Baselines (CLIP image + text)")
print("   Target: reproduce ~78% F1w baseline")
print("=" * 70)

SEEDS = [42, 123, 456]
all_results = {}

# --- A1: Concat + MLP (2-modal, no Guided CA) ---
print("\n--- A1: Concat + MLP (2-modal) ---")
a1_results = []
for seed in SEEDS:
    model = ThreeModalityClassifier(
        feat_dim=768, num_classes=2, dropout=0.2,
        use_guided_ca=False, use_llava=False,
    )
    result = trainer.run_experiment(
        "A1: Concat+MLP (2-modal)", model, loaders_3modal, loss_fn,
        epochs=50, lr=3e-4, patience=7, use_llava=False, seed=seed,
    )
    a1_results.append(result)

a1_f1 = np.mean([r["test_metrics"]["f1_weighted"] for r in a1_results])
a1_std = np.std([r["test_metrics"]["f1_weighted"] for r in a1_results])
all_results["A1_concat_2modal"] = a1_results
print(f"\n✅ A1 Average: F1w = {a1_f1:.4f} ± {a1_std:.4f}")

# --- A2: Guided CA + MLP (2-modal) ---
print("\n--- A2: Guided CA + MLP (2-modal) ---")
a2_results = []
for seed in SEEDS:
    model = ThreeModalityClassifier(
        feat_dim=768, num_classes=2, dropout=0.2,
        use_guided_ca=True, use_llava=False,
    )
    result = trainer.run_experiment(
        "A2: Guided CA (2-modal)", model, loaders_3modal, loss_fn,
        epochs=50, lr=3e-4, patience=7, use_llava=False, seed=seed,
    )
    a2_results.append(result)

a2_f1 = np.mean([r["test_metrics"]["f1_weighted"] for r in a2_results])
a2_std = np.std([r["test_metrics"]["f1_weighted"] for r in a2_results])
all_results["A2_guidedca_2modal"] = a2_results
print(f"\n✅ A2 Average: F1w = {a2_f1:.4f} ± {a2_std:.4f}")

# %%
# ============================================================================
# CELL 11: Experiment B — 3-Modal (+ LLaVA) 🧪🔑
# ============================================================================
print("\n" + "=" * 70)
print("🧪 Experiment B: 3-Modal (+LLaVA captions)")
print("   H1: LLaVA → +10% F1w")
print("   Target: >88% F1w")
print("=" * 70)

# --- B1: Concat + MLP (3-modal) ---
print("\n--- B1: Concat + MLP (3-modal, +LLaVA) ---")
b1_results = []
for seed in SEEDS:
    model = ThreeModalityClassifier(
        feat_dim=768, num_classes=2, dropout=0.2,
        use_guided_ca=False, use_llava=True,
    )
    result = trainer.run_experiment(
        "B1: Concat+MLP (3-modal)", model, loaders_3modal, loss_fn,
        epochs=50, lr=3e-4, patience=7, use_llava=True, seed=seed,
    )
    b1_results.append(result)

b1_f1 = np.mean([r["test_metrics"]["f1_weighted"] for r in b1_results])
b1_std = np.std([r["test_metrics"]["f1_weighted"] for r in b1_results])
all_results["B1_concat_3modal"] = b1_results
print(f"\n✅ B1 Average: F1w = {b1_f1:.4f} ± {b1_std:.4f}")

# --- B2: Guided CA + MLP (3-modal) --- KEY EXPERIMENT
print("\n--- B2: Guided CA + MLP (3-modal, +LLaVA) --- ⭐ KEY")
b2_results = []
for seed in SEEDS:
    model = ThreeModalityClassifier(
        feat_dim=768, num_classes=2, dropout=0.2,
        use_guided_ca=True, use_llava=True,
    )
    result = trainer.run_experiment(
        "B2: Guided CA (3-modal) ⭐", model, loaders_3modal, loss_fn,
        epochs=50, lr=3e-4, patience=7, use_llava=True, seed=seed,
    )
    b2_results.append(result)

b2_f1 = np.mean([r["test_metrics"]["f1_weighted"] for r in b2_results])
b2_std = np.std([r["test_metrics"]["f1_weighted"] for r in b2_results])
all_results["B2_guidedca_3modal"] = b2_results
print(f"\n B2 Average: F1w = {b2_f1:.4f} +/- {b2_std:.4f}")

# %%
# ============================================================================
# CELL 11b: Experiment C — LLaVA Encoding Ablation (Combined vs Separate)
# ============================================================================
print("\n" + "=" * 70)
print("Experiment C: LLaVA Encoding Ablation")
print("   Combined (tweet+caption concat) vs Separate (caption only)")
print("=" * 70)

# Build loaders with SEPARATE LLaVA features (caption-only encoding)
loaders_separate = create_3modal_loaders(
    image_features, text_features, llava_separate,
    labels, train_indices, val_indices, test_indices,
    batch_size=BATCH_SIZE,
)

# --- C1: Guided CA (3-modal) + LLaVA separate ---
print("\n--- C1: Guided CA + LLaVA (caption-only encoding) ---")
c1_results = []
for seed in SEEDS:
    model = ThreeModalityClassifier(
        feat_dim=768, num_classes=2, dropout=0.2,
        use_guided_ca=True, use_llava=True,
    )
    result = trainer.run_experiment(
        "C1: GCA + LLaVA-separate", model, loaders_separate, loss_fn,
        epochs=50, lr=3e-4, patience=7, use_llava=True, seed=seed,
    )
    c1_results.append(result)

c1_f1 = np.mean([r["test_metrics"]["f1_weighted"] for r in c1_results])
c1_std = np.std([r["test_metrics"]["f1_weighted"] for r in c1_results])
all_results["C1_guidedca_separate"] = c1_results
print(f"\n C1 Average: F1w = {c1_f1:.4f} +/- {c1_std:.4f}")

print(f"\n--- LLaVA Encoding Ablation ---")
print(f"   Combined (Munia-style): {b2_f1:.4f} +/- {b2_std:.4f}")
print(f"   Separate (caption-only): {c1_f1:.4f} +/- {c1_std:.4f}")
print(f"   Delta: {b2_f1 - c1_f1:+.4f}")

# %%
# ============================================================================
# CELL 12: Results Summary & Comparison
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY")
print("=" * 70)

summary_rows = []
for name, results in all_results.items():
    test_f1s = [r["test_metrics"]["f1_weighted"] for r in results]
    test_f1m = [r["test_metrics"]["f1_macro"] for r in results]
    test_bacc = [r["test_metrics"]["balanced_acc"] for r in results]
    
    summary_rows.append({
        "Experiment": results[0]["model_name"],
        "F1w_mean": np.mean(test_f1s),
        "F1w_std": np.std(test_f1s),
        "F1m_mean": np.mean(test_f1m),
        "BAcc_mean": np.mean(test_bacc),
        "Seeds": len(results),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("F1w_mean", ascending=False)

print("\n" + summary_df.to_string(index=False, float_format="%.4f"))

# Comparison with SOTA
print(f"\n{'─' * 50}")
print("📈 Comparison with SOTA:")
print(f"   Our V3 baseline (frozen CLIP):     0.7830 F1w")
print(f"   Our best (this experiment):        {summary_df['F1w_mean'].max():.4f} F1w")
print(f"   Munia et al. CLIP+Wiki+GCA:        0.9045 F1w")
print(f"   Munia et al. CLIP+LLaVA+GCA (SOTA): 0.9289 F1w")
delta = summary_df["F1w_mean"].max() - 0.7830
print(f"   Δ over V3 baseline:               +{delta:.4f}")
print(f"{'─' * 50}")

# %%
# ============================================================================
# CELL 13: Visualization 📊
# ============================================================================
print("\n" + "=" * 60)
print("📊 Visualizations")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Bar chart: F1w comparison
ax = axes[0]
names = [r["Experiment"].replace("(", "\n(") for _, r in summary_df.iterrows()]
means = summary_df["F1w_mean"].values
stds = summary_df["F1w_std"].values
colors = ["#4CAF50" if "3-modal" in n else "#2196F3" for n in names]
bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("F1w (weighted)")
ax.set_title("Model Comparison")
ax.axhline(y=0.9289, color='red', linestyle='--', alpha=0.7, label='SOTA (Munia)')
ax.axhline(y=0.783, color='gray', linestyle='--', alpha=0.5, label='V3 baseline')
ax.legend()
ax.set_ylim(0.7, 1.0)

# 2. Training curves (best model)
ax = axes[1]
best_key = summary_df.index[0]
best_name = list(all_results.keys())[best_key] if best_key < len(all_results) else list(all_results.keys())[-1]
best_history = list(all_results.values())[-1][0]["history"]
ax.plot(best_history["train_f1"], label="Train F1", color="#1976D2")
ax.plot(best_history["val_f1"], label="Val F1", color="#E53935")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 weighted")
ax.set_title("Best Model Training Curve")
ax.legend()

# 3. Ablation: LLaVA impact
ax = axes[2]
ablation_data = {
    "Without LLaVA\n(2-modal)": summary_df[summary_df["Experiment"].str.contains("2-modal")]["F1w_mean"].max(),
    "With LLaVA\n(3-modal)": summary_df[summary_df["Experiment"].str.contains("3-modal")]["F1w_mean"].max(),
}
bars = ax.bar(ablation_data.keys(), ablation_data.values(), color=["#FF9800", "#4CAF50"], alpha=0.8)
ax.set_ylabel("F1w (weighted)")
ax.set_title("LLaVA Caption Impact")
for bar, val in zip(bars, ablation_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0.7, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "experiments/h1-llava-captions/results/comparison.png"), dpi=150)
plt.show()
print("✅ Saved comparison plot")

# %%
# ============================================================================
# CELL 14: Classification Report (Best Model)
# ============================================================================
print("\n" + "=" * 60)
print("📋 Detailed Classification Report (Best Model)")
print("=" * 60)

# Get best model's test predictions
best_result = list(all_results.values())[-1][0]  # Last experiment, first seed
preds = best_result["test_metrics"]["preds"]
true_labels = best_result["test_metrics"]["labels"]

print(classification_report(
    true_labels, preds,
    target_names=["Not Informative", "Informative"],
    digits=4,
))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Inform.", "Informative"],
            yticklabels=["Not Inform.", "Informative"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {best_result['model_name']}")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "experiments/h1-llava-captions/results/confusion_matrix.png"), dpi=150)
plt.show()

# %%
# ============================================================================
# CELL 15: Save Results & Update Research State
# ============================================================================
print("\n" + "=" * 60)
print("💾 Saving results")
print("=" * 60)

# Save summary
results_summary = {
    "experiment": "V4_LLaVA_GuidedCA",
    "date": "2026-03-22",
    "results": {},
}

for name, results in all_results.items():
    test_f1s = [r["test_metrics"]["f1_weighted"] for r in results]
    results_summary["results"][name] = {
        "f1w_mean": float(np.mean(test_f1s)),
        "f1w_std": float(np.std(test_f1s)),
        "f1w_seeds": [float(x) for x in test_f1s],
    }

with open(os.path.join(PROJECT_DIR, "experiments/h1-llava-captions/results/summary.json"), "w") as f:
    json.dump(results_summary, f, indent=2)

print("✅ Results saved to experiments/h1-llava-captions/results/")
print(f"\n🎯 Best F1w: {summary_df['F1w_mean'].max():.4f}")
print(f"   SOTA gap: {0.9289 - summary_df['F1w_mean'].max():.4f}")

if summary_df["F1w_mean"].max() > 0.90:
    print("\n🎉 >90% F1 ACHIEVED! Proceed to H3 (causal disentanglement on top)")
elif summary_df["F1w_mean"].max() > 0.88:
    print("\n✅ H1 SUPPORTED: LLaVA captions significantly improve performance")
    print("   Next: Try Guided CA refinements or LoRA fine-tuning")
else:
    print("\n⚠️ H1 needs investigation: LLaVA improvement lower than expected")
    print("   Check: caption quality, encoding strategy, data splits")
