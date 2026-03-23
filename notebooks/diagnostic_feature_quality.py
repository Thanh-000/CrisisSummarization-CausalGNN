# ============================================================================
# 🔍 DIAGNOSTIC — CLIP Feature Quality Check
# ============================================================================
# Mục đích: Xác định vấn đề nằm ở CLIP features hay training protocol
#
# Chạy cell này NGAY SAU Cell 4 (CLIP Feature Extraction) trong notebook chính
# Cần có: image_features, text_features, labels, data (từ cells trước)
# ============================================================================

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from collections import Counter

print("=" * 70)
print("🔍 DIAGNOSTIC: CLIP Feature Quality Check")
print("=" * 70)

# ── 1. Basic Feature Statistics ──
print("\n📐 1. Feature Statistics")
print("─" * 50)

for name, feat in [("Image", image_features), ("Text", text_features)]:
    norms = np.linalg.norm(feat, axis=1)
    print(f"\n  {name} features: shape={feat.shape}, dtype={feat.dtype}")
    print(f"    Mean:   {feat.mean():.6f}")
    print(f"    Std:    {feat.std():.6f}")
    print(f"    Range:  [{feat.min():.6f}, {feat.max():.6f}]")
    print(f"    L2 norms: mean={norms.mean():.4f}, std={norms.std():.6f}")
    print(f"    L2 norms: min={norms.min():.4f}, max={norms.max():.4f}")
    
    # Kiểm tra features có all-zero hoặc near-zero
    zero_rows = np.sum(norms < 1e-6)
    print(f"    Zero/near-zero rows: {zero_rows}")
    
    # Kiểm tra NaN/Inf
    nan_count = np.sum(np.isnan(feat))
    inf_count = np.sum(np.isinf(feat))
    print(f"    NaN values: {nan_count}, Inf values: {inf_count}")

# ── 2. Feature Uniqueness ──
print(f"\n📊 2. Feature Uniqueness")
print("─" * 50)
n_total = len(image_features)
n_unique_img = len(np.unique(image_features, axis=0))
n_unique_txt = len(np.unique(text_features, axis=0))
print(f"  Total samples: {n_total}")
print(f"  Unique image features: {n_unique_img} ({n_unique_img/n_total*100:.1f}%)")
print(f"  Unique text features:  {n_unique_txt} ({n_unique_txt/n_total*100:.1f}%)")

if n_unique_img < n_total * 0.5:
    print("  ⚠️ WARNING: Many duplicate image features — extraction may be broken!")
if n_unique_txt < n_total * 0.5:
    print("  ⚠️ WARNING: Many duplicate text features — extraction may be broken!")

# ── 3. Text Data Quality ──
print(f"\n📝 3. Text Data Quality")
print("─" * 50)
if "text" in data.columns:
    texts = data["text"].fillna("")
    empty_text = (texts.str.strip() == "").sum()
    nan_text = data["text"].isna().sum()
    short_text = (texts.str.len() < 10).sum()
    avg_len = texts.str.len().mean()
    
    print(f"  Total texts: {len(texts)}")
    print(f"  NaN/missing: {nan_text} ({nan_text/len(texts)*100:.1f}%)")
    print(f"  Empty: {empty_text} ({empty_text/len(texts)*100:.1f}%)")
    print(f"  Very short (<10 chars): {short_text} ({short_text/len(texts)*100:.1f}%)")
    print(f"  Average length: {avg_len:.0f} chars")
    
    if empty_text + nan_text > len(texts) * 0.1:
        print("  ⚠️ WARNING: >10% texts are empty/missing — text features may be noise!")
else:
    print("  ❌ No 'text' column found!")

# ── 4. Label Distribution ──
print(f"\n🏷️ 4. Label Distribution")
print("─" * 50)
label_counts = Counter(labels)
for cls, count in sorted(label_counts.items()):
    pct = count / len(labels) * 100
    print(f"  Class {cls}: {count:,} ({pct:.1f}%)")

imbalance_ratio = max(label_counts.values()) / min(label_counts.values())
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 5:
    print(f"  ⚠️ WARNING: High class imbalance (>{5}x)")

# ── 5. Split Distribution ──
print(f"\n📂 5. Split Distribution")
print("─" * 50)
if "split" in data.columns:
    for split in ["train", "dev", "test"]:
        split_mask = data["split"] == split
        split_data = data[split_mask]
        if len(split_data) > 0:
            split_labels = split_data["label"].values
            split_counts = Counter(split_labels)
            split_counts_clean = {int(k): int(v) for k, v in split_counts.items()}
            print(f"  {split}: {len(split_data):,} samples — {split_counts_clean}")
else:
    print("  No 'split' column — using random splits")

# ── 6. LINEAR PROBE (Key Diagnostic) ──
print(f"\n🔬 6. LINEAR PROBE — Feature Separability Test")
print("─" * 50)
print("  Running 5-fold cross-validation with LogisticRegression...")
print("  This tells us the UPPER BOUND of what a linear model can achieve.")
print()

# Prepare data
X_img = image_features.copy()
X_txt = text_features.copy()
X_concat = np.concatenate([X_img, X_txt], axis=1)
y = labels.copy()

# Scale features (important for logistic regression)
scaler = StandardScaler()

# Test multiple feature combinations
configs = [
    ("Image only", X_img),
    ("Text only", X_txt),
    ("Image + Text (concat)", X_concat),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, X in configs:
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(
        max_iter=2000, 
        C=1.0,
        solver='lbfgs',
        random_state=42,
    )
    
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_weighted')
    results[name] = scores
    
    print(f"  {name:30s} → F1 = {scores.mean():.4f} ± {scores.std():.4f}")

print()
best_name = max(results, key=lambda k: results[k].mean())
best_f1 = results[best_name].mean()

# ── 7. Also test with un-normalized features ──
print(f"\n🔬 7. UN-NORMALIZED Feature Test")
print("─" * 50)
print("  Testing if L2 normalization hurts performance...")

# Try without L2 norm (just raw features * norm)
X_img_raw = X_img * np.linalg.norm(X_img, axis=1, keepdims=True)  
# Note: If features are already L2-normalized, this doesn't help
# We need to check if pre-norm features give better results

# More useful: try different scalings
scaler2 = StandardScaler()
X_concat_scaled = scaler2.fit_transform(X_concat)

# Logistic Regression with stronger regularization search
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(max_iter=2000, C=C, solver='lbfgs', random_state=42)
    scores = cross_val_score(clf, X_concat_scaled, y, cv=cv, scoring='f1_weighted')
    marker = " ← best" if scores.mean() >= best_f1 else ""
    print(f"  LR(C={C:5.2f}) → F1 = {scores.mean():.4f} ± {scores.std():.4f}{marker}")
    if scores.mean() > best_f1:
        best_f1 = scores.mean()

# ── 8. MLP Sanity Check (sklearn) ──
print(f"\n🔬 8. MLP Sanity Check (sklearn)")
print("─" * 50)
try:
    from sklearn.neural_network import MLPClassifier
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    
    scores = cross_val_score(mlp, X_concat_scaled, y, cv=cv, scoring='f1_weighted')
    print(f"  MLP(512,256) → F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    if scores.mean() > best_f1:
        best_f1 = scores.mean()
        best_name = "sklearn MLP(512,256)"
except Exception as e:
    print(f"  ⚠️ MLP failed: {e}")

# ── 9. Per-split linear probe (official splits) ──
print(f"\n🔬 9. Official Split Linear Probe")
print("─" * 50)
official_test_f1 = None
if "split" in data.columns:
    train_mask = data["split"] == "train"
    dev_mask = data["split"] == "dev"
    test_mask = data["split"] == "test"
    
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        X_train = scaler.fit_transform(X_concat[train_mask.values])
        y_train = y[train_mask.values]
        
        # Use dev for validation if available
        if dev_mask.sum() > 0:
            X_dev = scaler.transform(X_concat[dev_mask.values])
            y_dev = y[dev_mask.values]
            X_test = scaler.transform(X_concat[test_mask.values])
            y_test = y[test_mask.values]
        else:
            X_test = scaler.transform(X_concat[test_mask.values])
            y_test = y[test_mask.values]
        
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        pred_test = clf.predict(X_test)
        f1_test = f1_score(y_test, pred_test, average='weighted')
        official_test_f1 = float(f1_test)
        
        print(f"  Train on official train → Test F1 = {f1_test:.4f}")
        print(f"\n  Classification Report (official test):")
        print(classification_report(y_test, pred_test, digits=4))
        
        if dev_mask.sum() > 0:
            pred_dev = clf.predict(X_dev)
            f1_dev = f1_score(y_dev, pred_dev, average='weighted')
            print(f"  Dev F1 = {f1_dev:.4f}")
    else:
        print("  ⚠️ Not enough splits for train/test evaluation")
else:
    print("  No official splits found — skipping")

# ── 10. VERDICT ──
print(f"\n{'=' * 70}")
print(f"📋 DIAGNOSTIC VERDICT")
print(f"{'=' * 70}")
print(f"\n  Best linear probe F1: {best_f1:.4f} ({best_name})")
if official_test_f1 is not None:
    print(f"  Official split test F1: {official_test_f1:.4f}")
print()

decision_f1 = official_test_f1 if official_test_f1 is not None else float(best_f1)

if decision_f1 >= 0.86:
    print("  ✅ CLIP features are STRONG for frozen-feature probing")
    print("  → Focus on model/training design to gain final points")
elif decision_f1 >= 0.80:
    print("  ✅ CLIP features are GOOD and usable")
    print("  → No extraction bug signal; continue with training/architecture tuning")
elif decision_f1 >= 0.74:
    print("  ⚠️ CLIP features are ACCEPTABLE baseline for CrisisMMD Task 1")
    print("  → This range is common for frozen CLIP + linear probe")
    print("  → 0.90+ usually needs stronger nonlinear heads and full multimodal training")
else:
    print("  ❌ CLIP features are LIKELY WEAK (<0.74)")
    print("  → Check extraction setup, label alignment, and text/image path quality")

print(f"\n{'=' * 70}")
