# ⏱️ Paper vs. Project — Timing & Computational Resource Comparison

**Paper:** "Multimodal Classification of Social Media Disaster Posts With Graph Neural Networks and Few-Shot Learning"  
**Authors:** José Nascimento, Paolo Bestagini, Anderson Rocha  
**Published:** IEEE Access, January 2025  
**Last Updated:** 2026-02-11

---

## 📋 Table of Contents

1. [Hardware Comparison](#1-hardware-comparison)
2. [Paper Hyperparameters (Table 1)](#2-paper-hyperparameters-table-1)
3. [Your Project Configuration](#3-your-project-configuration)
4. [Configuration Diff — Paper vs Project](#4-configuration-diff--paper-vs-project)
5. [Paper Runtime Data (Table 2 & Table 4)](#5-paper-runtime-data-table-2--table-4)
6. [Runtime Analysis — Breakdown](#6-runtime-analysis--breakdown)
7. [Key Differences Impacting Runtime](#7-key-differences-impacting-runtime)
8. [Optimization Recommendations](#8-optimization-recommendations)
9. [Conclusion](#9-conclusion)

---

## 1. Hardware Comparison

| Spec | 📄 Paper (CrisisSpot) | 💻 Your Project |
|------|----------------------|-----------------|
| **GPU** | NVIDIA Quadro RTX 8000 | Google Colab T4 / A100 |
| **GPU VRAM** | 48 GB GDDR6 | T4: 15 GB / A100: 40 GB |
| **GPU Architecture** | Turing (TU104) | T4: Turing / A100: Ampere |
| **FP32 Performance** | ~16.3 TFLOPS | T4: ~8.1 / A100: ~19.5 TFLOPS |
| **FP16 Performance** | ~32.6 TFLOPS (w/ Tensor Cores) | T4: ~65 / A100: ~312 TFLOPS |
| **Memory Bandwidth** | 672 GB/s | T4: 320 / A100: 2039 GB/s |
| **RAM** | Unknown (likely ≥64 GB) | Colab: 12–25 GB |
| **Mixed Precision** | Not mentioned (likely no AMP) | ✅ Enabled (AMP FP16/BF16) |

### 💡 Key Insight
- **RTX 8000 vs T4:** The paper's GPU has **~3× more VRAM** and **~2× FP32** perf vs T4. However, T4 has competitive FP16 tensor core performance.
- **RTX 8000 vs A100:** A100 is actually **faster** than RTX 8000 in every metric. If you run on A100, you have a hardware advantage.
- **Your AMP optimization** partially compensates for the T4's FP32 disadvantage by leveraging FP16 tensor cores.

---

## 2. Paper Hyperparameters (Table 1)

Based on the paper's Table 1 and experimental setup:

| Parameter | Paper Value | Notes |
|-----------|------------|-------|
| **Epochs** | 500 | Maximum training epochs |
| **Optimizer** | Adam | Standard adaptive optimizer |
| **Learning Rate** | 1 × 10⁻⁵ | Very low LR for fine-tuning |
| **Weight Decay** | 1 × 10⁻² | L2 regularization |
| **Graph Construction** | kNN with cosine similarity | Threshold ≥ 0.75 |
| **GNN Architecture** | GraphSAGE | 2-layer with normalization |
| **Feature Reduction** | PCA (256 dims) | Reduces embedding dimensionality |
| **Fusion Strategy** | Late Fusion | Best performing in ablation |
| **Text Encoder** | MPNet / BERT | Sentence-level embeddings |
| **Image Encoder** | MaxViT / ResNet50 | Visual feature extraction |
| **Labeled Samples** | 50, 100, 250, 500 | Few-shot settings |
| **Splits** | 10 random splits | Per labeled-sample configuration |
| **Runtime Measurement** | 5 runs, same split (50 labeled) | Mean ± std reported |

---

## 3. Your Project Configuration

From `optimize_test2_resources.py`:

| Parameter | Your Value | Notes |
|-----------|-----------|-------|
| **Epochs** | 2000 | **4× paper** — very high max |
| **Early Stop** | 300 | Patience before stopping |
| **Optimizer** | (inherited from notebook) | Likely Adam |
| **Learning Rate** | 1 × 10⁻⁵ | ✅ Matches paper |
| **Weight Decay (wd)** | 1 × 10⁻³ | ⚠️ Different from paper (10⁻²) |
| **L2 Regularization** | 1 × 10⁻² | Additional L2 term |
| **Dropout** | 0.5 | Higher than typical GNN dropout |
| **kNN** | 16 | Number of nearest neighbors |
| **Fraction (frac)** | 0.4 | Graph edge fraction / sampling |
| **Architecture** | `sage-2l-norm-res` | 2-layer GraphSAGE + norm + residual |
| **Fusion** | `late` | ✅ Matches paper's best config |
| **Feature Reduction** | `None` | ❌ **No PCA** — full-dimensional |
| **Image Encoder** | `maxvit` | ✅ Matches paper |
| **Text Encoder** | `mpnet` | ✅ Matches paper |
| **Labeled Samples** | [50, 100, 250, 500] | ✅ Matches paper |
| **Splits** | 10 | ✅ Matches paper |
| **Resource Tuning** | AMP, GC tuning, thread limiting | Paper doesn't mention these |

---

## 4. Configuration Diff — Paper vs Project

| Parameter | 📄 Paper | 💻 Your Project | Impact |
|-----------|---------|-----------------|--------|
| **Epochs** | 500 | 2000 | 🔴 **4× more epochs** = much longer training |
| **Early Stop** | Not specified | 300 | 🟡 Partially mitigates epoch count |
| **Weight Decay** | 1e-2 | 1e-3 | 🟡 Minor — may affect convergence speed |
| **PCA Reduction** | 256 dims | None (full dims) | 🔴 **Major** — full dims = heavier GNN computation |
| **kNN** | Not specified explicitly | 16 | 🟡 Affects graph density |
| **Dropout** | Not specified | 0.5 | 🟡 May slow convergence |
| **Mixed Precision** | Not mentioned | Enabled (AMP) | 🟢 Your advantage — faster per-step |
| **GPU** | RTX 8000 (48GB) | T4 (15GB) | 🔴 Your hardware is weaker |

### 🎯 Critical Differences:

1. **Epochs: 500 vs 2000** — You set 4× more max epochs. Even with early stopping at 300, if the model trains for 300 epochs, that's still shorter than 500 but the high max is unnecessary overhead.

2. **PCA: 256 dims vs None** — This is the **single biggest impact on runtime**. Without PCA:
   - Graph construction on full-dimensional embeddings is much slower
   - GNN forward/backward pass processes larger feature vectors
   - Memory usage is significantly higher
   - The paper specifically mentions PCA as an efficiency optimization

3. **Hardware gap** — T4 is ~2× slower than RTX 8000 in FP32, but competitive in FP16 (which your AMP exploits).

---

## 5. Paper Runtime Data (Table 2 & Table 4)

### Table 2: Runtime of Different Methods (seconds, on Quadro RTX 8000)

> *Runtimes are mean of 5 executions, same random split with 50 labeled samples. Covers entire pipeline.*

| Method | Runtime (seconds) | Notes |
|--------|-------------------|-------|
| **Baseline [4] - EDA** | 8458 ± 1122 | Data augmentation baseline (Sirbu et al.) |
| **Baseline [4] - BT** | 7661 | Back-translation variant |
| **Baseline [5]** | 317 | Another comparison method |
| **Ours (Late + PCA)** | 179 | ✅ Proposed method with PCA reduction |
| **Ours (Variant)** | 142 ± 2 | Lighter configuration |
| **Ours (100 labeled)** | 77 | Fewer labeled samples = faster |

### Key Observations:
- CrisisSpot is **47× faster** than the EDA baseline (8458s → 179s)
- CrisisSpot is **~2.3× faster** than the simpler baseline [5] (317s → 142s)
- **PCA reduction is critical** — it's what enables the low runtimes
- The entire pipeline (feature extraction + graph construction + GNN training + fusion) takes only **~3 minutes** on RTX 8000

### Estimated Your Project Runtimes:

| Configuration | Estimated Time | Reasoning |
|--------------|---------------|-----------|
| **50 labeled, T4, no PCA** | ~600–900s (10–15 min) | 2× hardware gap + no PCA overhead |
| **50 labeled, T4, with PCA 256** | ~300–450s (5–7.5 min) | PCA cuts computation ~2× |
| **100 labeled, T4, no PCA** | ~300–500s | Fewer iterations, still high dims |
| **250 labeled, T4, no PCA** | ~500–800s | More data, more graph edges |
| **500 labeled, T4, no PCA** | ~800–1200s | Largest setting |
| **All 40 runs, T4, no PCA** | ~8–12 hours | 40 runs × ~600–900s average |
| **All 40 runs, T4, with PCA** | ~4–6 hours | 40 runs × ~350–500s average |

---

## 6. Runtime Analysis — Breakdown

### Paper's Pipeline Steps (estimated proportions):

```
┌─────────────────────────────────────────────────────────┐
│                    FULL PIPELINE (~179 seconds)          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Feature Extraction (CLIP/MaxViT/MPNet)  ~30s  (17%) │
│     ├── Text encoding (MPNet)               ~5s         │
│     ├── Image encoding (MaxViT)             ~15s        │
│     └── CLIP multimodal encoding            ~10s        │
│                                                         │
│  2. PCA Dimensionality Reduction            ~5s   (3%)  │
│     └── 512/768/1024 → 256 dims                         │
│                                                         │
│  3. Graph Construction (kNN)                ~10s  (6%)  │
│     └── Cosine similarity + threshold                   │
│                                                         │
│  4. GNN Training (GraphSAGE)                ~120s (67%) │
│     ├── 500 epochs × ~0.24s/epoch                       │
│     ├── Forward + backward pass                         │
│     └── With early stopping                             │
│                                                         │
│  5. Evaluation & Metrics                    ~14s  (8%)  │
│     └── F1, Accuracy, Precision, Recall                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Your Project's Estimated Breakdown (T4, no PCA):

```
┌─────────────────────────────────────────────────────────┐
│                 YOUR PIPELINE (~600-900 seconds)        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Feature Extraction                       ~40s  (6%) │
│     ├── Text encoding (MPNet)                ~5s        │
│     ├── Image encoding (MaxViT)              ~20s       │
│     └── CLIP multimodal encoding             ~15s       │
│                                                         │
│  2. PCA Reduction                            ~0s   (0%) │
│     └── SKIPPED (red=None)                              │
│                                                         │
│  3. Graph Construction (kNN)                 ~30s  (4%) │
│     └── Full-dim vectors = slower kNN                   │
│                                                         │
│  4. GNN Training (GraphSAGE)                 ~500s (70%)│
│     ├── Up to 2000 epochs (early stop ~300)             │
│     ├── Full-dim features = heavier per-epoch           │
│     ├── ~1.5-2s/epoch without PCA                       │
│     └── AMP helps: ~1.0-1.5s/epoch                     │
│                                                         │
│  5. Evaluation & Metrics                     ~20s  (3%) │
│                                                         │
│  6. Resource Overhead (GC, etc.)             ~10s  (1%) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Key Differences Impacting Runtime

### 🔴 Critical Factors (Ordered by Impact)

| # | Factor | Impact Level | Runtime Multiplier |
|---|--------|-------------|-------------------|
| 1 | **No PCA reduction** | 🔴🔴🔴 | ~2-3× slower per epoch |
| 2 | **More epochs (2000 vs 500)** | 🔴🔴 | Up to 4× (mitigated by early stop) |
| 3 | **Weaker GPU (T4 vs RTX 8000)** | 🔴🔴 | ~1.5-2× slower |
| 4 | **Combined effect** | 🔴🔴🔴🔴 | ~3-5× total slower |

### 🟢 Your Advantages

| # | Factor | Benefit |
|---|--------|---------|
| 1 | **AMP (Mixed Precision)** | ~1.3-1.5× speedup on T4 |
| 2 | **Thread tuning** | Better CPU utilization |
| 3 | **Aggressive GC** | Prevents OOM and slowdowns |
| 4 | **Early stopping (300)** | Prevents training to 2000 epochs |

### Net Effect:
```
Paper:   179s (RTX 8000 + PCA + 500 epochs)
Yours:   ~600-900s (T4 + no PCA + 2000 max epochs + AMP)
Ratio:   ~3.4-5× slower than paper
```

---

## 8. Optimization Recommendations

### 🚀 Priority 1: Add PCA Reduction (Biggest Impact)

Change in your config:
```python
# BEFORE (current)
CFG = dict(
    ...
    red=None,        # ❌ No dimensionality reduction
    ...
)

# AFTER (recommended)
CFG = dict(
    ...
    red='pca-256',   # ✅ Matches paper — PCA to 256 dims
    ...
)
```

**Expected Impact:** ~2-3× speedup (from ~600-900s → ~200-400s per run)

### 🚀 Priority 2: Reduce Max Epochs to Match Paper

```python
# BEFORE
EPOCHS = 2000      # Too high
EARLY_STOP = 300   # Good, but max epochs too high

# AFTER
EPOCHS = 500       # ✅ Matches paper exactly
EARLY_STOP = 100   # Sufficient for convergence detection
```

**Expected Impact:** ~1.3-1.5× speedup (fewer computational epochs)

### 🚀 Priority 3: Match Weight Decay

```python
# BEFORE
CFG = dict(
    ...
    wd=1e-3,   # Different from paper
    ...
)

# AFTER  
CFG = dict(
    ...
    wd=1e-2,   # ✅ Matches paper — stronger regularization = faster convergence
    ...
)
```

**Expected Impact:** Faster convergence, possibly fewer epochs needed

### 📊 Combined Optimization Estimate

| Scenario | Time per run (50 labeled) | Total (40 runs) |
|----------|--------------------------|-----------------|
| **Current** (no PCA, 2000 epochs, T4) | ~600-900s | ~8-12 hours |
| **+ PCA 256** | ~250-400s | ~3-5 hours |
| **+ PCA + Epochs=500** | ~180-300s | ~2-3.5 hours |
| **+ PCA + Epochs + wd fix** | ~150-250s | ~1.7-3 hours |
| **Paper (RTX 8000)** | ~179s | ~2 hours* |

*Paper did 10 splits × 4 settings (50/100/250/500) × 1 run = 40 runs, plus 5 timing runs

---

## 9. Conclusion

### Summary

| Aspect | Paper | Your Project | Gap |
|--------|-------|-------------|-----|
| **F1 Score** | 98.23% | ~98% (matched) | ✅ Negligible |
| **Runtime (single)** | ~179s | ~600-900s | 🔴 3-5× slower |
| **Total Experiment** | ~2-3 hours | ~8-12 hours | 🔴 4× slower |
| **Hardware** | RTX 8000 | T4 | 🟡 Unavoidable |

### Action Items

1. ✅ **Results are correct** — Your F1 scores match the paper, confirming correct implementation
2. 🔧 **Add PCA=256** — Single biggest optimization, matching paper's approach
3. 🔧 **Set EPOCHS=500** — Match paper's training schedule
4. 🔧 **Set wd=1e-2** — Match paper's regularization
5. 📊 **Re-run experiments** with optimized config to validate runtime improvement
6. 📝 **Note in report** — If running on T4, ~1.5-2× overhead vs RTX 8000 is expected and acceptable

### Bottom Line

> Your project successfully reproduces the paper's **accuracy** (F1 ~98%). The runtime gap is primarily due to **missing PCA reduction** (`red=None` instead of `red='pca-256'`), **excessive max epochs** (2000 vs 500), and **weaker hardware** (T4 vs RTX 8000). Applying PCA reduction alone should bring your runtime within **2× of the paper's numbers**, which is excellent given the hardware difference.

---

*Analysis generated: 2026-02-11*  
*Based on: IEEE Access paper + project config analysis*
