# 📊 Code Comparison Report: test2.ipynb vs Original Codebase
## `jdnascim/mm-class-for-disaster-data-with-gnn`

> **Paper**: "Multi-Modal Classification of Disaster Social Media Data with Graph Neural Networks and Few-Shot Learning" — Signal Processing Letters (IEEE)
>
> **Original Repo**: `jdnascim/mm-class-for-disaster-data-with-gnn`
>
> **Your Notebook**: `test2.ipynb`

---

## 1. 🔍 Architecture Overview Comparison

### Original Codebase Architecture
```
feature_extraction.py  →  Extract image/text features
         ↓
feature_fusion.py      →  Reduce (PCA/Autoenc/UMAP) + Fuse (Early/Middle/Late) + Build Graph + Train GNN
         ↓
src/arch/gnn_arch.py   →  Model definitions (BaseGNN, BaseGNNLateFusion, BaseGNNMiddleFusionConcat, BaseMLP)
src/gnn/gnn_utils.py   →  Training loop, evaluation, graph construction (kNN cosine similarity)
src/arch/arch.json     →  Architecture configs (sage-base, sage-2l-norm-res, etc.)
scripts/base/exp_*.sh  →  Experiment configurations (24 experiments)
```

### Your test2.ipynb Architecture
```
Cell 1 (Setup)     →  Install deps + Patch feature_fusion.py (inject CLIP support)
Cell 2 (Data)      →  Manual upload + extract dataset
Cell 3 (Features)  →  Extract CLIP features (image only, missing text!)
Cell 4 (Training)  →  Run feature_fusion.py (but with WRONG arguments)
```

---

## 2. ⚡ Feature Extraction Comparison

### 2.1 Image Feature Extraction

| Aspect | Original Code | Your test2.ipynb | Status |
|--------|--------------|------------------|--------|
| **Model** | MaxViT-tiny (`maxvit_tiny_tf_224.in1k`) via `timm` | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | ⚠️ **DIFFERENT MODEL** |
| **Feature Dim** | 512 (MaxViT-tiny output) | 512 (CLIP ViT-B/32 output) | ✅ Same dimension |
| **Normalization** | `normalize_vector()` (L2 norm from `forensic_lib`) | ❌ **No normalization** | 🔴 **MISSING** |
| **Data Loading** | `ImageDataset` class with PyTorch `DataLoader` | Manual PIL loading + `CLIPProcessor` | ⚠️ Different approach |
| **Batch Size** | 128 (MaxViT) | 64 | Minor difference |
| **Output Format** | DataFrame: image_files, text, embeddings, labels | DataFrame: image_files, text, embeddings, labels, original_label | ✅ Compatible |
| **Label Encoding** | `(1,0)` if not_informative, `(0,1)` if informative → `argmax` → 0 or 1 | `1` if informative else `0` | ✅ Same final values |
| **Image Path** | `join(IMAGEPATH, row['image'])` where IMAGEPATH=`"./data/CrisisMMD_v2.0/"` | `os.path.join(IMG_PATH, d['image'])` where IMG_PATH=`"data/CrisisMMD_v2.0/data_image"` | ⚠️ **PATH MISMATCH** - original joins with `row['image']` which already includes subfolder |

### 2.2 Text Feature Extraction

| Aspect | Original Code | Your test2.ipynb | Status |
|--------|--------------|------------------|--------|
| **Model** | MPNet (`all-mpnet-base-v2`) via `SentenceTransformer` | ❌ **MISSING** — Only `clip_image_features.pkl` is saved | 🔴 **CRITICAL: NO TEXT FEATURES** |
| **Feature Dim** | 768 (MPNet) | N/A | 🔴 |
| **Preprocessing** | `preprocess.pre_process(text, keep_hashtag=True, keep_special_symbols=True)` | `clean_text()` — removes URLs, mentions, hash signs | ⚠️ Different preprocessing |
| **Cache File** | Features extracted and used in-memory | Should save `clip_text_features.pkl` | 🔴 |

### 🔴 CRITICAL FINDING: Text Features Missing
Your `extract_clip_features()` function only extracts **image features** (`model.get_image_features()`). There is **NO `model.get_text_features()`** call. The cached file `clip_text_features.pkl` is **never created**, but the patched `feature_fusion.py` expects to load it.

---

## 3. 🔧 Feature Fusion / Training Pipeline Comparison

### 3.1 Arguments Comparison

| Argument | Original exp_1.sh (Paper's Best?) | Your training cell | Status |
|----------|----------------------------------|-------------------|--------|
| `--epochs` | 2000 | ❌ Missing (defaults to 1000) | 🔴 |
| `--lr` | 1e-5 | 1e-5 | ✅ |
| `--weight_decay` | 1e-3 | ❌ Missing (defaults to 1e-3) | ⚠️ Uses default |
| `--n_neigh_train` | 16 | ❌ Missing | 🔴 |
| `--n_neigh_full` | 16 | ❌ Missing | 🔴 |
| `--lbl_train_frac` | 0.4 (**required**) | ❌ **MISSING - WILL CRASH** | 🔴 **CRITICAL** |
| `--imagepath` | `./data/CrisisMMD_v2.0/` | ❌ Missing | 🔴 |
| `--datasplit` | `{size}_s{id}` (e.g., `50_s0`) | ❌ Missing | 🔴 |
| `--reg` | l2 | ❌ Missing | 🔴 |
| `--l2_lambda` | 1e-2 | ❌ Missing | 🔴 |
| `--exp_id` | 1 (**required**) | ❌ **MISSING - WILL CRASH** | 🔴 **CRITICAL** |
| `--dropout` | 0.5 | ❌ Missing (defaults to 0.5) | ⚠️ |
| `--arch` | `sage-2l-norm-res` | ❌ Missing (defaults to `sage-base`) | 🔴 **WRONG ARCHITECTURE** |
| `--loss` | nll | ❌ Missing (defaults to nll) | ✅ |
| `--shuffle_split` | Yes | ❌ Missing | 🔴 |
| `--imageft` | maxvit | clip_image | ⚠️ Different (intentional) |
| `--textft` | mpnet | clip_text | ⚠️ Different (intentional) |
| `--fusion` | late | ❌ Missing (defaults to early) | 🔴 **WRONG FUSION** |
| `--reduction` | autoenc / pca | ❌ Missing (no reduction) | 🔴 **MISSING** |
| `--autoenc` / `--pca_red` | autoenc-base / 256 | ❌ Missing | 🔴 |
| `--early_stopping` | 300 | ❌ Missing (defaults to -1 = disabled) | 🔴 |
| `--best_model` | best_hm | ❌ Missing (defaults to best_val) | 🔴 |
| `--gnn_layer` | N/A (not an original argument!) | GNN_LAYERS = '1' | 🔴 **INVALID ARG** |
| `--batch_size` | 32 (only default) | `--batch_size 32` sent | ⚠️ Not used by training |

### 🔴 CRITICAL: Invalid Arguments in Training Cell
Your training cell passes `--gnn_layer` which **does not exist** in `feature_fusion.py`'s argparse. The GNN architecture is controlled by `--arch` (e.g., `sage-2l-norm-res` for 2-layer SAGEConv with normalization and residual connections).

Also, `--batch_size` is defined in the argparse but **never actually used** in the training code. The GNN trains on the full graph (transductive learning), not mini-batches.

---

## 4. 🏗️ GNN Architecture Comparison

### Original Architecture (`sage-2l-norm-res` from arch.json)
```json
{
    "arch_layer": "SAGE",
    "residual": true,
    "norm": true,
    "layers_size": [1024, 1024]
}
```

This creates:
```
Input → SAGEConv(input, 1024) → NodeNorm → Residual → ReLU → Dropout(0.5)
      → SAGEConv(1024, 1024)  → NodeNorm → Residual → ReLU → Dropout(0.5)
      → Linear(1024, 2) → LogSoftmax
```

### Your Notebook's Default (`sage-base`)
```json
{
    "arch_layer": "SAGE",
    "residual": false,
    "norm": false,
    "layers_size": [1024]
}
```

This creates:
```
Input → SAGEConv(input, 1024) → ReLU → Dropout(0.5)
      → Linear(1024, 2) → LogSoftmax
```

**Differences:**
- Missing **second SAGEConv layer** (1-layer vs 2-layer)
- Missing **NodeNorm** (normalization)
- Missing **Residual connections**

---

## 5. 📊 Graph Construction Comparison

| Aspect | Original Code | Impact |
|--------|--------------|--------|
| **Method** | kNN with cosine similarity | Graph topology depends heavily on feature quality |
| **k (train)** | 16 (`n_neigh_train`) | Your notebook uses default 16 — OK if arg passed |
| **k (full)** | 16 (`n_neigh_full`) | Same |
| **Feature for Graph** | Concatenated `[image_features, text_features]` | Graph built on fused features |
| **Semi-supervised Split** | Labeled + Unlabeled nodes in graph | Both labeled and unlabeled tweets used |

The graph construction is identical in code (`generate_graph()` in `gnn_utils.py`), but the quality depends entirely on whether features are correctly loaded.

---

## 6. 🔄 Fusion Strategy Comparison

### Original (Late Fusion with `BaseGNNLateFusion`)
```python
# Splits concatenated features back into image/text
x_image = x[:, :input_size_image]
x_text  = x[:, input_size_image:]

# Separate GNN branches
pred_image = gnn_image(x_image, edge_index)  # SAGEConv → LogSoftmax
pred_text  = gnn_text(x_text, edge_index)    # SAGEConv → LogSoftmax

# Average fusion
x = (pred_image + pred_text) / 2  # or linear fusion
return LogSoftmax(x)
```

### Your Notebook (defaults to Early Fusion — `BaseGNN`)
```python
# Single GNN on concatenated features
x = SAGEConv(concat_features, edge_index)
return LogSoftmax(linear(x))
```

**⚠️ Your notebook defaults to Early Fusion** because `--fusion` is not specified, but the paper uses **Late Fusion**.

---

## 7. 📐 Dimensionality Reduction Comparison

### Original (PCA or Autoencoder)
```python
# PCA to 256 dimensions
pca = PCA(n_components=256, random_state=13)
pca.fit(all_images)  # fit on labeled + unlabeled
ft_labeled_images = pca.transform(ft_labeled_images)
# Same for text

# OR Autoencoder
autoenc = Autoencoder(input_dim, 'autoenc-base')  # → 256 dim
# Train for 2000 epochs, then encode features
```

### Your Notebook
**❌ No reduction applied** — Raw 512-dim CLIP features are used directly.

This means:
- Graph construction uses 512-dim vectors (instead of 256-dim)
- GNN input size is 1024 (512+512) vs 512 (256+256) in original
- Feature space may be less optimal for few-shot learning

---

## 8. 📁 Data Split & Path Comparison

### Original Data Structure
```
data/CrisisMMD_v2.0_baseline_split/
├── data_splits/informative_orig/          # Full splits
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── data_splits_ssl/{size}_s{id}/          # Semi-supervised splits
    └── informative_orig/
        ├── train.jsonl        # Labeled subset (50/100/250/500 samples)
        ├── unlabeled.jsonl    # Unlabeled data
        └── dev.jsonl          # Dev/test set

data/CrisisMMD_v2.0/                      # Images directory
```

### Your Data Structure
```
data/CrisisMMD_v2.0/
├── data_image/     # Images
├── train.jsonl     # ← Wrong location? Original uses data_splits
├── dev.jsonl
└── test.jsonl
```

**⚠️ Your notebook reads from `data/CrisisMMD_v2.0/` directly**, while the original code:
1. Reads **full** train/dev/test from `data_splits/informative_orig/` for **feature extraction**
2. Reads **semi-supervised** splits from `data_splits_ssl/{size}_s{id}/informative_orig/` for **training**

This is a **fundamental structural difference** — the semi-supervised learning setup requires both the full dataset (for feature extraction) and the subset splits (for labeled/unlabeled partitioning).

---

## 9. 🧪 Evaluation Comparison

| Aspect | Original Code | Your Notebook |
|--------|--------------|---------------|
| **Metric** | Weighted F1 + Balanced Accuracy | No evaluation code |
| **Test Set** | Dev set from semi-supervised split | No test evaluation |
| **Confusion Matrix** | Generated and saved | Not implemented |
| **Results** | Saved to `results/CrisisMMD/gnn/{exp_id}/{split}_{run_id}.json` | Not saved |
| **Cross-validation** | 10 splits × 4 sizes = 40 runs per experiment | Single run |

---

## 10. ✅ Summary: What's Correct vs What's Wrong

### ✅ Correct
1. **CLIP model choice** (ViT-B/32) — 512-dim features
2. **Basic label mapping** — informative=1, not_informative=0
3. **General pipeline concept** — extract features → fuse → GNN classification

### 🔴 Critical Issues (Will cause crash or wrong results)
1. **Missing Text Features** — Only image features extracted, no `clip_text_features.pkl`
2. **Missing Required Arguments** — `--lbl_train_frac` and `--exp_id` are required and not provided
3. **Invalid Argument** — `--gnn_layer` doesn't exist in the argparse
4. **Wrong Architecture** — Defaults to `sage-base` (1-layer, no norm, no residual) instead of `sage-2l-norm-res`
5. **Wrong Fusion** — Defaults to `early` instead of `late`
6. **Missing Data Path Arguments** — `--imagepath` and `--datasplit` not provided
7. **No Feature Normalization** — CLIP features not L2-normalized like original

### ⚠️ Important Differences
1. **Different Base Models** — CLIP vs MaxViT/MPNet (intentional per paper's alternate experiments)
2. **No Dimensionality Reduction** — Missing PCA/Autoencoder step
3. **No Semi-supervised Splits** — Data structure doesn't match SSL setup
4. **Missing Hyperparameters** — epochs, weight_decay, l2_lambda, early_stopping not set
5. **Different Data Paths** — Image path structure differs

---

## 11. 🛠️ Recommended Fixes (Priority Order)

### Priority 1: Fix Critical Crashes

**1.1 Add Text Feature Extraction:**
```python
# Add to extract_clip_features():
# After image features, add text feature extraction:
texts = [clean_text(d['text']) for d in valid_data]
text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_embeddings = text_features.cpu().numpy()

# Save separately:
with open('cached_features/clip_text_features.pkl', 'wb') as f:
    pickle.dump(text_output_dfs, f)
```

**1.2 Fix Training Arguments:**
```python
cmd = [
    'python', 'feature_fusion.py',
    '--gpu_id', GPU_ID,
    '--imageft', 'clip_image',
    '--textft', 'clip_text',
    '--arch', 'sage-2l-norm-res',      # 2-layer SAGE with norm + residual
    '--fusion', 'late',                 # Late fusion
    '--reduction', 'pca',              # PCA reduction
    '--pca_red', '256',                # Reduce to 256 dimensions
    '--epochs', '2000',
    '--lr', '1e-5',
    '--weight_decay', '1e-3',
    '--lbl_train_frac', '0.4',         # REQUIRED
    '--exp_id', '1',                   # REQUIRED
    '--datasplit', '250_s0',           # Semi-supervised split
    '--imagepath', './data/CrisisMMD_v2.0/',
    '--n_neigh_train', '16',
    '--n_neigh_full', '16',
    '--reg', 'l2',
    '--l2_lambda', '1e-2',
    '--dropout', '0.5',
    '--shuffle_split',
    '--early_stopping', '300',
    '--best_model', 'best_hm',
    '--loss', 'nll',
    '--run_id', '0',
]
```

### Priority 2: Align with Paper

**2.1 Add Feature Normalization:**
```python
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings, norm='l2')  # L2 normalize like original
```

**2.2 Set up SSL Data Splits** (or modify code to work without them)

### Priority 3: Evaluation
- Add evaluation metrics (F1, Balanced Accuracy)
- Add confusion matrix plotting
- Add multi-run support for statistical significance

---

## 12. 📋 File-by-File Mapping

| Original File | Purpose | Your Equivalent | Notes |
|---------------|---------|-----------------|-------|
| `feature_extraction.py` | MaxViT/MobileNet/MPNet | Cell 3 (CLIP only) | Different models, missing text |
| `feature_fusion.py` | Full pipeline orchestrator | Cell 4 (calls it) | Missing most args |
| `src/arch/gnn_arch.py` | Model definitions | Unchanged (from repo) | Architecture selection wrong |
| `src/gnn/gnn_utils.py` | Training/eval/graph util | Unchanged (from repo) | OK if args correct |
| `src/arch/arch.json` | Architecture configs | Unchanged (from repo) | Using wrong config |
| `scripts/base/exp_1.sh` | Best experiment config | Cell 4 training args | Major gaps |
| `preprocess.py` | Text preprocessing | `clean_text()` in Cell 3 | Different approach |
