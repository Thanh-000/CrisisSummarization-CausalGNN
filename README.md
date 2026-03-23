# CausalCrisis V3 — Multimodal Causal Classification for Crisis Events

> **Per-Modality Causal Disentanglement + Cross-Modal Causal Fusion for Domain-Generalizable Crisis Classification**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L%2F14-green.svg)](https://github.com/openai/CLIP)

---

## 🎯 Overview

CausalCrisis V3 is a multimodal causal classification framework for crisis/disaster events that achieves **>90% F1** on CrisisMMD by combining:

1. **CLIP ViT-L/14** frozen features (768-dim, pre-cached)
2. **Hybrid ICA-Adversarial Disentanglement** (per-modality: visual + text)
3. **Cross-Modal Causal Factor** (C_vt via gated cross-attention)
4. **Supervised Contrastive Loss** on causal features
5. **Backdoor Adjustment** at inference (do-calculus causal intervention)

### Key Novelties

| Feature | Description |
|:--------|:-----------|
| 🆕 **Hybrid Disentanglement** | ICA init + adversarial refinement (inspired by CCA, Jiang 2025) |
| 🆕 **Per-Modality Causal** | Separate visual/textual spurious factors (vs CAMO joint) |
| 🆕 **Adaptive Loss Weighting** | Auto-balance 4+ losses (Kendall uncertainty) |
| 🆕 **SupCon on Causal** | Force discriminative causal representations |
| 🆕 **Bilinear Fusion Option** | Simpler alternative to cross-attention |

---

## 📂 Project Structure

```
CrisisSummarization/
├── src/                          ← Core implementation
│   ├── config.py                 ← Centralized hyperparameters
│   ├── models.py                 ← CausalCrisisV3 + Baseline
│   ├── losses.py                 ← Focal, Ortho, SupCon, Adaptive
│   ├── data.py                   ← CrisisMMD loader + CLIP caching
│   ├── trainer.py                ← 2-phase training loop
│   └── evaluate.py               ← Metrics, LODO, ablation, t-SNE
│
├── notebooks/
│   └── causalcrisis_v3_experiment.py  ← Colab experiment script
│
├── experiments/                  ← Experiment protocols & results
├── docs/                         ← Research documentation
├── .agent/skills/                ← 13 AI research skills
│
├── research-state.yaml           ← Autoresearch state tracking
├── findings.md                   ← Research findings (updated)
└── research-log.md               ← Decision log
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on Google Colab (Recommended)

```python
# Clone and run
!git clone https://github.com/Thanh-000/CrisisSummarization-CausalGNN.git
%cd CrisisSummarization-CausalGNN

# Upload CrisisMMD v2.0 dataset
# Then run experiment script
%run notebooks/causalcrisis_v3_experiment.py
```

### 3. Run Locally

```python
from src.config import get_config
from src.models import CausalCrisisV3
from src.losses import CausalCrisisLoss

config = get_config("task1")
model = CausalCrisisV3(
    input_dim=768,
    causal_dim=384,
    num_classes=2,
    use_ica_init=True,
    fusion_type="cross_attention",
)
print(f"Parameters: {model.get_trainable_params():,}")
```

---

## 🏗️ Architecture

```
Input → CLIP ViT-L/14 (frozen, cached)
  ↓
Per-Modality Hybrid ICA-Adversarial Disentanglement
  ├── Visual: f_v → (C_v, S_v)
  └── Text:   f_t → (C_t, S_t)
  ↓
Cross-Modal Causal Fusion
  └── C_vt = CrossAttn(C_v, C_t) OR Bilinear(C_v, C_t)
  ↓
Classification: concat(C_v, C_t, C_vt) → MLP → ŷ
  ↓ (inference)
Backdoor Adjustment: P(Y|do(C)) = Σ_s P(Y|C,s)·P(s)
```

### Loss Function

```
L = (1/σ₁²)·L_focal + (1/σ₂²)·L_adv + (1/σ₃²)·L_ortho + (1/σ₄²)·L_supcon + Σlog(σᵢ)
```

Where σ₁-σ₄ are learnable (Adaptive Loss Weighting).

---

## 📊 Expected Results

| Method | F1 (weighted) | Notes |
|:-------|:-------------|:------|
| CLIP + MLP Baseline | ~88% | H1 experiment |
| + Hybrid Disentangle | ~90% | H2/H5 |
| + C_vt + SupCon | ~91.5% | H3/H7 |
| **Full V3** | **~92-93%** | H6 target |
| CrisisSpot (2025) | 90.9% | Current SOTA |
| CAMO (2025) | ~85% | VGG+BERT |

---

## 📚 Key References

1. **CCA** (Jiang et al., 2025) — ICA disentanglement for CLIP features
2. **CAMO** (Ma et al., 2025) — Causal adversarial multimodal DG
3. **CIRL** (Lv et al., CVPR 2022) — Causal representation learning
4. **CrisisSpot** (Dar et al., ESWA 2025) — Current SOTA on CrisisMMD
5. **Sun et al.** (ICLR 2025) — Multimodal causal identifiability

---

## 📋 Research Skills (13 Integrated)

This project uses [AI Research Skills](https://github.com/Orchestra-Research/AI-Research-SKILLs):

| Skill | Purpose |
|:------|:--------|
| `autoresearch` | Two-loop experiment orchestration |
| `multimodal-clip` | CLIP feature extraction |
| `evaluation` | Metrics, LODO, ablation, significance |
| `ml-paper-writing` | Paper writing (IEEE Access / AAAI) |
| `research-ideation` | Brainstorming + creative thinking |
| `data-processing` | Dataset pipeline design |
| `debug` | Evidence-first debugging |
| `dev-lifecycle` | SDLC workflow |
| `capture-knowledge` | Knowledge documentation |

---

## 📜 License

MIT License

## 👥 Authors

CausalCrisis V3 Research Team
