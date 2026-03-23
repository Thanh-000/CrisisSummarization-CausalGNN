# Research Findings — CausalCrisis V4

## Research Question

**How to achieve >90% F1 on CrisisMMD Task 1 (Informativeness) with causal interpretability, surpassing or matching SOTA (Munia et al. 92.89%)?**

---

## Current Understanding

### Phase 1: V3 Architecture Failure (Completed)

CausalCrisis V3 was designed around **per-modality causal disentanglement** of frozen CLIP features. After extensive debugging (3 rounds), we conclusively proved that:

1. **Frozen CLIP ViT-L/14 features have a ~78-80% F1 ceiling on CrisisMMD** — far below the assumed 88% baseline in the original architecture document.

2. **All V3 auxiliary components (disentanglement, GRL, SupCon, BA) either don't help or actively hurt performance**, adding only +0.3% F1 over an MLP baseline while causing massive overfitting (Train 0.99 vs Val 0.78).

3. **The true SOTA (Munia et al., CVPRw 2025) achieves 92.89% F1w** using CLIP Vision + CLIP Text + **LLaVA captions** + Guided Cross-Attention. The key ingredient is the **third modality (LLaVA)**, not architectural complexity.

### The Mechanism Behind the Ceiling

CLIP features are L2-normalized 768-dim vectors encoding general visual/textual semantics. For crisis classification, they lack:
- **Domain-specific crisis vocabulary** (e.g., "triage", "SAR", infrastructure damage terminology)
- **Fine-grained damage visual patterns** (beyond "building" vs "water")
- **Joint visual-textual reasoning** (e.g., "flooded street" + "help us" → urgent, but each alone is ambiguous)

LLaVA captions bridge this gap by providing **detailed, image-specific textual descriptions** that CLIP can then encode. This converts implicit visual information into explicit textual features that CLIP's text encoder excels at processing.

### The Landscape of Related Work

| Method | Approach | CrisisMMD Task 1 F1w | Key Innovation |
|:-------|:---------|:---------------------|:---------------|
| Munia et al. 2025 | CLIP+LLaVA+Guided CA | **92.89%** | LLaVA caption augmentation |
| CLIP-BCA-Gated 2024 | CLIP+BiCrossAttn+Gating | 91.77% acc | Adaptive noise-robust gating |
| CAMO 2025 | Adversarial disentanglement+unified repr | N/A (DG focus) | Joint disentangle for domain generalization |
| CausalCLIP 2025 | Disentangle-then-filter | N/A (fake detection) | Adversarial mask on frozen CLIP |
| CausalCrisis V3 (ours) | Per-modality disentangle+GRL | 78.3% | Failed due to frozen feature ceiling |

---

## Key Results

### Experiment: V3 Frozen Feature Baseline
- **Setup:** CLIP ViT-L/14 (frozen) → MLP (1536→512→256→2), AdamW lr=3e-4, 100 epochs
- **Result:** 77.8% ± 0.5% F1w (4 seeds)
- **Interpretation:** Frozen features insufficient for >80% F1

### Experiment: V3 Full Model
- **Setup:** Full V3 pipeline (Adapter + HybridDisentangler + CrossAttention + GRL + SupCon + BA)
- **Result:** 78.3% F1w (best seed), massive overfitting
- **Interpretation:** Added ~4M params but only +0.5% F1, all auxiliary losses hurt val performance
- **Phase 2 (auxiliary losses) actively degraded validation F1 by ~4-6% during training**

### Critical Finding: Data Split Mismatch
- **Munia et al.** use official CrisisMMD splits: 9599/1573/1534 (filtered for agreed labels)
- **Our V3** uses random stratified 70/15/15 split on ALL 18082 samples
- **This makes direct comparison impossible** — different data splits = different benchmarks

---

## Patterns and Insights

### What Works
1. **LLaVA captions** — Munia et al. showed +2.44% F1w over Wikipedia captions
2. **Guided Cross-Attention** — Better than standard cross-attention (+0.5-1%)
3. **Frozen CLIP embeddings** — Massively outperform DenseNet+Electra (+4-6% F1w)
4. **Simple architectures** — Munia et al. use SGD, 50 epochs, no complex training protocol

### What Doesn't Work
1. **Per-modality disentanglement on frozen features** — No signal to separate C vs S
2. **GRL adversarial training with few epochs** — Destabilizes training, needs 100+ epochs
3. **SupCon on overfit features** — Memorizes training set
4. **Backdoor Adjustment** — Never activates due to early stopping
5. **Complex multi-phase training** — Overkill for a dataset with only ~10K training samples

### Key Pattern: Feature Quality > Architecture
The dominant factor is **input feature quality** (what information goes into the model), not **how you process it**. Munia et al.'s simple Guided CA + FC layer outperforms our complex V3 pipeline because they have better input features (CLIP + LLaVA).

---

## Lessons and Constraints

1. **Never assume baseline numbers from literature** — always verify with your own data splits
2. **Frozen features have hard ceilings** — no amount of downstream architecture can overcome missing information
3. **Cross-attention with seq_len=1 is mathematically an MLP** — no attention pattern to learn
4. **Adversarial training (GRL) needs long training** and large datasets to converge
5. **CrisisMMD is a small dataset (~18K)** — complex models overfit easily; prefer simpler architectures with strong regularization
6. **LLaVA captions are the single highest-impact addition** for crisis classification
7. **Use official data splits** for fair comparison with published results

---

## Open Questions

1. **Can LoRA fine-tuning of CLIP itself close part of the feature gap without LLaVA?**
2. **Does causal disentanglement add value on TOP of LLaVA-enriched features?**
3. **Is CAMO's joint unified representation better than per-modality disentanglement for LODO?**
4. **What's the cost-benefit of LLaVA inference vs LoRA training?**
5. **Can we combine our causal story with Munia et al.'s practical recipe for a novel contribution?**

---

## Optimization Trajectory

```
Run 1: CLIP MLP Baseline (frozen)     → 77.8% F1w  [Baseline]
Run 2: V3 Full Model (all components) → 78.3% F1w  [+0.5%, insignificant]
── PIVOT ──  
Run 3: [PLANNED] LLaVA caption + concat → target 88%+
Run 4: [PLANNED] Guided CA fusion       → target 90%+
Run 5: [PLANNED] + Causal disentangle   → target 91%+
Run 6: [PLANNED] + LoRA CLIP            → target 93%+
```
