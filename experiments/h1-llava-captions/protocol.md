# Experiment Protocol: H1 — LLaVA Caption Augmentation

## Hypothesis
**Adding LLaVA captions as a third modality will boost frozen CLIP baseline from 78% to 88%+ F1w on CrisisMMD Task 1.**

## Motivation
Munia et al. (CVPRw 2025) demonstrated that LLaVA captions provide a +2.44% F1w improvement over Wikipedia captions when used with CLIP features. Their best result (92.89% F1w) relies on CLIP Vision + CLIP Text + LLaVA captions. The key insight is that LLaVA generates **image-specific, detailed descriptions** that convert implicit visual information into explicit textual features.

## Prediction
- **Concat+MLP (3 modalities):** 85-88% F1w
- **With Guided CA fusion:** 89-91% F1w
- **With Guided CA + fine-tuning:** 91-93% F1w

## What Changes
1. Extract LLaVA captions for all 18K CrisisMMD images
2. Encode LLaVA captions with CLIP text encoder → `f_c` (768-dim)
3. Concatenate all 3 modality features: `f_v || f_t || f_c` (2304-dim)
4. Train MLP classifier on concatenated features
5. Use **official CrisisMMD splits** (matching Munia et al.)

## Implementation Plan

### Step 1: LLaVA Caption Generation (on Colab A100)
```python
# Use LLaVA-1.5-7B or LLaVA-1.5-13B
# Input: image + prompt "Describe this social media image in detail, 
#        focusing on any crisis, damage, or emergency content."
# Output: text caption (50-200 words)
# Time: ~2-4 hours for 18K images on A100
```

### Step 2: CLIP Encoding of Captions
```python
# Encode captions with same CLIP ViT-L/14 text encoder
# Cache to cached_features/clip_ViT-L_14_llava_text.npy
```

### Step 3: Baseline (Concat + MLP)
```python
# f_combined = concat(f_v, f_t, f_c)  # 2304-dim
# MLP: 2304 → 512 → 256 → 2
# Training: AdamW, lr=3e-4, 50 epochs, batch=32 (match Munia)
```

### Step 4: With Guided CA
```python
# Implement Guided Cross-Attention from Munia et al.
# Self-attention → Projection → Sigmoid masks → Cross-guidance
```

## Sanity Checks
- [ ] LLaVA captions are non-empty and meaningful (spot-check 20 samples)
- [ ] CLIP encoding of captions produces valid features (not all zeros)
- [ ] Data splits match Munia et al. (9599/1573/1534)
- [ ] Baseline (CLIP image+text only) reproduces our previous ~78% F1w
- [ ] No data leakage between splits

## Success Criteria
- **Minimum:** >85% F1w with concat+MLP (proves LLaVA adds value)
- **Target:** >90% F1w with Guided CA
- **Stretch:** >92% F1w (matching Munia et al.)

## Failure Modes
- LLaVA captions too generic → try better prompts or LLaVA-1.5-13B
- Feature dimensions mismatch → verify CLIP encoding pipeline
- Still overfitting → add dropout, label smoothing, mixup
