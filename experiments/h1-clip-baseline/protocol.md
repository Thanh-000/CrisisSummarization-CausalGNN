# Experiment Protocol: H1 — CLIP ViT-L/14 MLP Baseline

## Hypothesis
CLIP ViT-L/14 frozen features achieve >88% weighted F1 with simple MLP on CrisisMMD Task 1 (Informative).

## Prediction
- F1 ≈ 87-89% (based on ViT-B/32 ~86% + upgrade to ViT-L/14 ~+1-2%)

## Setup
- **Encoder**: CLIP ViT-L/14 (frozen, pre-cached features)
- **Classifier**: 768 → 512 → 256 → 2 (MLP with LayerNorm + GELU + Dropout)
- **Loss**: Focal Loss (gamma=2.0)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: Cosine annealing → 1e-5
- **Epochs**: 100 (early stop patience=15)
- **Batch size**: 128
- **Seeds**: 5 (report mean ± std)

## Steps
1. Extract CLIP ViT-L/14 features for all CrisisMMD samples → cache .npy
2. Create stratified train/val/test splits (70/15/15)
3. Train MLP classifier (5 seeds)
4. Record: F1, Precision, Recall, Accuracy
5. Save best model checkpoint

## Success Criteria
- F1 > 88% → H1 supported → proceed to H2
- F1 < 85% → investigate feature quality, try ViT-B/16 as backup

## Estimated Time
- Feature extraction: ~10 min on A100
- Training: ~2 min per seed × 5 = 10 min
- Total: ~20 min
