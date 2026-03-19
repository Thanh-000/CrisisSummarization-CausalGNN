# Research Log — CausalCrisis V3

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-03-19 | bootstrap | Project restart: CausalCrisis V3 with CLIP ViT-L/14 + per-modality causal disentanglement |
| 2 | 2026-03-19 | bootstrap | Literature scan: 9 key papers identified, including new CCA (2025) and CLIP-DCA (2025) |
| 3 | 2026-03-19 | bootstrap | 7 hypotheses formed (H1-H7), prioritized by dependency chain |
| 4 | 2026-03-19 | bootstrap | Skills integrated: autoresearch, multimodal-clip, evaluation, ml-paper-writing, research-ideation, data-processing |
| 5 | 2026-03-19 | bootstrap | Architecture analysis using research-ideation frameworks → 5 improvement proposals generated |
| 6 | 2026-03-19 | implement | Created `src/config.py` — centralized hyperparameters with all 5 improvements |
| 7 | 2026-03-19 | implement | Created `src/models.py` — HybridDisentangler (ICA+Adv), CrossAttention/Bilinear fusion, GRL, BA, full V3 pipeline |
| 8 | 2026-03-19 | implement | Created `src/losses.py` — FocalLoss, OrthogonalLoss, SupConLoss (🆕), AdaptiveLossWeighting (🆕) |
| 9 | 2026-03-19 | implement | Created `src/data.py` — CrisisMMD loader, CLIP caching, stratified splits, LODO splits |
| 10 | 2026-03-19 | implement | Created `src/trainer.py` — 2-phase training, EarlyStopping, BaselineTrainer |
| 11 | 2026-03-19 | implement | Created `src/evaluate.py` — metrics, bootstrap significance, LODO runner, ablation framework, t-SNE |
| 12 | 2026-03-19 | implement | Created `notebooks/causalcrisis_v3_experiment.py` — Colab-ready 10-cell experiment script |
| 13 | 2026-03-19 | implement | Created experiment protocol for H1 (CLIP MLP baseline) |
