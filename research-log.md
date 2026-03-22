# Research Log — CausalCrisis V3→V4

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-03-19 | bootstrap | Project initialized: CausalCrisis V3 with per-modality causal disentanglement targeting >90% F1 on CrisisMMD |
| 2 | 2026-03-22 | inner-loop | V3 baseline (frozen CLIP+MLP): 77.8% F1w — far below expected 88% |
| 3 | 2026-03-22 | inner-loop | V3 full model: 78.3% F1w — disentanglement+GRL+SupCon adds only +0.5% |
| 4 | 2026-03-22 | inner-loop | Diagnosed: CosineAnnealingWarmRestarts was killing LR before Phase 2. Fixed to CosineAnnealingLR. |
| 5 | 2026-03-22 | inner-loop | Diagnosed: Phase 2 auxiliary losses actively degrade val F1 (0.81→0.75) |
| 6 | 2026-03-22 | inner-loop | Diagnosed: Frozen CLIP features have hard ceiling ~78-80% F1 regardless of downstream architecture |
| 7 | 2026-03-22 | outer-loop | **REFLECTION CYCLE 1:** V3 architecture fundamentally limited. Feature quality is the bottleneck, not architecture. |
| 8 | 2026-03-22 | outer-loop | Literature review: Munia et al. 2025 achieves 92.89% F1w with CLIP+LLaVA+Guided CA |
| 9 | 2026-03-22 | outer-loop | Literature review: CAMO 2025 uses joint disentanglement (not per-modality like V3) |
| 10 | 2026-03-22 | outer-loop | Literature review: CLIP-BCA-Gated 2024 achieves 91.77% acc with bidirectional cross-attention |
| 11 | 2026-03-22 | outer-loop | Literature review: CausalCLIP 2025 uses disentangle-then-filter on CLIP (AAAI 2026) |
| 12 | 2026-03-22 | pivot | **PIVOT to V4:** Drop per-modality disentanglement. Add LLaVA captions + Guided CA. Keep causal story for LODO. |
| 13 | 2026-03-22 | outer-loop | **DIRECTION: PIVOT** — New research question: Can causal disentanglement add value on TOP of LLaVA-enriched features? |
| 14 | 2026-03-22 | bootstrap | Formed 7 hypotheses (H1-H7). Priority: H1 (LLaVA), H2 (Guided CA), H6 (full pipeline) |
| 15 | 2026-03-22 | inner-loop | V4 experiment notebook created: 5 experiments (A1: concat 2-modal, A2: GCA 2-modal, B1: concat 3-modal, B2: GCA 3-modal, C1: GCA separate encoding). Includes: checkpoint resume for LLaVA, dual encoding strategy (combined vs separate), official CrisisMMD splits. Ready for Colab A100 execution. |
