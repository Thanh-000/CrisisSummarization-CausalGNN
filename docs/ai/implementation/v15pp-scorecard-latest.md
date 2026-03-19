# V15++ Auto Scorecard

- Generated at (UTC): `2026-03-14 08:24:43Z`
- Source metrics CSV: `evaluation/results/baseline_5seeds.csv`
- Output file: `docs/ai/implementation/v15pp-scorecard-latest.md`

## Filters

- `ablation_name`: `full_v15_plus`
- `model_name`: `paper1_gnn`
- `task`: `task1`
- `few_shot`: `500`
- `n_runs`: `5`

## KPI Summary

| Metric | Mean | Std | Status |
|:--|--:|--:|:--|
| Weighted-F1 | 0.8413 | 0.0181 | STRETCH |
| Macro-F1 | 0.8216 | 0.0145 | STRETCH |
| Balanced Acc | N/A | N/A | N/A |

## Hard Conditions

- No class with F1 = 0.00: `N/A (missing class report)`
- `vehicle_damage` status: `N/A (missing class report)`
- `vehicle_damage` F1: `N/A`
- `vehicle_damage` Recall: `N/A`
- `infra_damage` Recall: `N/A`

## Seed Runs

| Seed | Weighted-F1 | Macro-F1 | Balanced Acc | Accuracy |
|:--|--:|--:|--:|--:|
| 42 | 0.8499 | 0.8334 | N/A | 0.8548 |
| 123 | 0.8176 | 0.8004 | N/A | 0.8453 |
| 456 | 0.8365 | 0.8156 | N/A | 0.8440 |
| 789 | 0.8662 | 0.8365 | N/A | 0.8587 |
| 1024 | 0.8363 | 0.8223 | N/A | 0.8655 |

## Go/No-Go

- Decision: `INCOMPLETE`

## Notes

- This scorecard is auto-generated from CSV metrics.
- If class report CSV is not provided, rare-class constraints are marked as `N/A`.
