# V15++ Success Criteria Checklist

> Project: CrisisMMD Task2 (merge policy configurable: 8 -> 6 or 8 -> 5)  
> Model line: CausalCrisis V15++  
> Purpose: Chuan chot ket qua de bao cao nghien cuu

---

## 1. KPI Muc Tieu

Baseline tham chieu gan nhat:
- Weighted-F1: ~0.67
- Macro-F1: ~0.52
- Balanced Accuracy: ~0.50

Nguong danh gia:

| Metric | Minimum (Pass) | Target (Good) | Stretch (Strong) |
|:--|:--|:--|:--|
| Weighted-F1 | >= 0.68 | >= 0.70 | >= 0.72 |
| Macro-F1 | >= 0.56 | >= 0.58 | >= 0.60 |
| Balanced Acc | >= 0.54 | >= 0.56 | >= 0.58 |

---

## 2. Dieu Kien Bat Buoc Truoc Khi Nhan Ket Qua

- [ ] Khong co lop nao F1 = 0.00
- [ ] `vehicle_damage` co recall > 0 (hoac trang thai `MERGED` neu dung policy 8->5)
- [ ] `infra_damage` recall >= baseline
- [ ] Macro-F1 khong giam khi Weighted-F1 tang
- [ ] Quy trinh val/test khong leakage (track + threshold + calibration chon tren val)

Neu fail bat ky muc nao tren, khong claim model dat muc toi uu.

---

## 3. On Dinh Thong Ke

- [ ] Chay toi thieu 3 seeds (`42, 123, 2026`)
- [ ] Bao cao `mean +- std` cho Weighted-F1, Macro-F1, Balanced Acc
- [ ] Do lech chuan Macro-F1 <= 0.02 (khuyen nghi)
- [ ] Ket qua test duoc khoa policy truoc khi chay test

---

## 4. Dieu Kien Ablation

Ablation toi thieu:
- [ ] `full_v15_plus`
- [ ] `w_o_mutual_knn`
- [ ] `w_o_bilinear_linker`
- [ ] `w_o_focal_loss`
- [ ] `w_o_macro_selection`

Ky vong:
- [ ] `full_v15_plus` dung top-1 theo `macro_f1_mean`, neu hoa thi uu tien `weighted_f1_mean`
- [ ] Moi module chinh co dong gop duong hoac giai thich duoc trade-off

---

## 5. Readiness De Bao Cao/Paper

- [ ] Co bang ket qua day du (overall + per-class)
- [ ] Co confusion matrix cho test
- [ ] Co phan tich loi cho lop hiem (`infra_damage`, `vehicle_damage`)
- [ ] Co mo ta ro protocol va random seeds de tai lap

---

## 6. Mau Bang Tong Ket De Dien

| Run | Weighted-F1 | Macro-F1 | Bal.Acc | vehicle_damage F1 | infra_damage Recall | Ghi chu |
|:--|--:|--:|--:|--:|--:|:--|
| Seed 42 |  |  |  |  |  |  |
| Seed 123 |  |  |  |  |  |  |
| Seed 2026 |  |  |  |  |  |  |
| Mean +- Std |  |  |  |  |  |  |

---

## 7. Quyet Dinh Go/No-Go

`GO` khi dong thoi thoa:
1. Dat it nhat muc `Pass` cho 3 metric tong.
2. Khong con lop F1=0.00.
3. Ket qua on dinh qua multi-seed.
4. Ablation khong phu dinh dong gop cua kien truc.

`NO-GO` neu:
1. Weighted-F1 tang nhung Macro-F1/Balanced Acc giam manh.
2. Lop hiem van bi bo qua (F1=0.00).
3. Dao dong seed qua lon.

---

## 8. Auto Scorecard (Command)

```bash
python evaluation/generate_v15pp_scorecard.py \
  --metrics-csv results/v15_ablation_raw.csv \
  --ablation-summary-csv results/v15_ablation_summary.csv \
  --class-report-csv results/v15_class_report.csv \
  --ablation-name full_v15_plus \
  --output-md docs/ai/implementation/v15pp-scorecard-latest.md
```
