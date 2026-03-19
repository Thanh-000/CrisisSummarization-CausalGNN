# CausalCrisis System Architecture - V15++ (Research Sync)

> **Entry point:** `causal_crisis_v2_training.ipynb`  
> **Scope:** CrisisMMD v2.0, Task2 (current notebook: 8 -> 5 merged classes), multimodal causal classification  
> **Last updated:** 2026-03-14

---

## 1. Mục tiêu hệ thống

V15++ tối ưu đồng thời 3 trục:
1. **Robustness nhân quả:** tách `xc`/`xs` rõ hơn, giảm shortcut.
2. **Ổn định đồ thị:** giảm noisy edges trong batch graph.
3. **Hiệu năng lớp hiếm:** giảm chênh lệch giữa weighted-F1 và macro-F1.

---

## 2. Kiến trúc hiện tại

### 2.1 Backbone và fusion
- CLIP feature extraction: `ViT-L-14 (openai)`, batch encoding + OOM fallback.
- Cross-modal fusion dùng attention dạng **multi-token** (không còn SeqLen=1 trivial).

### 2.2 Causal stack (V2 -> V9 -> V15++)
- `V2`: base projection + disentangle + graph + classifier.
- `V9`: domain adversarial branch + causal linker.
- `V15++`:
  - deeper disentangler (`causal_head_plus`, `spurious_head_plus`)
  - hybrid linker (`CausalTransformerLinker` + `BilinearResidualLinker` + gated fusion)
  - `graph_gate` cho graph residual
  - real MTL heads: `head_task1`, `head_task3`

### 2.3 Graph construction
- Graph build trên **causal features** (không dùng raw CLIP concat).
- kNN + mutual-kNN + threshold pruning.
- Warmup graph theo epoch schedule (không hard-code).

### 2.4 Backdoor adjustment
- Memory bank lưu `xs` + domain id.
- Stratified sampling theo domain.
- BA dùng pool đủ lớn (`_valid_size()`), không phụ thuộc `ptr`.

---

## 3. Huấn luyện và loss

### 3.1 Loss tổng
- Main loss: CE + Focal theo phase.
- Regularizers: domain loss, orthogonality, HSIC (non-linear independence), reconstruction, intervention consistency.
- Gradient clipping `max_norm=1.0`.

### 3.2 Phase schedule
- Dùng `phase_for_epoch(epoch)` theo `max_epochs`.
- `warmup_epochs = max(3, int(0.2 * max_epochs))`.
- `graph_warmup_epochs` mặc định sync với warmup (set `None` để auto-sync).

### 3.3 Checkpoint criterion (đã thống nhất)
- Single-run, multi-seed, ablation đều dùng:
  - `score = 0.7 * f1_raw + 0.3 * macro_raw`

---

## 4. Dữ liệu và MTL alignment

### 4.1 Label pipeline
- Task2 giữ canonical 8-class vocab để bảo toàn mapping gốc và cho phép remap linh hoạt (8->6 hoặc 8->5).
- Task1/Task3 dùng vocab theo task, lưu `classes.npy` riêng.
- Sprint hiện tại đang dùng merge policy 8->5:
  - `affected_individuals + injured_or_dead + missing_or_found -> affected_merged`
  - `vehicle_damage -> infrastructure_and_utility_damage_merged`

### 4.2 Sample-level ID alignment
- Extraction lưu thêm `*_ids.npy` cho từng split/task.
- `create_merged_loader(..., include_aux=True)`:
  - ưu tiên align aux labels theo ID
  - nếu có ID nhưng mismatch -> **skip MTL labels** (an toàn)
  - nếu thiếu ID -> fallback index mode + warning

---

## 5. Evaluation protocol

### 5.1 Leakage-safe calibration
1. Chọn track `raw/ba` trên validation.
2. Tối ưu thresholds trên validation.
3. Quyết định dùng calibration trên validation.
4. Khóa policy và evaluate test.

### 5.2 Threshold search
- Đã vectorize `find_best_thresholds_scientificly(...)` (500 candidates/class) để giảm thời gian.

### 5.3 Ensemble BA
- BA pools xây từ train loader, có thể stratify theo domain để gần phân phối memory-bank lúc train.

---

## 6. Các lỗi kiến trúc đã fix

1. **Double-GNN path ở V15 forward:** parent call đổi sang `adj=None`.
2. **Ablation hook sai method:** patch đúng `_build_adj_from_causal`.
3. **BA gate theo `ptr`:** đổi sang `_valid_size()`.
4. **MemoryBank truncate:** đổi sang circular write đầy đủ.
5. **Early-stopping inconsistency:** thống nhất checkpoint score giữa các block train.
6. **Threshold brute-force chậm:** vectorized implementation.
7. **MTL alignment theo index:** bổ sung ID-based alignment + safe fallback policy.

---

## 7. Trạng thái thực nghiệm

| Hạng mục | Trạng thái |
|:--|:--|
| V15++ code sync với notebook | ✅ |
| Core bug fixes (critical/high) | ✅ |
| MTL ID-alignment support | ✅ |
| Full benchmark rerun sau patch mới nhất | ⏳ |
| Cập nhật số liệu báo cáo/paper | ⏳ |

---

## 8. Hướng chạy tiếp khuyến nghị

1. Set `REQUIRE_IDS_FOR_MTL=True` và rerun extraction nếu muốn MTL alignment nghiêm ngặt.
2. Chạy lại single-seed sanity (10-15 epochs) để kiểm tra stability.
3. Chạy full multi-seed + ensemble + ablation với cùng protocol.
4. Báo cáo `mean ± std` cho Weighted-F1, Macro-F1, Balanced Accuracy và per-class F1.

---

## 9. Tai lieu chot ket qua

- Checklist chot nghien cuu: `docs/ai/implementation/v15pp-success-criteria-checklist.md`
- Auto scorecard script: `evaluation/generate_v15pp_scorecard.py`
- Output scorecard moi nhat (auto-generated): `docs/ai/implementation/v15pp-scorecard-latest.md`
