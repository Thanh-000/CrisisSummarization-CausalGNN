# So Sánh Phương Pháp: GNN (IEEE Access 2025) vs DiffAttn (CVPRw MMFM 2025)

> **Ngày**: 26/02/2026  
> **Dataset**: CrisisMMD v2.0  
> **Hardware**: Google Colab L4/T4 GPU

---

## 1. Tổng Quan Hai Phương Pháp

| Tiêu chí | GNN (IEEE Access 2025) | DiffAttn (CVPRw MMFM 2025) |
|----------|------------------------|---------------------------|
| **Paper** | Multimodal Classification with GNN & Few-Shot Learning | Differential Attention for Multimodal Crisis Event Analysis |
| **Kiến trúc** | GraphSAGE + CLIP Late Fusion | CLIP + Differential Attention (CrisiKAN) |
| **Image Encoder** | CLIP ViT-B/32 (512-d) | CLIP ViT-B/32 (512-d) |
| **Text Encoder** | CLIP Multilingual (512-d) | CLIP Tokenizer (512-d) + LLaVA captions |
| **Fusion** | Late Fusion (PCA 256-d) | Differential Attention mechanism |
| **Graph?** | ✅ GraphSAGE (cosine similarity graph, KNN=16) | ❌ Không sử dụng graph |
| **Few-shot?** | ✅ 50, 100, 250, 500 labeled samples | ❌ Full supervised |
| **Optimizer** | Adam (lr=1e-5) | SGD (lr=0.001) |
| **Epochs** | 2000 (early stop 300) | 80 (early stop ~5) |
| **Notebook** | `mm_class_experiment.ipynb` | `new_colab_notebook.ipynb` |

---

## 2. Kết Quả So Sánh

### 2.1 Task 3: Damage Severity (3 classes: Little/No, Mild, Severe)

Đây là task chính để so sánh vì cả hai phương pháp đều báo cáo kết quả trên Task 3.

| Phương pháp | Accuracy | Macro F1 | Weighted F1 | Ghi chú |
|-------------|----------|----------|-------------|---------|
| **DiffAttn (Paper)** | ~75.0% | ~72.0% | — | Paper Table 2 |
| **DiffAttn (Ours)** | *chạy trên Colab* | *chạy trên Colab* | *chạy trên Colab* | `new_colab_notebook.ipynb` Cell 36 |
| **GNN CLIP+PCA (250 labels)** | — | — | 76.5% ± 1.2 | 10 splits, best config |
| **GNN CLIP+PCA (100 labels)** | — | — | 76.2% ± 1.0 | Few-shot setting |
| **GNN Paper (CLIP, 250 labels)** | — | — | 77.1% ± 0.4 | Paper reference |
| **Sirbu Baseline (250 labels)** | — | — | 71.9% | Traditional baseline |

> **Lưu ý**: GNN reports **Weighted F1** (mean over 10 random splits), trong khi DiffAttn reports **Accuracy** và **Macro F1** trên 1 split. Hai metrics này không trực tiếp comparable nhưng cho thấy xu hướng.

### 2.2 Task 1: Informative vs Not Informative (Binary)

| Phương pháp | Accuracy | Macro F1 |
|-------------|----------|----------|
| **DiffAttn (Paper)** | ~87.0% | ~86.0% |
| **DiffAttn (Ours)** | *xem Cell 36* | *xem Cell 36* |

### 2.3 Task 2: Information Type (8 classes)

| Phương pháp | Accuracy | Macro F1 |
|-------------|----------|----------|
| **DiffAttn (Paper)** | ~70.0% | ~65.0% |
| **DiffAttn (Ours)** | *xem Cell 36* | *xem Cell 36* |

---

## 3. So Sánh Kiến Trúc Chi Tiết

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    GNN Pipeline (IEEE Access 2025)                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Image ──► CLIP ViT-B/32 ──► 512-d ──┐                                   │
│                                       ├──► PCA(256) ──► Concat(512)       │
│  Text  ──► CLIP Multilingual ──► 512-d ┘                                  │
│                                              │                             │
│                                              ▼                             │
│                                    KNN Graph (k=16)                        │
│                                              │                             │
│                                              ▼                             │
│                                    GraphSAGE (2L)                          │
│                                    + BatchNorm                             │
│                                    + Residual                              │
│                                              │                             │
│                                              ▼                             │
│                                       NLL Loss                             │
│                                       Few-shot                             │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                    DiffAttn Pipeline (CVPRw MMFM 2025)                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Image ──► CLIP ViT-B/32 ──┐                                              │
│                              ├──► Differential Attention ──► Fusion        │
│  Text  ──► CLIP Tokenizer ──┘    (MultiheadDiffAttn)                      │
│            + LLaVA captions                                                │
│                                              │                             │
│                                              ▼                             │
│                                    CrisiKAN Layers                         │
│                                    (KAN activation)                        │
│                                              │                             │
│                                              ▼                             │
│                                    CrossEntropy Loss                       │
│                                    Full supervised                          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phân Tích Ưu Nhược Điểm

| Tiêu chí | GNN | DiffAttn |
|----------|:---:|:--------:|
| **Performance** | 🟢 Tốt (matches paper) | 🟡 Khá (needs tuning) |
| **Few-shot learning** | 🟢 Xuất sắc (50-250 labels) | 🔴 Không hỗ trợ |
| **Data efficiency** | 🟢 Cần ít labeled data | 🔴 Cần full dataset |
| **Training speed** | 🟡 Chậm (2000 epochs × 10 splits) | 🟢 Nhanh (80 epochs) |
| **Inference speed** | 🟡 Cần graph construction | 🟢 Direct forward pass |
| **Interpretability** | 🟡 Graph structure | 🟢 Attention weights |
| **Text quality** | 🟡 Raw CLIP encoding | 🟢 LLaVA enhanced captions |
| **Scalability** | 🔴 O(n²) graph | 🟢 O(n) per sample |

---

## 5. Điểm Mạnh Riêng

### GNN (IEEE Access)
1. **Few-shot learning**: Chỉ cần 50-250 labeled samples mà đạt F1 ≈ 75-76%
2. **Graph propagation**: Tweets tương tự "khuếch đại" lẫn nhau → robust hơn
3. **Reproducible**: Kết quả ±0.7pp so với paper qua 10 random splits

### DiffAttn (CVPRw MMFM)
1. **Differential Attention**: Phát hiện cả alignment (HAM) LẪN contradiction (CAM)
2. **LLaVA captions**: Làm giàu text input qua image description  
3. **Multi-task**: Train được cả 3 tasks (informative, type, severity)
4. **End-to-end**: Không cần feature extraction + graph construction riêng

---

## 6. Kết Luận

| Kịch bản sử dụng | Phương pháp tốt hơn | Lý do |
|-------------------|---------------------|-------|
| **Ít labeled data** (< 500) | 🏆 **GNN** | Few-shot learning vượt trội |
| **Full supervised** | 🏆 **DiffAttn** | Attention mechanism mạnh hơn |
| **Real-time inference** | 🏆 **DiffAttn** | Không cần graph construction |
| **Misinformation detection** | 🏆 **DiffAttn** | CAM phát hiện text-image contradiction |
| **Large-scale deployment** | 🏆 **DiffAttn** | O(n) scalability |
| **Robustness** | 🏆 **GNN** | Graph propagation giảm noise |

> **Tổng kết**: Hai phương pháp bổ sung cho nhau. GNN vượt trội trong few-shot setting, DiffAttn mạnh ở full-supervised và real-time inference. Kết hợp cả hai (DiffAttn features → GNN graph) có thể là hướng nghiên cứu tiếp theo.
