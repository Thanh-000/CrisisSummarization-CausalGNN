# Phase 3: Kiến Trúc Causal GNN v2 & Đồng Bộ Paper (Alignment)

**Ngày cập nhật:** 2026-03-11  
**Trạng thái:** Hoàn tất đồng bộ (Aligned & Validated)  
**Tập trung vào Task:** Task 3 (Damage Assessment)  

## Mục Tiêu
Đồng bộ hoàn toàn mã nguồn thực thi (Colab Notebook `causal_crisis_v2_training.ipynb`) với những mô tả lý thuyết, công thức toán học và thiết kế architecture trong **[Research Paper: Multimodal Classification of Social Media Disaster Posts with GNN]**. 

Báo cáo này lưu trữ những điều chỉnh trọng yếu nhất trong giai đoạn hoàn thiện mô hình.

---

## 🏗️ 1. Cập Nhật Architecture (CausalCrisisV2Model)

Nhận thấy sự sai lệch giữa bản V1 (Dùng MLP phẳng) và mô tả trong lý thuyết (Dùng GCN và disentanglement), kiến trúc đã được đại tu:

*   **Explicit Disentanglement Branches:**
    Phân tách mạnh mẽ Feature thành 2 nhánh minh bạch theo nguyên lý Causal:
    *   `causal_head` ($P_m$ Projection) $\rightarrow X_c$: Trích xuất các đặc trưng nguyên nhân cốt lõi (Bất biến).
    *   `spurious_head` $\rightarrow X_s$: Trích xuất các đặc trưng gây nhiễu, liên quan đến ngữ cảnh (Biến đổi).
*   **Chuẩn Hoá GCN Layer:**
    Áp dụng công thức gốc của Graph Convolutional Networks thay thế cho lớp xử lý phẳng:
    *   Toán học: $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W$
    *   Ý nghĩa: Giúp mô hình thu thập bằng chứng từ cấu trúc Graph (lan truyền thông điệp giữa các điểm dữ liệu lân cận) thông qua k-NN adjacency matrix.

## 🧮 2. Backdoor Adjustment (Can Thiệp Nhân Quả)

Trong quá trình Evaluate và Test, mô hình phải khử sự phụ thuộc ảo do $X_s$ tạo ra:

*   **Monte Carlo Sampling ($N=50$):**
    Điều chỉnh logic trong `CausalTrainer.evaluate()` và `calibrate_and_report()`. Với mỗi prediction batch, mô hình lấy mẫu $N=50$ vectors $X_s$ (Spurious Features) từ một `MemoryBank` kích thước 2000.
*   **Averaging:** Tiến hành Forward-pass $N$ lần cho mỗi cụm $X_c$ với các $X_s$ khác nhau, sau đó trung bình hoá (*Average*) logits để triệt tiêu ảnh hưởng của Spurious Bias. Ghi nhận thành công chỉ số **Backdoor Adjusted Logits (`logits_ba`)**.

## ⚙️ 3. Tối Ưu Chiến Lược Đào Tạo (Training Strategy)

Chốt sổ cấu hình Multi-Seed Experiment (Seeds: 42, 2024, 2025) để báo cáo cáo kết quả cuối cùng:

*   **Epochs (Tổng: 40):**
    *   `PHASE_1` (15 Epochs đầu): Warmup, chỉ dùng Base Classifier (không Graph, không Backdoor).
    *   `PHASE_2_AND_3` (25 Epochs sau): Kích hoạt hệ thống Graph Reasoning và châm ngòi Backdoor Adjustment.
*   **Learning Rates (AdamW):**
    *   GNN Layers (Các layer mới, Graph): `1e-4` (Phát triển nhanh).
    *   Non-GNN Layers (Các Backbone và Projection cũ): `2e-5` (Tinh chỉnh).

## 🐛 4. Khắc Phục Lỗi Hệ Thống Sinh Báo Cáo (Bug Fixes)

Trong quá trình phân tích SOTA Calibration (`calibrate_and_report`) và xuất hình ảnh Gallery (`export_significant_cases_with_media`), tiến hành vá các lỗi:

1.  **AttributeError (ln_f, fusion):** Xoá bỏ các lệnh gọi attribute đã lỗi thời của bản cũ, thay bằng workflow lấy trực tiếp `out_pre = model(img, txt)` và `out_pre['xc']` tạo Adjacency matrix.
2.  **TypeError (allow_pickle=True):** Di chuyển keyword này vào trong tham số của `np.load()` tương thích với các version NumPy hiện đại (1.16+).
3.  **NameError & Null-Safety Logic:** Khắc phục lỗi báo cáo Logic Multi-Class ở Gallery, cài đặt Fetching Logits an toàn với Backdoor `get('logits_ba', logits)`. Đảm bảo code chạy tự động với Task N-class bất kỳ.

---

## ✅ Kết Quả Validate

*   Hệ thống đào tạo trơn tru 40 Epochs trên GPU T4 (Colab).
*   Đồ thị **t-SNE** cho thấy sự phân cụm rõ ràng ở `Phase 2` (Graph-enhanced Feature $X_c^{graph}$) so với `Phase 1` (Raw Feature $X_c$).
*   Trích xuất thành công **Causal Success Cases**: Tìm thấy và hiển thị những ví dụ thực tễ (Rank #4, #5) nơi Backdoor Adjustment đảo ngược quyết định (Sai $\rightarrow$ Đúng) nhờ cộng hưởng Confidence Gain $\sim 13\%$.

### 📊 Báo Cáo Kết Quả Cuối Cùng (Task 3 - Damage Assessment)
*Kết quả dưới đây thu được sau khi chạy Multi-seed experiment (Seeds: 42, 2024, 2025)*

| Metric | Mean ± Std |
| :--- | :--- |
| **Weighted F1** | `0.6931 ± 0.0027` |
| **Balanced Accuracy** | `0.6881 ± 0.0000` |

Document này đóng vai trò xác nhận sự toàn vẹn của Model Causal v2 trong toàn bộ quá trình Review.
