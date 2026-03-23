# CausalCrisis Roadmap: Trục Nghiên Cứu & Trục Triển Khai Hệ Thống

**Ngày cập nhật:** 2026-03-11

Với nền tảng Causal GNN Phase 2 hiện tại, hệ thống sẽ phát triển theo 2 trục song song:
(1) Mở rộng năng lực mô hình (Research), và (2) Nâng cấp hệ thống triển khai thực tế (Engineering).

---

## 1. Trục Nghiên Cứu: Nâng cấp GNN & Graph (Pushing OOD Boundaries)

### 1.1. Phase 3c – Heterophily / Label-Guided Graph
**Mục tiêu:** Giảm thiểu lỗi từ các cạnh "sai nhãn" (Spurious/Heterophilic edges) trong domain mới, đẩy mạnh năng lực Out-Of-Distribution (OOD).

**Hướng triển khai (HA-GAT, GATv3):**
- Thêm một `edge weight branch` học "mức độ heterophily" cho từng cạnh: nếu 2 node có embedding và (pseudo-)label khác xa, attention của cạnh đó bị giảm mạnh thay vì được trung bình trơn tru như hiện tại.
- **Tích hợp Nhãn (Label/Pseudo-label Guide):**
  - *Khi Training (IID/LODO):* Dùng ground-truth label để regularize attention (tăng mạnh homophilic edges, triệt tiêu heterophilic edges).
  - *Khi Inference (OOD):* Dùng Soft-Logits hiện tại làm pseudo-label để reweight graph động.
- **Concrete Tasks:**
  1. Thêm module `EdgeHeterophilyScorer(x_i, x_j, y_i, y_j)` trả về dạng vô hướng (scalar) cho mỗi cạnh.
  2. Nhân hệ số này vào Attention Score trước bước `Softmax` của GraphSAGE/GAT.
  3. Chạy Ablation Study: *GNN hiện tại* vs *GNN + Heterophily*, đo đạc trên cả IID và LODO.

### 1.2. Phase 3d – Mở rộng bài toán (Future Paper Value)
Các bước mở rộng nhẹ nhưng mang lại điểm nhấn học thuật rất cao:
- **Multi-label / Multi-task extension:** Mở rộng CrisisMMD theo Setting D của (Abavisani CVPR 2020), giải quyết đồng thời đa nhãn trên cùng một event.
- **Cross-event Few-Shot:** Giữ cấu trúc LODO, nhưng cho phép "thấy" 5-10 labeled samples của Disaster mới (ví dụ: bão Harvey), sau đó fine-tune cực nhanh (Fast-Adapt) cho GNN/BA head.

---

## 2. Trục Hệ Thống: Biến Mô Hình Thành Sản Phẩm Thực Chiến

### 2.1. Inference Pipeline Hoàn Chỉnh (REST API)
**Mục tiêu:** Cho phép đưa (Image JPEG + Text) vào, trả thẳng ra JSON prediction (3 task + confidence).

**Pipeline kiến trúc:**
- *Preprocess I/O:* Resize/Normalize ảnh, Tokenize Text -> Bắn qua CLIP Image & Text Encoders.
- *Phase 1:* Lấy $X_c$ (Causal) và $X_s$ (Spurious) từ Backbone Disentangler.
- *Phase 2:* Xây mini-batch graph động (e.g. 32 request gần nhất làm láng giềng) -> chạy k-NN + Soft-Attention GAT + Backdoor Adjustment ($X_{gnn} + X_s$).
- *Heads & Output:* Trả kết quả của 3 sub-task dưới định dạng chuẩn JSON để tích hợp (ví dụ Ushahidi, dashboard bộ chỉ huy nội bộ).
- *Triển khai:* Đóng gói thành Python package + REST API (FastAPI / Flask). Quản trị bằng `config.yaml` cho phép Switch linh hoạt giữa (Phase 1 vs Phase 2), (IID vs LODO checkpoint).

### 2.2. Monitoring & Evaluation Service
Để duy trì độ bền vững của hệ thống thực chiến:
- **Logging JSON:** Lưu toàn bộ meta request (event, time), output logits, và latency.
- **Online Evaluation:** Tạo script tự động hóa lưới đánh giá; khi user cập nhật ground-truth mới -> Tự động vẽ lại Confusion Matrix, F1, bAcc theo disaster.
- **Simple Drift Detection:** Đo lường sự sai khác phân phối Logits (KL divergence). Nếu tăng vượt ngưỡng -> Phát tín hiệu cảnh báo cần Fine-tune (Retrain).

### 2.3. Active Learning Loop (Người-Máy Cộng Tác)
- Hệ thống tự chủ chọn lọc các bản tin (tweet) có **Entropy cao** hoặc nằm sát vùng biên quyết định (Decision Boundary) để đẩy ra giao diện chờ con người dán nhãn (Annotators).
- Khi gom đủ $N$ mẫu mới của một Disaster lạ -> Tự động trích xuất chu trình chạy **Mini Fine-tune OOD** để mô hình thích nghi ngay trong thời gian thực.
