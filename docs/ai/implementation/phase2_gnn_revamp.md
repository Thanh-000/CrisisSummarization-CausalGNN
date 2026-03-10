# Phase 2 Causal GNN & Backdoor Adjustment Revamp

## Tổng Quan (Overview)
Trong quá trình huấn luyện mô hình CausalCrisis V2 ở Phase 2 (Causal GNN), chúng tôi gặp phải hiện tượng F1/bAcc suy giảm hoặc không thể vượt qua ranh giới do Baseline MLP (Phase 1) thiết lập (`~0.617 bAcc`). 

Qua một loạt các phân tích (Config `G_ONLY`) nhằm cô lập và kiểm chứng sức mạnh của từng module, chúng tôi phát hiện ra những vấn đề cốt lõi trong (1) Thuật toán Xây dựng Đồ thị k-NN và (2) Cách tích hợp Backdoor Adjustment. Từ đó, chúng tôi tiến hành **Đại tu toàn diện (Revamp)** nhánh Phase 2, đưa bAcc trung bình ở tập Test lên mốc `0.6229` (Max `0.6249`), chính thức xác lập SOTA mới cho dự án.

---

## 1. Phân Tích Lỗi (Root Cause Analysis)

### Thuật toán Graph k-NN cũ kém hiệu quả
1. **"Mù Cục Bộ" do Batch Size nhỏ:** Do giới hạn phần cứng ban đầu, `batch_size` được đặt ở `32`. Trong một batch 32 với bài toán 8 Classes, một lớp chỉ xuất hiện trung bình 4 lần. GNN tìm $K=3$ láng giềng thì gần như lấy toàn bộ mẫu trong batch, khiến đồ thị chứa vô vàn liên kết ngẫu nhiên (Noise/Spurious Edges).
2. **Hard-Edge Gây Over-smoothing:** Các giá trị trong ma trận kề cũ hoàn toàn là `1.0` (Hard-Edge). GNN chỉ lấy lân cận và trung bình cộng một cách cào bằng, làm bào mòn và phá hủy những Feature "tinh khiết" học được từ Phase 1 thay vì bổ sung ngữ nghĩa kiến trúc.

### Backdoor Adjustment (BA) Gây Mất Ổn Định
1. **Lớp Linear Rác "Phá Hoại":** Nhánh BA sử dụng một lớp `classifier_ba` (Linear mới toanh) và chỉ bắt đầu huấn luyện từ Epoch 10. Khi Epoch 10 kích hoạt, trọng số rác (Random Init) của lớp này gửi luồng tín hiệu (Gradient) độc hại ngược về Vector $X_c$, phá nát tính tĩnh/đặc trưng đã học.
2. **BYPASS Không Xuyên Qua GNN:** Nhánh BA cũ chỉ lấy trực tiếp $X_c$ và $X_s$ (nguyên bản), và đưa vào bộ phân loại tĩnh. Trong quy trình này, toàn bộ lớp `GraphSAGE` và Vector GNN ($X_{gnn}$) bị "nhảy cóc" hoàn toàn. Điều này đi ngược lại thiết kế Causal GNN (nơi mà $X_{gnn}$ mới là cái cần được Backdoor Adjustment).

---

## 2. Giải Pháp Triệt Để (The Revamp Architecture)

### 2.1. Nâng Cấp GNN: Soft-Attention GAT + Batch-Vast
- **Nâng `batch_size` lên `256`:** Việc tăng kích thước Batch (khả thi vì VRAM giờ chỉ xử lý Feature tĩnh 512D) cung cấp cho mô hình một vĩ mô 256 mẫu để tính Cosine Similarity. $K=5$ lúc này chứa đựng những liên kết mạnh mẽ, đáng tin cậy. 
- **Soft-Attention (Cosine Softmax):** Chuyển đổi Hard-Edge sang kiến trúc tương đồng **Graph Attention Networks (GAT)**. Giá trị $Cosine\_Sim$ được phân bố qua `Softmax(sim / temperature)` với `T=0.1`. Ghi nhận $90\%$ mức độ "chú ý" (Attention) cho láng giềng sát nhất, chỉ san sẻ $10\%$ cho những láng giềng khác biệt.

### 2.2. Hợp Nhất Hệ Sinh Thái GNN & Backdoor Adjustment
- **Loại bỏ `classifier_ba`:** Không còn sự hiện diện của bất kỳ lớp Linear nào mới.
- **Dung hòa Vector Space (Element-wise Addition):** Tại thời điểm Inference BA (và cả quá trình Training), $X_{gnn}$ và $X_s$ được cộng trực tiếp với nhau ($X_{gnn} + X_s$). Phép cộng vector này nhúng hoàn hảo ngữ cảnh Spurious vào trong Causal Graph Feature mà không làm tăng số chiều.
- **Sử dụng Main `Classifier`:** Kết quả của phép cộng được đưa trực tiếp vào `classifier` chính (đã được warmup và mang tri thức nền tảng). Qua đó, toàn bộ kiến thức được bảo tồn.

### 2.3. Tối ưu hóa Regularization & Scheduler
Tăng cường mức độ "bạo lực" (Aggressive) của quá trình chống Overfitting:
- `Dropout`: Tăng từ `0.3` lên `0.5` cho toàn bộ nhánh phân loại sau cùng.
- `Weight Decay` riêng biệt: Cho `GNN layer` chịu $1e-3$ (gấp đôi MLP) để ép mô hình tiết chế ma trận trọng số. 
- Mở GNN & BA đồng bộ tính ngay từ `Epoch 1`. Trọng số GNN Loss (`alpha_gnn`) Ramp chậm từ $0$ lên $0.3$ tại `Epoch 15`, sau đó giữ nguyên. `Patience Limit` đạt mốc `15` chống kết thúc sớm.

---

## 3. Kết Quả Thực Nghiệm (Final Results)

Cấu hình mới nhất được test qua $3$ Seeds Ngẫu nhiên (Multi-seed: `42`, `100`, `2026`):

```text
============================================================
  FINAL MULTI-SEED RESULTS
============================================================
    Seed   42 | F1: 0.6042 | bAcc: 0.6223
    Seed  100 | F1: 0.6112 | bAcc: 0.6249
    Seed 2026 | F1: 0.6163 | bAcc: 0.6214
  ----------------------------------------------------------
    AVERAGE   | F1: 0.6106 | bAcc: 0.6229
============================================================
```

**Kết Luận:** Sự kết hợp hoàn chỉnh của **Soft-Attention GNN** và **Vector-Addition Backdoor Adjustment** ổn định ở mức `~0.623` cho chuẩn `bAcc`, tự tin vượt qua ranh giới `0.617` do Baseline MLP Phase 1 đặt ra, đánh dấu thành công chói lọi của phương pháp Causal Machine Learning trên dữ liệu Disaster Twitter.
