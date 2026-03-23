# Phương Pháp Luận Huấn Luyện Causal GNN (Training Methodology)

Tài liệu này trình bày chi tiết về phương pháp luận và cơ chế huấn luyện của mô hình Causal GNN v2 (được triển khai trong `causal_crisis_v2_training.ipynb`), phục vụ bài toán Phân loại Thảm họa Đa phương thức dựa trên Suy luận Nhân quả.

## 1. Mở Đầu (Overview)
Trong phân loại thảm họa mạng xã hội, các mô hình học sâu truyền thống thường bị "đánh lừa" bởi các chi tiết nhiễu (Spurious Correlations/Biases) – ví dụ: một hình ảnh có chứa "xe cứu hỏa" đôi khi luôn bị gán nhãn là thảm họa mà không cần hiểu bối cảnh thực. 

Phương pháp luận **Causal GNN v2** giải quyết sự thiên lệch này thông qua kiến trúc **Disentangled Representation** (Tách bạch đặc trưng) và kỹ thuật **Backdoor Adjustment** (Điều chỉnh Cửa sau) dựa vào đồ thị (Graph Networks). Việc huấn luyện được thực hiện theo chiến lược **Curriculum Learning (2 Giai đoạn Khởi động & Mở rộng)** nhằm đảm bảo sự ổn định của đồ thị batch-wise k-NN.

---

## 2. Chiến Lược Huấn Luyện 2 Giai Đoạn (Two-Phase Strategy)

Bởi vì GNN tính toán sự tương tác giữa các sample (thông qua k-Nearest Neighbors), nếu khởi tạo GNN ngay từ Epoch đầu tiên khi trọng số còn ngẫu nhiên, đồ thị tạo ra sẽ hoàn toàn nhiễu và làm sai lệch quá trình hội tụ. Do đó, quá trình train đi qua 2 Phase:

### Phase 1: Warm-up (Khởi động Giai đoạn 1)
- **Chu kỳ**: Epoch 1 $\rightarrow$ Epoch 15.
- **Mục tiêu**: Huấn luyện ổn định các bộ trích xuất đặc trưng gốc (CLIP Text/Vision Encoders & Fusion Layers) để tạo ra các Node Features chất lượng.
- **Cơ chế**: Mô hình bỏ qua module GNN. Ma trận kề đồ thị (`adj`) truyền vào Model bằng `None`. Dữ liệu chỉ chảy qua mạng Multi-layer Perceptron (MLP) thông thường.
- **Output phase này**: Các đỉnh (nodes) đã có các véc-tơ đại diện ngữ nghĩa tương đối chính xác.

### Phase 2: Graph-enhanced Learning (Giai đoạn Tăng cường Cấu trúc)
- **Chu kỳ**: Epoch 16 $\rightarrow$ Epoch 40.
- **Mục tiêu**: Thu thập tính chất "Nhân Quả" của các sample nhờ phân tích cụm (Graph Aggregation) và giảm thiểu các nhiễu.
- **Cơ chế hình thành đồ thị**: Tại mỗi mini-batch, mô hình nối $k=3$ láng giềng gần nhất giữa các sample để tạo ma trận kề Batch-Graph. Tầng **GCN (Graph Convolutional Network)** lúc này kích hoạt bằng phép nhân ma trận chuẩn ($\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$), tổng hợp thông tin từ láng giềng. Nó cộng dồn dạng Residual Connection ($X_c + Graph(X_c)$) để biến véc-tơ thô $X_c$ thành $X_c^{graph}$.
- **Spurious Memory Bank**: Cũng tại Phase 2, các đặc trưng nhiễu ($X_s$) sinh ra của từng sample được liên tục **lưu trữ** vào một "Ngân hàng Bộ nhớ" (`MemoryBank`, size=2000). Điều này phục vụ trực tiếp cho thuật toán triệt tiêu nhiễu ở khâu Validate/Test.

---

## 3. Kiến Trúc Tách Bạch Đặc Trưng (Disentanglement)

Để tính toán phương pháp luận nhân quả, trước bước Final Classifier, biểu diễn hợp nhất (Fused Representation) bắt buộc phải được "chẻ" làm đôi:

1. **Nhánh Causal ($X_c$)**: Đây là nhánh "Nhân quả", chịu trách nhiệm chính trong việc đưa ra dự đoán. Tại Phase 2, $X_c$ đi xuyên qua mạng GNN để làm mượt đặc trưng (Graph-enhanced).
2. **Nhánh Spurious ($X_s$)**: Đây là nhánh "Nhiễu". Nhánh này cố ý học các bối cảnh (Context/Domain bias) và bị tách biệt khỏi suy luận cốt lõi. 
3. Cả hay nhánh này nhận Gradient thông qua **Hàm Loss Đa mục tiêu**: $\mathcal{L}_{Total} = \mathcal{L}_{Causal} + \alpha \cdot \mathcal{L}_{Spurious}$. Giai đoạn này buộc mô hình hiểu sâu 2 khái niệm đối lập trong cùng một sample đa phương tiện.

---

## 4. Trái Tim Phương Pháp: Causal Intervention (Backdoor Adjustment)

Bản chất thuật toán để mô hình "miễn nhiễm" với các Spurious Biases nằm ở khâu Evaluation (Validation/Testing) ở Phase 2 thông qua khái niệm Can Thiệp (Intervention) của Judea Pearl: $\mathbb{P}(Y | do(X))$.

Thay vì dự đoán nhãn $Y$ từ bộ đôi đặc trưng nội tại của sample: $Y = \text{Classifier}(X_c^{graph}, X_s)$, phương pháp luận tiến hành:
1. Giữ nguyên $X_c^{graph}$ của ảnh hiện tại.
2. Xóa bỏ $X_s$ của ảnh hiện tại, thay thế bằng $N=50$ véc-tơ $X_s$ ngẫu nhiên lấy từ **Memory Bank** (tập hợp các nhiễu không liên quan lấy từ lịch sử Dataset).
3. Đưa $N=50$ cặp $(X_c^{graph}, \text{Random } X_s)$ qua bộ phân loại để lấy trung bình cộng xác suất (Monte Carlo Approximation).

**Ý nghĩa nhân quả:** Bằng cách "ép ghép" $X_c$ bản thể với hàng loạt các bối cảnh ngẫu nhiên xa lạ ($X_s$), biến giao hội (Confounders) bị đánh gãy. Các biases không còn khả năng lèo lái phán đoán, và nhãn $Y$ đầu ra trở nên tinh khiết, trung thực dựa trên yếu tố Causal.

---

## 5. Các Tham Số Huấn Luyện Cốt Lõi (Hyperparameters)

*   **Chỉ số GNN (k-NN)**: Khởi tạo với 3 láng giềng gần nhất cho các ma trận đồ thị (Batch-wise Adj).
*   **Optimizer & LR**: Sử dụng AdamW. `GNN` và `Classifier` được học chênh lệch với backbone `CLIP` (Tỷ lệ học `1e-4` và `2e-5`). Các layers đã *Pretrained* học chậm hơn để không làm hỏng Feature Map tổng quát.
*   **Số lượng can thiệp (N Samples)**: $N=50$ cho phương pháp lấy mẫu Monte Carlo trong Backdoor Adjustment.
*   **Metric đo lường tối ưu (Best Model Logging)**: Báo cáo bằng chuẩn **Weighted F1** và **Balanced Accuracy**, cho phép xử lý hiệu quả tình huống mất cân bằng dữ liệu lớn của dataset thảm họa sinh thái Twitter.
