# 3 Trục Chính Của Đề Tài CausalGNN Crisis Summarization

Dựa trên những phân tích thực nghiệm và các rào cản từ kiến trúc cũ, đề tài được chốt lại xoay quanh 3 trục nghiên cứu cốt lõi. Đây sẽ là xương sống cho phần Introduction, Methodology và Experiments của bài báo:

## 1. Trục 1: Hướng Causal OOD cho CrisisMMD (Protocol LODO)
- **Đặt bài toán:** Generalization (Tổng quát hóa) ngoài miền dữ liệu (Out-of-Distribution) cho bài toán Multimodal Crisis. Các mô hình hiện tại (GEDA, Baseline Fusion) học vẹt bối cảnh ảo (spurious correlations, ví dụ: bão Harvey = lụt), dẫn đến thất bại khi áp dụng vào một thảm họa hoàn toàn mới.
- **Phương pháp đánh giá:** Chuyển trọng tâm từ IID protocol sang **Leave-One-Disaster-Out (LODO)** trên CrisisMMD.
- **Mục tiêu:** Chứng minh sức mạnh cốt lõi của *Causal Disentanglement* & *Intervention*: Khi test trên một thảm họa chưa từng xuất hiện, mô hình nhân quả có độ sụt giảm F1 thấp hơn đáng kể và duy trì sự ổn định so với các baseline thông thường. Sự tương đồng với CAMO/MMDG nhưng với kiến trúc tinh gọn và tập trung hơn.

## 2. Trục 2: Hướng Domain Alignment "Mềm" (Conditional MMD Thay GRL)
- **Vấn đề:** Gradient Reversal Layer (GRL) là một rào cản lớn vì nó cố gắng cạo sạch mọi tín hiệu liên quan đến Event/Domain một cách thô bạo (Aggressive Matching). Điều này vô tình làm tổn thương tín hiệu phân loại (Discriminative Signal) của các task chính.
- **Phương pháp giải quyết:** Thay thế (hoặc bổ sung) GRL bằng **Conditional MMD (Soft Alignment)** tương tự DCAN/DSAN. Áp dụng MMD dựa trên điều kiện Lớp Nhãn (ví dụ: chỉ ép các ảnh "Thiệt hại nặng" của Bão Harvey và Bão Irma lại gần nhau, thay vì ép toàn bộ các ảnh).
- **Mục tiêu:** Giảm sự khác biệt (distribution gap) giữa các Domain nhưng không triệt tiêu tín hiệu phân biệt informative. Giữ lại được "những gì cần giữ" để Causal Features thực sự chỉ chứa thông tin liên quan đến lớp nhãn.

## 3. Trục 3: Hướng Đồ Thị & Tính Dị Cuồng (Graph & Heterophily / No-Graph)
- **Câu hỏi nghiên cứu:** Vị thế thật sự của GNN trong phân loại thảm họa đa phương thức là gì? *"Liệu GNN với kNN graph từ CLIP có thực sự giúp ích, hay kiến trúc Self-Attention/Transformer trên Batch là đủ (thậm chí tốt hơn) khi đối mặt với Graph Heterophily (đồ thị chứa nhiều node kề khác nhãn)?"*
- **Thiết kế thực nghiệm (Ablations):** So sánh trực tiếp 3 biến thể kiến trúc trên cùng một Causal backbone:
    1. **GNN với kNN thuần (Cosine Graph):** Baseline nguyên thủy chịu ảnh hưởng của Heterophily.
    2. **GNN với Label-Guided kNN:** Định hướng lại các cạnh (edges) trong lúc huấn luyện để triệt tiêu Heterophily (bảo đảm node kề cùng nhãn).
    3. **Pure MLP / Transformer (No-Graph):** Loại bỏ hoàn toàn quy trình truyền tin Message Passing của GNN, chỉ dùng Multimodal Fusion và Attention nội bộ.
- **Mục tiêu:** Kết luận minh bạch và thực chứng vai trò của GNN (có Graph vs không Graph) trên cả 2 mặt trận IID và OOD.
