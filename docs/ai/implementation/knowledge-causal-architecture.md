# Phân Tích Chuyên Sâu Kiến Trúc Causal GNN v2 (Architecture Analysis)

Tài liệu này cung cấp cái nhìn khoa học, chi tiết về kiến trúc luồng dữ liệu (Forward Pass) của mô hình **Causal GNN v2** trong dự án Phân loại Thảm họa Đa phương thức. Trọng tâm của kiến trúc là khả năng bóc tách đặc trưng (Disentanglement) và suy luận nhân quả dựa trên đồ thị (Graph-based Causal Reasoning).

## 1. Sơ Đồ Kiến Trúc Luồng Tổng Thể (Overall Architecture)

Kiến trúc giải quyết bài toán đa phương thức thông qua 4 module chính: **Bắt đặc trưng**, **Tách bạch Nhân - Quả**, **Tương tác Đồ thị**, và **Triệt tiêu Nhiễu (Backdoor Adjustment)**.

```mermaid
graph TD
    %% Định nghĩa Style
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef feature fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef causal fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef spurious fill:#ffebee,stroke:#b71c1c,stroke-width:2px;
    classDef graph fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef classifier fill:#fffde7,stroke:#f57f17,stroke-width:2px;

    %% 1. Input & Backbone
    I_Img[Image Input]:::input -->|CLIP ViT| F_Img[Visual Embedding 768d]:::feature
    I_Txt[Text Input]:::input -->|CLIP Text| F_Txt[Text Embedding 768d]:::feature

    %% 2. Modality Fusion
    F_Img -->|Linear Projection| P_Img[Visual Proj 512d]
    F_Txt -->|Linear Projection| P_Txt[Text Proj 512d]
    P_Img -->|+ & LayerNorm| Fused[Fused Representation]:::feature
    P_Txt -->|+ & LayerNorm| Fused

    %% 3. Disentanglement
    Fused -->|MLP Causal| Xc[X_c: Causal Feature]:::causal
    Fused -->|MLP Spurious| Xs[X_s: Spurious Feature]:::spurious

    %% 4. Graph Neural Network (Phase 2)
    Xc -->|k-NN| Adj[Adjacency Matrix A]:::graph
    Adj --> GCN[GCN Layer: AX_cW]:::graph
    Xc -->|Skip Connection| GCN
    GCN --> Xc_graph[X_c_graph: Graph-Enhanced Causal]:::causal

    %% 5. Memory Bank & Inference
    Xs -.->|Store during Train| Bank[(Spurious Memory Bank\nSize: 2000)]:::spurious
    Bank -.->|Sample N=50\nduring Eval| BA[Backdoor Adjustment]:::spurious
    
    Xc_graph -->|Evaluate with Intervention| BA
    BA -->|Average Logits| Cls_Final[Final Robust Prediction]:::classifier
    
    Xc_graph -->|Standard Forward\n(Phase 1 Train)| Cls_Base[Standard Classifier]:::classifier
```

---

## 2. Diễn Giải Toán Học Chi Tiết (Mathematical Formulation)

### Bước 1: Trích Xuất và Dung Hợp Đa Phương Thức (Multimodal Fusion)
Đầu vào là hình ảnh $\mathcal{I}$ và văn bản $\mathcal{T}$. Mô hình sử dụng bộ mã hóa `open_clip` đã được pre-train để quét đặc trưng cơ sở:
$$v = \text{CLIP}_{vision}(\mathcal{I}), \quad t = \text{CLIP}_{text}(\mathcal{T})$$

Tiếp theo, hai Vector được ánh xạ về cùng không gian nhúng $d=512$ và hợp nhất (Late Fusion) nhằm cộng hưởng thông tin:
$$f_{fused} = \text{LayerNorm}(W_v v + W_t t)$$
*Tại đây, $f_{fused}$ chứa toàn bộ thông tin của mẫu dữ liệu, bao gồm cả nguyên nhân cốt lõi gây ra thảm họa và cả những bối cảnh nhiễu (bias) xung quanh.*

### Bước 2: Tách Bạch Đặc Trưng (Causal & Spurious Disentanglement)
Thay vì sử dụng chung $f_{fused}$, mạng tạo hai rẽ nhánh phi tuyến độc lập (MLP Heads) để ép mô hình phải cô lập hóa các ranh giới khái niệm:
$$X_c = \text{ReLU}(W_{c} \cdot f_{fused}) \quad \text{(Nhánh Nhân quả)}$$
$$X_s = \text{ReLU}(W_{s} \cdot f_{fused}) \quad \text{(Nhánh Bối cảnh nhiễu)}$$

**Tính Khoa Học:** Theo biểu diễn đồ thị nhân quả (SCM - Structural Causal Model), $X_c \perp^d X_s$, tức là Nhân quả phải độc lập với Bối cảnh. Việc tối ưu hóa hai nhánh này độc lập trong hàm Loss đa mục tiêu đảm bảo sự phân tách này xảy ra trong không gian High-dimensional.

### Bước 3: Giao Tiếp Đồ Thị (Graph Convolutional Operations)
Trong **Phase 2**, mô hình không coi các dữ liệu trong batch là độc lập (I.I.D) mà xem chúng như các điểm tương quan. Dựa trên khoảng cách Cosine hoặc Euclid của $X_c$, mạng xây dựng ma trận kề $A$ với $k=3$ láng giềng.
Lớp GCN tiêu chuẩn được kích hoạt kết hợp với Residual Connection (Cầu nối dư) để tránh mất mát Feature ban đầu:
$$X_{g} = \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \right) X_c W_{gnn}$$
$$X_{c}^{graph} = X_c + X_g$$
*Tương tác này giúp một Post thảm họa có thông tin nhập nhằng sẽ "vay mượn" độ tự tin từ các Post tương tự rõ ràng hơn trong cùng một batch.*

### Bước 4: Điều Chỉnh Cửa Sau (Backdoor Adjustment)
Để đánh giá tác động thuần túy của Nhân quả ($X_c^{graph}$) lên kết quả dự đoán ($Y$) nhằm tránh các nhiễu từ bối cảnh, phương pháp vận dụng lý thuyết của Judea Pearl:
Khởi tạo Phép can thiệp $\mathbb{P}(Y | do(X_c))$:
$$\mathbb{P}(Y | do(X_c)) = \sum_{x_s \sim \text{Bank}} \mathbb{P}(Y | X_c^{graph}, X_s = x_s) \mathbb{P}(x_s)$$

**Triển khai thực tế qua Monte Carlo Sampling:**
Trong chế độ `eval()`, mô hình bốc ngẫu nhiên $N$ véc-tơ $x_s^{(i)}$ từ **Memory Bank**:
$$\text{Logits}_{ba} = \frac{1}{N} \sum_{i=1}^{N} \text{Classifier}(X_c^{graph} + x_s^{(i)})$$
Xác suất trung bình cuối cùng này hoàn toàn loại bỏ các đường thiên lệch (Backdoor Paths) vì $X_s$ (nhiễu ngẫu nhiên) hoàn toàn không có ràng buộc phân phối chung hệ quả với $X_c^{graph}$. 

---

## 3. Tổng Kết Luồng Logic (Logical Lifecycle)

| Giai đoạn | Vai trò của $X_c$ | Vai trò của $X_s$ | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Phase 1 Training** | Tự dự đoán trực tiếp (Warm-up). | Lưu trữ vào Memory Bank. | Đảm bảo tính toàn vẹn của Feature Extractor trước khi đưa lên đồ thị. |
| **Phase 2 Training** | Lan truyền thông tin qua mạng GCN. | Tiếp tục nạp vào / cập nhật Random Bank. | Cụm tương đồng $X_c$ trở nên đồng nhất, giảm phương sai. |
| **Backdoor Inference** | Ghép cặp véc-tơ với tập $X_s$ lạ. | (Là dữ liệu nhiễu lạ lấy từ Bank). | Triệt tiêu Spurious Correlation, trả về quyết định chuẩn xác và vững chãi nhất. |

Kiến trúc này đánh dấu một bước đột phá trong phương pháp luận kết hợp cấu trúc GNN và Causal Inference, vượt qua các giới hạn của GRL (Gradient Reversal Layer) hay Contrastive Learning đơn thuần.
