# CausalCrisis: Khung Suy Luận Nhân Quả Đa Phương Thức cho Phân Loại Thảm Họa Tổng Quát Hóa Liên Miền — Đề Xuất Nghiên Cứu Chi Tiết Cấp Q1/Q2

## Tổng Quan

Đề xuất này trình bày một dự án nghiên cứu mới với tên gọi **CausalCrisis** — khung suy luận nhân quả đa phương thức (Causal Multimodal Reasoning Framework) cho bài toán phân loại thảm họa trên mạng xã hội, với khả năng tổng quát hóa sang các loại thảm họa chưa từng thấy. Khác với đề xuất trước chỉ kết hợp kỹ thuật, nghiên cứu này xây dựng trên **một bài toán nghiên cứu gốc chưa được giải quyết**, có nền tảng lý thuyết nhân quả vững chắc, và nhắm mục tiêu tạp chí Q1/Q2 như *Information Fusion* (IF 15.5), *Expert Systems with Applications* (IF 7.63), hoặc hội nghị A* (ACL, CVPR).[^1][^2]

## Phần I: Phân Tích Cảnh Quan Nghiên Cứu Hiện Tại

### Bản đồ nghiên cứu và khoảng trống quyết định

Phân tích toàn bộ các công trình liên quan cho thấy một **blind spot** rõ ràng mà chưa ai giải quyết:

| Công trình | Modality | Causal Theory | DG | Few-Shot | VLM | Graph | Venue |
|---|---|---|---|---|---|---|---|
| Munia et al. (2025)[^3] | Multimodal | ✗ | ✗ | ✗ | ✓ (CLIP+LLaVA) | ✗ | arXiv/Workshop |
| Nascimento et al. (2025)[^4] | Multimodal | ✗ | ✗ | ✓ | ✓ (CLIP) | ✓ (k-NN) | IEEE Access |
| CAMO (Ma et al., 2025)[^5] | Multimodal | ✓ (Adversarial) | ✓ | ✗ | ✗ (VGG+BERT) | ✗ | arXiv |
| Seeberger et al. (2025)[^6] | **Text-only** | ✓ (Counterfactual) | ✓ | ✗ | ✗ | ✗ | AACL-IJCNLP |
| CIRL (Lv et al., 2022)[^7] | **Image-only** | ✓ (SCM) | ✓ | ✗ | ✗ | ✗ | CVPR Oral |
| InPer (Tang et al., 2024)[^8] | **Image-only** | ✓ (Intervention) | ✓ | ✗ | ✗ | ✗ | BMVC |
| CrisisSpot (Dar et al., 2025)[^9] | Multimodal | ✗ | ✗ | ✗ | ✗ | ✓ (Social) | ESWA (Q1) |
| **CausalCrisis (Đề xuất)** | **Multimodal** | **✓ (SCM + Intervention)** | **✓** | **✓** | **✓ (CLIP+LLaVA)** | **✓ (Causal Graph)** | **Target Q1/Q2** |

**Khoảng trống quyết định**: Chưa có công trình nào kết hợp đồng thời: (1) suy luận nhân quả có lý thuyết (SCM + do-calculus) cho (2) dữ liệu đa phương thức (text + image) với (3) VLM-enhanced features và (4) propagation qua đồ thị trong (5) bối cảnh ít nhãn (few-shot) cho (6) domain generalization across disaster types.

### Tại sao CAMO chưa đủ mạnh cho Q1/Q2

CAMO (Ma et al., 2025) là công trình gần nhất với ý tưởng này, nhưng có **4 hạn chế cốt lõi** có thể khai thác:[^10]

1. **Features lạc hậu**: Sử dụng VGG-16 + BERT — không tận dụng được CLIP/LLaVA pre-trained representations vốn đã chứng minh vượt trội 5-8% accuracy[^3]
2. **Causal graph quá đơn giản**: SCM chỉ có 4 biến \(\{D_l, X_s, X_c, y\}\), không mô hình hóa **cross-modal causal interactions** (ảnh gây bias khác text)
3. **Không tận dụng cấu trúc đồ thị**: Adversarial disentanglement áp dụng ở mức sample, không propagate causal features qua graph — bỏ lỡ unlabeled data
4. **Evaluation hạn chế**: Chỉ leave-one-domain-out, không few-shot scenario — không phản ánh thực tế khi thảm họa mới xảy ra[^10]

### Tại sao Seeberger et al. chưa đủ mạnh

Seeberger et al. (AACL-IJCNLP 2025 Findings) đạt +4.42% F1 trên CrisisLex nhưng:[^11]

1. **Chỉ xử lý text**: Hoàn toàn bỏ qua visual modality — nhưng trong thực tế ảnh thảm họa chứa thông tin rất quan trọng (mức độ hư hại, lũ lụt...)
2. **Bias identification thủ công**: Dựa vào NER để tìm bias tokens — không tổng quát hóa
3. **Không có feature disentanglement**: Chỉ model direct effect qua bias encoder, không tách causal vs. spurious representations
4. **Cải thiện khiêm tốn**: +1.9% F1 — reviewer Q1 sẽ yêu cầu improvement lớn hơn

## Phần II: Câu Hỏi Nghiên Cứu Gốc và Đóng Góp Lý Thuyết

### Research Questions chính

**RQ1 (Lý thuyết):** *Liệu có tồn tại điều kiện identifiability cho phép phân tách causal features bất biến miền khỏi spurious domain-specific features trong không gian biểu diễn VLM đa phương thức?*

Câu hỏi này lấy cảm hứng từ ICLR 2025 Poster về Causal Representation Learning from Multimodal Biomedical Observations, đã chứng minh rằng **structural sparsity of causal connections between modalities** cho phép identifiability lên từng latent component. Nghiên cứu đề xuất mở rộng kết quả này cho crisis multimodal data.[^12][^13]

**RQ2 (Phương pháp):** *Cơ chế causal intervention trên multimodal representations có thể cải thiện bao nhiêu so với adversarial disentanglement đơn thuần (CAMO) trong bài toán domain generalization cho crisis classification?*

**RQ3 (Ứng dụng):** *Trong điều kiện few-shot (50-250 nhãn), việc propagate causal-invariant features qua graph neural network có giúp mô hình tổng quát hóa hiệu quả sang loại thảm họa chưa từng thấy không?*

### Structural Causal Model cho Crisis Multimodal Data

Đây là đóng góp lý thuyết cốt lõi — xây dựng SCM chi tiết hơn CAMO với mô hình hóa tường minh cross-modal causal interactions.[^5]

**Định nghĩa SCM (Crisis Multimodal):**

Cho bài đăng mạng xã hội \(X = (x_v, x_t)\) gồm ảnh \(x_v\) và text \(x_t\), domain (loại thảm họa) \(D\), và nhãn \(y\). Mô hình nhân quả cấu trúc được định nghĩa qua:

\[
\mathcal{M} = \{D, S_v, S_t, C_v, C_t, C_{vt}, x_v, x_t, y\}
\]

[^3]

với các biến:
- \(C_v\): **Visual causal factors** — đặc trưng nhân quả từ ảnh (mức độ hư hại cấu trúc, nước lũ, đám cháy)
- \(C_t\): **Textual causal factors** — đặc trưng nhân quả từ text (mô tả thiệt hại, yêu cầu cứu trợ)
- \(C_{vt}\): **Cross-modal causal factors** — thông tin nhân quả chỉ xuất hiện khi kết hợp cả hai modalities (ví dụ: ảnh lũ + text kêu cứu → "affected individuals")
- \(S_v\): **Visual spurious factors** — nhiễu từ ảnh phụ thuộc domain (phong cảnh địa phương, phong cách ảnh platform-specific)
- \(S_t\): **Textual spurious factors** — nhiễu từ text phụ thuộc domain (hashtag sự kiện, địa danh, số liệu cụ thể)

**Quan hệ nhân quả:**

\[
D \to S_v, \quad D \to S_t, \quad D \not\to C_v, \quad D \not\to C_t, \quad D \not\to C_{vt}
\]

\[
C_v, C_t, C_{vt} \to y, \quad S_v, S_t \not\to y
\]

\[
x_v = g_v(C_v, S_v), \quad x_t = g_t(C_t, S_t), \quad C_{vt} = h(C_v, C_t)
\]

[^4]

**Điểm khác biệt với CAMO**: CAMO chỉ có \(\{D_l, X_s, X_c, y\}\) — không phân biệt spurious/causal giữa 2 modalities, và **hoàn toàn bỏ qua** cross-modal causal factor \(C_{vt}\). Bài báo Seeberger et al. chỉ model event bias \(E \to P \to Y\) cho text, hoàn toàn bỏ qua visual.[^11][^10]

### Điều kiện Identifiability (Đóng góp lý thuyết)

Dựa trên kết quả về structural sparsity từ ICLR 2025, đề xuất **Định lý Identifiability cho Crisis Multimodal SCM**:[^12]

**Định lý (Informal):** *Cho VLM encoder \(f\) (ví dụ CLIP) đã pre-trained trên dữ liệu đa miền. Nếu (i) các causal connections giữa modalities thỏa mãn structural sparsity (số causal factors liên miền \(|C_{vt}| \ll |C_v| + |C_t|\)), và (ii) spurious factors thỏa mãn domain-conditional independence (\(S_v \perp S_t \mid D\)), thì các causal factors \(C_v, C_t, C_{vt}\) có thể identified lên subspace từ VLM representations.*

Đây là mở rộng của Theorem 4.3 trong Sun et al. (ICLR 2025) cho setting crisis-specific, và tạo nền tảng lý thuyết cho toàn bộ phương pháp. **Đây chính là "sức nặng" mà reviewer Q1/Q2 cần — không phải engineering trick mà là mathematical guarantee.**[^13]

## Phần III: Kiến Trúc CausalCrisis Chi Tiết

### Tổng quan pipeline

```
Input: (x_v, x_t, D) — ảnh, text, domain (disaster type)
                │
    ┌───────────┴───────────┐
    ▼                       ▼
[Stage 1] VLM Feature     [Stage 1] VLM Feature  
Extraction (Visual)        Extraction (Text + LLaVA Caption)
  CLIP ViT-L/14 → f_v       CLIP Text → f_t, f_t'
    │                       │
    ▼                       ▼
[Stage 2] Modality-Specific Causal Disentanglement
  ├── Visual: f_v → (C_v, S_v) via adversarial + orthogonal
  └── Text:   f_t → (C_t, S_t) via adversarial + orthogonal
                │
                ▼
[Stage 3] Cross-Modal Causal Intervention
  ├── Compute C_vt via Guided Cross-Attention on (C_v, C_t)
  ├── do(C_vt) — causal intervention via backdoor adjustment
  └── Differential Attention for noise suppression
                │
                ▼
[Stage 4] Causal Graph Propagation (for few-shot)
  ├── k-NN graph on causal features (C_v, C_t, C_vt)
  ├── GraphSAGE propagation (causal only)
  └── Semi-supervised learning with unlabeled data
                │
                ▼
[Stage 5] Domain-Invariant Classification
  ├── Multi-task head (Informative, Humanitarian, Damage)
  └── Focal Loss + Causal Regularization
```

### Stage 1: VLM-Enhanced Feature Extraction

Kế thừa trực tiếp từ bài báo 1 (Munia et al.):[^3]
- **CLIP ViT-L/14** (frozen) cho image embeddings \(f_v \in \mathbb{R}^{768}\)
- **CLIP Text Encoder** (frozen) cho text embeddings \(f_t \in \mathbb{R}^{768}\)
- **LLaVA-v1.6** sinh caption bổ sung \(x'_t\) — mô tả ảnh dưới góc nhìn crisis, tăng cường text alignment[^3]
- Cross-Feature Fusion Module (CFM) từ CapFuse-Net kết hợp \(x_t\) và \(x'_t\)[^14]

Lý do sử dụng CLIP frozen: Bài báo 1 đã chứng minh CLIP frozen outperforms DenseNet+Electra fine-tuned 3-5% accuracy. CLIP embeddings nằm trong shared semantic space, là tiền đề cho causal disentanglement.[^3]

### Stage 2: Modality-Specific Causal Disentanglement

**Khác biệt cốt lõi so với CAMO**: Disentangle riêng biệt cho mỗi modality thay vì gộp chung.

Cho mỗi modality \(m \in \{v, t\}\), feature \(f_m\) được phân tách qua MLP disentangler \(\psi_m\):

\[
C_m, S_m = \psi_m(f_m), \quad \text{trong đó } C_m = \psi_m^c(f_m), \; S_m = \psi_m^s(f_m)
\]

[^15]

**Ràng buộc orthogonal** (đảm bảo tách biệt):

\[
\mathcal{L}_{\perp}^m = \frac{|C_m \cdot S_m|}{||C_m|| \cdot ||S_m||}
\]

[^16]

**Adversarial domain classifier** (enforce domain invariance cho \(C_m\)):

\[
\mathcal{L}_{adv}^m = \text{CE}(\phi_m(C_m), D) \quad \text{(minimize bởi discriminator, maximize bởi encoder qua GRL)}
\]

[^17]

Đây mở rộng CAMO bằng cách: (a) disentangle ở mức từng modality, cho phép kiểm soát fine-grained hơn; (b) sử dụng VLM features thay vì VGG+BERT, tận dụng pre-trained cross-modal alignment.[^10]

### Stage 3: Cross-Modal Causal Intervention

**Đây là đóng góp phương pháp chính — chưa ai thực hiện trong crisis domain.**

Sau khi có \(C_v\) và \(C_t\), tính cross-modal causal factor \(C_{vt}\) qua **Guided Cross-Attention** (từ bài báo 1):[^3]

\[
C_v' = \text{Self-Attn}(C_v), \quad C_t' = \text{Self-Attn}(C_t)
\]

\[
z_v = F(W_v^\top C_v'), \quad \alpha_v = \sigma(W_v'^\top C_v')
\]

\[
z_t = F(W_t^\top C_t'), \quad \alpha_t = \sigma(W_t'^\top C_t')
\]

\[
C_{vt} = \text{concat}(\alpha_t \odot z_v, \; \alpha_v \odot z_t)
\]

[^18]

**Causal Intervention qua Backdoor Adjustment:**

Thay vì sử dụng \(C_{vt}\) trực tiếp, áp dụng **do-calculus** để loại bỏ confounding effect từ domain:

\[
P(y \mid do(C_{vt})) = \sum_{d \in \mathcal{D}} P(y \mid C_{vt}, D=d) \cdot P(D=d)
\]

[^19]

Trong thực hành, intervention được approximated bằng cách: với mỗi sample, trộn (mix) \(C_{vt}\) với causal features trung bình từ các domain khác, sử dụng stratified sampling theo domain distribution. Đây tương tự cách CIRL (CVPR 2022) và InPer (BMVC 2024) thực hiện intervention, nhưng mở rộng cho multimodal setting.[^8][^7]

**Differential Attention** trên causal representation cuối cùng:[^3]

\[
z = \text{DiffAttn}(C_{vt}) = \left(\text{softmax}\left(\frac{Q_1 K_1^\top}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\left(\frac{Q_2 K_2^\top}{\sqrt{d}}\right)\right) V
\]

[^20]

Vai trò của Diff Attn ở đây có **ý nghĩa nhân quả rõ ràng**: softmax thứ nhất capture causal signal, softmax thứ hai capture common-mode noise — phép trừ chính là form of **causal intervention loại bỏ confounding noise**. Đây là insight lý thuyết mới — liên kết Differential Attention với causal inference, chưa ai phát biểu.[^21][^22]

### Stage 4: Causal Graph Propagation

Kế thừa từ bài báo 2 (Nascimento et al.) nhưng với **thay đổi quan trọng**: xây dựng đồ thị chỉ trên causal features.[^4]

1. **PCA reduction** trên \(C_v, C_t, C_{vt}\) (xuống 256-d mỗi loại).  
*Lưu ý Data Leakage*: Quá trình PCA `fit` và `L2-normalization` chỉ áp dụng **trên tập Huấn Luyện (hoặc Labeled Test set)**, sau đó ánh xạ biểu diễn lên Test (bảo toàn unit-sphere cosine similarity).[^4]
2. **k-NN graph construction** ($k$ trượt động dựa theo quy mô Labeled) trên causal features đã reduce. Xây dựng đồ thị **Sparse COO Matrix** thay vì toán hạng Dense để ngăn sập VRAM với không gian hàng chục nghìn điểm ảnh.
3. **GraphSAGE** late fusion (2 layers/modality, 1024-d) — áp dụng **Semi-supervised Transductive Setting**. Propagate causal message qua tập Unlabeled nodes, tạo ưu thế cho mẩu dữ liệu nhỏ giọt. Cấu trúc un-labeled là ẩn danh và không chịu gánh nặng gradients của Loss CE.[^4]

**Tại sao graph trên causal features tốt hơn**: Khi graph được xây dựng trên raw features, k-NN neighbors có thể bao gồm samples tương tự về spurious factors (cùng disaster type) nhưng khác semantic (ví dụ: cùng là ảnh bão nhưng một informative, một non-informative). Graph trên causal features đảm bảo neighbors chia sẻ **causal similarity** — cải thiện label propagation.

### Stage 5: Training và Inference

**Tổng loss function:**

\[
\mathcal{L} = \underbrace{\mathcal{L}_{cls}}_{\text{Focal Loss}} + \alpha_1 \underbrace{\sum_m \mathcal{L}_{adv}^m}_{\text{Adversarial}} + \alpha_2 \underbrace{\sum_m \mathcal{L}_{\perp}^m}_{\text{Orthogonal}} + \alpha_3 \underbrace{\mathcal{L}_{supcon}}_{\text{Contrastive}} + \alpha_4 \underbrace{\mathcal{L}_{int}}_{\text{Intervention}}
\]

[^23]

trong đó:
- \(\mathcal{L}_{cls}\): Focal loss cho multi-task classification (xử lý class imbalance)[^3]
- \(\mathcal{L}_{adv}^m\): Adversarial loss enforcement domain invariance per modality[^10]
- \(\mathcal{L}_{\perp}^m\): Orthogonal regularization đảm bảo \(C_m \perp S_m\)[^10]
- \(\mathcal{L}_{supcon}\): Supervised contrastive loss căn chỉnh causal features cross-modal[^10]
- \(\mathcal{L}_{int}\): Intervention consistency loss — đảm bảo prediction không đổi khi intervention trên domain

**Training strategy**: Alternating optimization giữa discriminator (freeze encoder) và encoder (freeze discriminator + GRL). Quá trình huấn luyện thực thi cơ chế **Phase-aware Adaptive Loss Weighting** để phân tầng sự hội tụ. Sử dụng Capped Focal Loss ($0.5 \le W \le 5.0$) kết hợp Parameter Segmentation (Freeze Bias & Norm decay) trên Optimizer AdamW giúp cực đại hóa sức mạnh cho chế độ Few-shot (N=50).[^4]

**Inference**: Chỉ sử dụng causal branch (\(C_v, C_t, C_{vt}\)) → classification. Spurious branch bị loại bỏ hoàn toàn.  
*Chống Leakage Tối Đại*: Force `domain_labels=None` qua hàm `eval()`, đảm bảo Causal Intervention hoàn toàn mô phỏng Zero-shot và không dựa dẫm thông tin Miền Mục Tiêu Ground-truth.[^4]

## Phần IV: Thiết Kế Thí Nghiệm Nghiêm Ngặt

### Datasets mở rộng

| Dataset | Samples | Events | Modalities | Tasks | Vai trò |
|---|---|---|---|---|---|
| CrisisMMD[^24] | ~18K img, 16K tweets | 7 disasters (2017) | Text + Image | Informative, Humanitarian, Damage | Primary benchmark |
| HumAid[^11] | ~76K tweets | 19 events | Text | 9-class humanitarian | Cross-domain text eval |
| CrisisLex[^11] | ~17K tweets | 26 events | Text | 7-class information type | Cross-domain text eval |
| TSEqD[^9] | 10K+ samples | Turkey-Syria EQ (2023) | Text + Image | Informative, Humanitarian | Multimodal cross-domain |
| DMD[^10] | Damage dataset | Multiple disasters | Text + Image | Binary damage | CAMO comparison |

### Kịch bản đánh giá (Evaluation Protocols)

**Protocol 1 — In-Domain (so sánh với Munia et al., CrisisSpot):**
CrisisMMD standard split. Đánh giá 3 tasks. Metrics: Accuracy, Macro F1, Weighted F1. Mục tiêu: vượt 92.91% accuracy Task 1.[^3]

**Protocol 2 — Leave-One-Disaster-Out (LODO) Domain Generalization (so sánh với CAMO):**
Cắt cứng phân vùng dữ liệu: Huấn luyện trên 6/7 disasters trong CrisisMMD, test trên disaster còn lại. Lặp lại 7 lần quay vòng với Module `run_lodo_all_experiments()`. So sánh trực tiếp với CAMO (4-21% improvement over baselines). Kịch bản này không chia Train/Test ngẫu nhiên mà kiểm soát hoàn toàn Out-Of-Distribution Disjoint.[^10]

**Protocol 3 — Cross-Dataset Domain Generalization:**
Huấn luyện trên CrisisMMD, test trên TSEqD (và ngược lại). Đây là kịch bản **khó nhất và mới nhất** — thay đổi cả domain, thời gian, và ngôn ngữ.

**Protocol 4 — Few-Shot Domain Generalization (50, 100, 250 nhãn từ target):**
Huấn luyện trên source domain (CrisisMMD) + few-shot target labels. So sánh với Nascimento et al. (GNN few-shot) và Sirbu et al. (FixMatch).[^4]

**Protocol 5 — Comprehensive Ablation:**

| Variant | Mô tả | Trả lời câu hỏi |
|---|---|---|
| CausalCrisis w/o intervention | Bỏ Stage 3 intervention | Causal intervention có cần thiết? |
| CausalCrisis w/o modal-specific | Disentangle gộp (như CAMO) | Tách biệt per-modality có tốt hơn? |
| CausalCrisis w/o graph | Bỏ Stage 4, dùng MLP trực tiếp | Graph propagation có giúp few-shot? |
| CausalCrisis w/o DiffAttn | Thay Diff Attn bằng standard attention | Diff Attn có vai trò causal? |
| CausalCrisis w/o VLM caption | Bỏ LLaVA caption | VLM enrichment có cần thiết cho causal? |
| CausalCrisis (raw graph) | Graph trên raw features (như bài 2) | Graph trên causal features có tốt hơn? |
| CAMO + CLIP | Thay VGG+BERT bằng CLIP trong CAMO | Fair comparison khi cùng features |

### Phân tích định tính (Qualitative Analysis)

- **t-SNE visualization**: So sánh causal features vs. raw features — domain gap có giảm không?[^10]
- **Grad-CAM**: Mô hình attend vào vùng nhân quả (hư hại, nạn nhân) hay spurious (logo, background)?[^14]
- **Causal probing** (theo Seeberger et al.): Dùng shallow classifier đánh giá xem causal branch encode domain info hay task info[^11]
- **Statistical Significance (Paired $t$-test)**: Report P-values (< 0.05) bằng Script đính kèm cho từng Phase Baseline vs. Proposed.
- **Error analysis**: Phân tích cross-modal dependency errors (giống bài 2, Table 5)[^4]

## Phần V: Lập Luận Sức Nặng Q1/Q2

### 5 tiêu chí reviewer Q1/Q2 và cách đáp ứng

| Tiêu chí Reviewer | Cách đề xuất trước thất bại | Cách CausalCrisis đáp ứng |
|---|---|---|
| **Novelty** | "Combination of existing techniques" | SCM mới cho crisis multimodal + cross-modal causal factor \(C_{vt}\) + identifiability theorem |
| **Theoretical depth** | Không có lý thuyết | SCM formal, identifiability conditions, link DiffAttn → causal intervention |
| **Significance** | Incremental improvement | Giải quyết open problem: causal DG cho multimodal crisis — chưa ai làm |
| **Experimental rigor** | Chỉ 1 dataset, 1 protocol | 5 datasets, 5 protocols, 7 ablation variants, qualitative analysis |
| **Broader impact** | Chỉ academic | Ứng dụng trực tiếp cho humanitarian response — khi thảm họa mới xảy ra |

### Tại sao CausalCrisis > CAMO + Seeberger tách rời

CausalCrisis không phải phép cộng đơn giản mà tạo ra **synergy effects**:

1. **VLM features + causal disentanglement**: CLIP đã được train trên 400M image-text pairs — representations của nó đã có alignment cross-modal, tạo tiền đề tốt cho identifiability. CAMO dùng VGG+BERT không có alignment này.[^25][^10]

2. **Per-modality disentanglement + cross-modal causal factor**: Bằng cách tách \(C_v, S_v\) và \(C_t, S_t\) riêng biệt TRƯỚC KHI cross-modal fusion, đảm bảo spurious information từ mỗi modality không "lây nhiễm" sang modality kia. CAMO disentangle SAU fusion — spurious đã trộn lẫn.

3. **Graph trên causal features + few-shot**: Nascimento et al. xây graph trên raw features. Nhưng raw features chứa domain-specific patterns → neighbors trong graph có thể misleading. Graph trên causal features đảm bảo **semantic neighbors**, cải thiện label propagation quality.[^4]

4. **Differential Attention as implicit intervention**: Insight mới — DiffAttn loại bỏ common-mode noise tương đương backdoor adjustment trên attention space. Đây là **theoretical link** chưa ai phát biểu.[^22]

### Tạp chí/Hội nghị mục tiêu và chiến lược

| Target | IF/Rank | Lý do phù hợp | Timeline |
|---|---|---|---|
| **Information Fusion** (Elsevier)[^2] | IF 15.5, Q1 | Multimodal fusion + causal reasoning — core topic | Primary target |
| **Expert Systems with Applications** (Elsevier)[^1] | IF 7.63, Q1 | Crisis management trong scope; causal DG papers accepted | Backup target |
| **IEEE Trans. Multimedia**[^26] | IF ~5.5, Q1 | Multimodal learning + application | Alternative |
| **ACL/EMNLP** (NLP venue, A*) | Top conference | Nếu focus text-heavy version | Conference option |
| **CVPR/ICCV** (Vision venue, A*) | Top conference | Nếu focus visual causal | Conference option |

## Phần VI: Kế Hoạch Thực Hiện 12 Tháng

### Giai đoạn chi tiết

| GĐ | Tháng | Nội dung | Milestone | Rủi ro & Giải pháp |
|---|---|---|---|---|
| **1** | 1-2 | Literature review sâu, reproduce CAMO + Nascimento baseline, chuẩn bị data | Baseline results trên CrisisMMD | Code CAMO không public → implement từ paper |
| **2** | 2-4 | Stage 1+2: VLM extraction + per-modality disentanglement | Disentanglement module hoạt động; t-SNE visualization cho thấy separation | Adversarial training không hội tụ → tuning GRL λ |
| **3** | 4-6 | Stage 3: Cross-modal causal intervention + DiffAttn | In-domain results ≥ SOTA; ablation cho Stage 3 | Intervention approximation quality → test nhiều sampling strategies |
| **4** | 6-8 | Stage 4: Graph construction trên causal features + GraphSAGE | Few-shot results; so sánh causal graph vs. raw graph | Memory bottleneck → ANN approximation[^4] |
| **5** | 8-10 | Full pipeline + DG evaluation (5 protocols) | Complete results trên 5 datasets | Negative results ở cross-dataset → focus trên leave-one-out |
| **6** | 10-12 | Theoretical analysis, writing, submission | Paper hoàn thiện + code release | Identifiability proof phức tạp → state as conjecture with empirical evidence |

### Yêu cầu tài nguyên

- **GPU**: NVIDIA A100 40GB × 2 (CLIP inference + GraphSAGE training + adversarial optimization)
- **Storage**: ~100GB (datasets + embeddings + checkpoints)
- **Framework**: PyTorch 2.x + PyTorch Geometric + Hugging Face Transformers + OpenAI CLIP
- **LLaVA**: Chạy inference 1 lần để sinh captions, lưu cache → không cần GPU liên tục cho LLaVA

## Phần VII: Kết Luận — Tại Sao Đủ Sức Nặng Q1/Q2

CausalCrisis tạo sức nặng qua 3 tầng đóng góp:

**Tầng 1 — Lý thuyết (Theory):** SCM mới cho crisis multimodal data với cross-modal causal factor \(C_{vt}\) + identifiability conditions. Đây là đóng góp **không thể tìm thấy** trong bất kỳ công trình crisis nào hiện tại. Liên kết lý thuyết Differential Attention ↔ causal intervention cũng là insight hoàn toàn mới.[^27][^22]

**Tầng 2 — Phương pháp (Method):** Pipeline end-to-end kết hợp per-modality causal disentanglement → cross-modal intervention → causal graph propagation. Mỗi stage đều có justification lý thuyết từ SCM, không phải ghép nối kỹ thuật tùy ý. So sánh fair với CAMO, Seeberger, Nascimento, Munia qua 5 protocols.[^6][^5][^3][^4]

**Tầng 3 — Ứng dụng (Application):** Giải quyết bài toán thực tế: khi thảm họa mới xảy ra (ví dụ: sóng thần ở quốc gia mới), mô hình chỉ cần 50-100 nhãn từ sự kiện mới + kiến thức causal-invariant từ thảm họa cũ để phân loại hiệu quả. Đây là capability chưa có phương pháp nào đạt được.

---

## References

1. [Expert Systems with Applications - IF 7.63 - JCR Q1 - OPERATIONS RESEARCH & MANAGEMENT SCIENCE · ivySCI](https://www.ivysci.com/en/journals/0957-4174) - Journal Expert Systems with Applications, ISSN: 0957-4174, 1873-6793. Expert Systems with Applicatio...

2. [Information Fusion - Impact Factor, Quartile, Ranking](https://wos-journal.info/journalid/11626) - 5-year Impact Factor: 17.9, Best ranking: COMPUTER SCIENCE, THEORY & METHODS, Percentage rank: 98%, ...

3. [Differential-attention.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/182790943/58e69a72-8b7b-4b30-bb9f-8ab073d19764/Differential-attention.pdf?AWSAccessKeyId=ASIA2F3EMEYE2W6H4BX3&Signature=70fcmYyO0sHtHBsNrzuVjFQYDo8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMSJIMEYCIQCJ7FY2bqGE%2FobKBwxYh7T13UtXzt7dEPHQz1HIi4A0IQIhAJfvXOcET%2Fi0%2BvcjLDxdoPeSSEIDlAx%2FFLzXoZ10MgCpKvwECOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igw2jTLdijQ%2BbcDqh8wq0AQnkX0j3dnLUPb3lR0GW6uT6CJXVsybx0MNmFim93gJJgtL8IIRVv6NReMyy9uUHPNXXWHLsXZowKLhY41%2BdqVLOnYJ%2Fh3xHjKBh9wie4UP4GayXv79xKFI7q7446q25qYWVXmQ9jupUYgzG4xyj5jIhX%2BtS9KE7yAhTVXH%2BhEcAtGH7qctMZtfEvqSnf3w9y8nlHRGvoZSqzRc2ruQKc7Ysr0UOj%2Foa9EA7mcDWgpxW7Y1ijKlJGVRuiF7kRYYh3usgBqhedCGGUBHwiT0g08Y%2FSa8B8JiqgvB6X5e8dFZ1r5mXWcbORpDPe1i8Ilt3rBk1OCMfFrh0wKmLkMNyW6ex8O8ZsmhFK77jJ4cSkVx05Itf0jISe4SDjvPKJTGUD0y99aNUKzqd2DCPNRx3JGMqB08OGpjIycHLN0dEaicKvfx6hEUJ%2FXYcE2OLVUvmiwSTYzQKqGWDOUFQktPCsud97Fh04ptov999pcJ2283VbsaQ6dGZ0nMuDEVC9IQtrAbkpEhA2PyU6GT3rpFuW9iHSmTv6Alji%2BbLeK7WbTmSAj3U%2FVmRS87FoR75vHrxJYFl9rQxsxpk8xcoUhradmFRteCjwI2hSIeJjlb9vQNL4LSjclFrsNh%2BTb2iwiry6BXIVy2ryfdGxa1JL4Nl44xGd%2FIQDYGnVwFPjDbF%2FRdXQsEj1ORGhNRcVsguXanapk01FFHF8QNpo35ftVDP0ba8%2FaNOgzyoVY3cNaY237i%2BRP8TY0iFUGEPdJYUjAXeGq0zpddHrDAdPjTXc1UAP%2BVMIzzqc0GOpcBliBaZ%2BJhQ39AO8twhA%2Bi9jwD%2FGaAi49wFp%2Bkm6AOmRgF2Q0RDXlU9%2BGGbyAXjIk0%2F%2ByqAjo2YA9ry%2BFMlrjJQclAwbrTA8XQQownhKhiBKEL5KGiUeDc6wi8nhs6FG7WI%2Bk0pV4y2OVFTuzGRFSQa81FlfrtZtlcl74GTSv9uB7P815FDcWZS6JoL08Rzi%2FqTbZQz0qSiQ%3D%3D&Expires=1772785238) - Differential Attention for Multimodal Crisis Event Analysis

4. [Multimodal_Classification_of_Social_Media_Disaster_Posts_With_Graph_Neural_Networks_and_Few-Shot.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/182790943/d6c9ca98-ba57-4e22-98f8-31e0e78a4a39/Multimodal_Classification_of_Social_Media_Disaster_Posts_With_Graph_Neural_Networks_and_Few-Shot_Learning.pdf?AWSAccessKeyId=ASIA2F3EMEYE2W6H4BX3&Signature=mFqw6kqqigb33VXeVsFbqTYiw74%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMSJIMEYCIQCJ7FY2bqGE%2FobKBwxYh7T13UtXzt7dEPHQz1HIi4A0IQIhAJfvXOcET%2Fi0%2BvcjLDxdoPeSSEIDlAx%2FFLzXoZ10MgCpKvwECOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1Igw2jTLdijQ%2BbcDqh8wq0AQnkX0j3dnLUPb3lR0GW6uT6CJXVsybx0MNmFim93gJJgtL8IIRVv6NReMyy9uUHPNXXWHLsXZowKLhY41%2BdqVLOnYJ%2Fh3xHjKBh9wie4UP4GayXv79xKFI7q7446q25qYWVXmQ9jupUYgzG4xyj5jIhX%2BtS9KE7yAhTVXH%2BhEcAtGH7qctMZtfEvqSnf3w9y8nlHRGvoZSqzRc2ruQKc7Ysr0UOj%2Foa9EA7mcDWgpxW7Y1ijKlJGVRuiF7kRYYh3usgBqhedCGGUBHwiT0g08Y%2FSa8B8JiqgvB6X5e8dFZ1r5mXWcbORpDPe1i8Ilt3rBk1OCMfFrh0wKmLkMNyW6ex8O8ZsmhFK77jJ4cSkVx05Itf0jISe4SDjvPKJTGUD0y99aNUKzqd2DCPNRx3JGMqB08OGpjIycHLN0dEaicKvfx6hEUJ%2FXYcE2OLVUvmiwSTYzQKqGWDOUFQktPCsud97Fh04ptov999pcJ2283VbsaQ6dGZ0nMuDEVC9IQtrAbkpEhA2PyU6GT3rpFuW9iHSmTv6Alji%2BbLeK7WbTmSAj3U%2FVmRS87FoR75vHrxJYFl9rQxsxpk8xcoUhradmFRteCjwI2hSIeJjlb9vQNL4LSjclFrsNh%2BTb2iwiry6BXIVy2ryfdGxa1JL4Nl44xGd%2FIQDYGnVwFPjDbF%2FRdXQsEj1ORGhNRcVsguXanapk01FFHF8QNpo35ftVDP0ba8%2FaNOgzyoVY3cNaY237i%2BRP8TY0iFUGEPdJYUjAXeGq0zpddHrDAdPjTXc1UAP%2BVMIzzqc0GOpcBliBaZ%2BJhQ39AO8twhA%2Bi9jwD%2FGaAi49wFp%2Bkm6AOmRgF2Q0RDXlU9%2BGGbyAXjIk0%2F%2ByqAjo2YA9ry%2BFMlrjJQclAwbrTA8XQQownhKhiBKEL5KGiUeDc6wi8nhs6FG7WI%2Bk0pV4y2OVFTuzGRFSQa81FlfrtZtlcl74GTSv9uB7P815FDcWZS6JoL08Rzi%2FqTbZQz0qSiQ%3D%3D&Expires=1772785238) - Multimodal Classification of Social Media Disaster Posts With Graph Neural Networks and Few-Shot Lea...

5. [Causality-Guided Adversarial Multimodal Domain Generalization for ...](https://arxiv.org/abs/2512.08071) - Crisis classification in social media aims to extract actionable disaster-related information from m...

6. [Generalizing to Unseen Disaster Events: A Causal View](https://aclanthology.org/2025.findings-ijcnlp.2/) - Our approach outperforms multiple baselines by up to +1.9% F1 and significantly improves a PLM-based...

7. [Causality Inspired Representation Learning for Domain Generalization](https://arxiv.org/abs/2203.14237) - Extensive experimental results on several widely used datasets verify the effectiveness of our appro...

8. [Domain Generalization with Causal Intervention and Perturbation](https://arxiv.org/abs/2408.03608) - In this paper, we propose a novel and holistic framework based on causality, named InPer, designed t...

9. [A social context-aware graph-based multimodal attentive ...](https://arxiv.org/abs/2410.08814) - In times of crisis, the prompt and precise classification of disaster-related information shared on ...

10. [[論文評述] CAMO: Causality-Guided Adversarial Multimodal Domain ...](https://www.themoonlight.io/tw/review/camo-causality-guided-adversarial-multimodal-domain-generalization-for-crisis-classification) - CAMO is a causality-guided adversarial multimodal domain generalization (MMDG) framework designed fo...

11. [Generalizing to Unseen Disaster Events: A Causal View - arXiv](https://arxiv.org/html/2511.10120v1)

12. [Causal Representation Learning from Multimodal Biomedical ...](https://openreview.net/forum?id=hjROBHstZ3) - In this work, we aim to develop flexible identification conditions for multimodal data and principle...

13. [Causal Representation Learning from Multimodal Biomedical ...](https://iclr.cc/virtual/2025/poster/28747) - Recent advances in causal representation learning have shown promise in identifying interpretable la...

14. [[PDF] Multimodal vision-language models with guided ... - OpenReview](https://openreview.net/pdf/5be644ae35b4115e20081a5bd7476d2889730659.pdf) - Understanding crisis events from social media posts to support response and rescue efforts often req...

15. [Leveraging multimodal deep learning for natural disaster event ...](https://ijai.iaescore.com/index.php/IJAI/article/view/24433) - This paper proposes a hybrid learning model to improve disaster event classification and damage seve...

16. [Unveiling the dynamics of crisis events: Sentiment and emotion analysis via multi-task learning with attention mechanism and subject-based intent prediction](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?params=%2Fcontext%2Fsis_research%2Farticle%2F9699%2F&path_info=DynamicsCrisisEvents_pvoa_cc_by_nc.pdf)

17. [Multimodal Classification of Social Media Disaster Posts with Graph Neural Networks and Few-Shot Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4995887) - Social media has emerged during the last decade as a potential information source in crisis scenario...

18. [Can We Predict the Unpredictable? Leveraging ...](https://arxiv.org/abs/2506.23462) - Effective disaster management requires timely and accurate insights, yet traditional methods struggl...

19. [Differential Attention for Multimodal Crisis Event Analysis](http://arxiv.org/abs/2507.05165) - Social networks can be a valuable source of information during crisis events. In particular, users c...

20. [A Social Context-aware Graph-based Multimodal Attentive Learning Framework for Disaster Content Classification during Emergencies](https://ar5iv.labs.arxiv.org/html/2410.08814) - In times of crisis, the prompt and precise classification of disaster-related information shared on ...

21. [Paper Review: Differential Transformer](https://andlukyane.com/blog/paper-review-diff) - My review of the paper Differential Transformer

22. [[Literature Review] Differential Transformer - Moonlight](https://www.themoonlight.io/en/review/differential-transformer) - Multi-Head Attention: The paper employs a multi-head mechanism where each head computes its differen...

23. [Cross-Attention Multimodal Classification of Disaster-Related Tweets](https://www.academia.edu/123987260/CAMM_Cross_Attention_Multimodal_Classification_of_Disaster_Related_Tweets) - The proposed cross-attention-based multimodal deep learning method outperforms the current state-of-...

24. [CrisisMMD: Multimodal Crisis Dataset - CrisisNLP](https://crisisnlp.qcri.org/crisismmd.html) - The CrisisMMD multimodal Twitter dataset consists of several thousands of manually annotated tweets ...

25. [When and How Does CLIP Enable Domain and Compositional ...](https://icml.cc/virtual/2025/poster/45573) - (2025) demonstrated that CLIP's “generalization performance [...] drops to levels similar to what ha...

26. [IEEE MultiMedia - researchr journal](https://researchr.org/journal/ieeemm/volume/25)

27. [[2410.05258] Differential Transformer - arXiv.org](https://arxiv.org/abs/2410.05258) - Specifically, the differential attention mechanism calculates attention scores as the difference bet...

