# 🔍 Phân Tích Vấn Đề Nghiên Cứu - Dự Án CrisisSpot

**Bài báo:** Khung học tập đa phương thức dựa trên đồ thị và ngữ cảnh xã hội cho phân loại nội dung thảm họa trong tình huống khẩn cấp

**Nguồn:** https://arxiv.org/abs/2410.08814

---

## 📋 Mục Lục

1. [Vấn đề nghiên cứu cốt lõi](#1-vấn-đề-nghiên-cứu-cốt-lõi)
2. [Vấn đề kỹ thuật](#2-vấn-đề-kỹ-thuật)
3. [Thách thức phương pháp luận](#3-thách-thức-phương-pháp-luận)
4. [Vấn đề liên quan đến dữ liệu](#4-vấn-đề-liên-quan-đến-dữ-liệu)
5. [Vấn đề đánh giá và xác thực](#5-vấn-đề-đánh-giá-và-xác-thực)
6. [Câu hỏi nghiên cứu mở](#6-câu-hỏi-nghiên-cứu-mở)
7. [Hướng mở rộng và phát triển](#7-hướng-mở-rộng-và-phát-triển)
8. [So sánh với Pipeline MEASF](#8-so-sánh-với-pipeline-measf)

---

## 1. Vấn Đề Nghiên Cứu Cốt Lõi

### 1.1 Phát Biểu Vấn Đề Nghiên Cứu Chính

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     VẤN ĐỀ NGHIÊN CỨU CỐT LÕI                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  "Làm thế nào để phân loại hiệu quả nội dung liên quan đến thảm họa            │
│   từ dữ liệu mạng xã hội đa phương thức (văn bản + hình ảnh) trong             │
│   các tình huống khẩn cấp?"                                                     │
│                                                                                 │
│  Các vấn đề con:                                                                │
│  ───────────────                                                                │
│  1. Làm sao nắm bắt CẢ sự đồng thuận VÀ mâu thuẫn giữa các phương thức?        │
│  2. Làm sao tận dụng ngữ cảnh xã hội (độ tin cậy, tương tác) hiệu quả?         │
│  3. Làm sao lan truyền kiến thức giữa các nội dung tương tự?                    │
│  4. Làm sao kết hợp tối ưu các đặc trưng hỗn tạp (attention, đồ thị, xã hội)?  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Các Khoảng Trống Nghiên Cứu Được Giải Quyết

| Khoảng trống | Hiện trạng (Trước đây) | Giải pháp CrisisSpot |
|--------------|------------------------|----------------------|
| **Tương tác phương thức** | Chỉ nắm bắt sự tương đồng/đồng thuận | IDEA: HAM (đồng thuận) + CAM (mâu thuẫn) |
| **Lan truyền thông tin** | Xử lý từng tweet độc lập | GraphSAGE: Trí tuệ tập thể từ tweet tương tự |
| **Ngữ cảnh người dùng** | Bỏ qua lịch sử/độ tin cậy người dùng | UIS: Điểm thông tin người dùng dựa trên lịch sử |
| **Từ vựng khủng hoảng** | Phương pháp NLP tổng quát | CIS: Từ điển khủng hoảng đặc thù (4.268 thuật ngữ) |
| **Hiểu cảm xúc** | Chỉ phân tích cảm xúc cơ bản | EmoQuotient: Phân tích 11 chiều cảm xúc |

### 1.3 Các Giả Thuyết Nghiên Cứu

1. **GT1:** Nắm bắt cả quan hệ hài hòa và trái ngược giữa các phương thức cải thiện độ chính xác phân loại
2. **GT2:** Lan truyền kiến thức dựa trên đồ thị giữa các tweet tương tự tăng cường quyết định tập thể
3. **GT3:** Đặc trưng ngữ cảnh xã hội (độ tin cậy, tương tác) cung cấp tín hiệu bổ sung đáng kể
4. **GT4:** Từ điển đặc thù khủng hoảng vượt trội so với từ vựng tổng quát trong hiểu nội dung thảm họa

---

## 2. Vấn Đề Kỹ Thuật

### 2.1 Vấn Đề Cơ Chế Attention IDEA

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  CƠ CHẾ IDEA - CÁC VẤN ĐỀ NGHIÊN CỨU                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Vấn đề 1: Lựa chọn nhiệt độ (Temperature)                                      │
│  ─────────────────────────────────────────                                      │
│  • HAM: T = 1.65 (chọn theo kinh nghiệm)                                        │
│  • CAM: T = 0.75 (chọn theo kinh nghiệm)                                        │
│  • Câu hỏi: Làm sao TỰ ĐỘNG chọn nhiệt độ tối ưu?                              │
│  • Câu hỏi: Các loại khủng hoảng khác nhau có cần giá trị T khác nhau không?   │
│                                                                                 │
│  Vấn đề 2: Tính hợp lệ của phát hiện mâu thuẫn                                  │
│  ─────────────────────────────────────────────                                  │
│  • Giả định: Đảo ngược embedding (-1) nắm bắt sự đối lập ngữ nghĩa             │
│  • Câu hỏi: Phép phủ định đơn giản có đủ cho mọi loại mâu thuẫn không?         │
│  • Câu hỏi: Còn các mâu thuẫn tinh tế như châm biếm, mỉa mai thì sao?          │
│                                                                                 │
│  Vấn đề 3: Chiến lược kết hợp Attention                                         │
│  ───────────────────────────────────                                            │
│  • Hiện tại: Nối các đầu ra HAM và CAM                                          │
│  • Câu hỏi: Kết hợp có trọng số có hiệu quả hơn không?                          │
│  • Câu hỏi: Trọng số kết hợp nên được học hay cố định?                          │
│                                                                                 │
│  Vấn đề 4: Độ phức tạp tính toán                                                │
│  ───────────────────────────────                                                │
│  • Ma trận tương đồng: O(d²) với d = 128                                        │
│  • Hai lần tính attention riêng biệt (HAM + CAM)                                │
│  • Câu hỏi: Có thể tính HAM và CAM hiệu quả hơn không?                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Vấn Đề Học Đồ Thị

```
Vấn đề 1: Đồ thị thưa
──────────────────────
• Ngưỡng = 0.75 để tạo cạnh
• Ngưỡng cao → Đồ thị thưa → Lan truyền hạn chế
• Ngưỡng thấp → Đồ thị dày → Lan truyền nhiễu
• Câu hỏi: Làm sao xác định ngưỡng tối ưu một cách thích ứng?

Vấn đề 2: Khả năng mở rộng xây dựng đồ thị
─────────────────────────────────────────
• Độ tương đồng cặp: O(n²) cho n mẫu
• Với 10.000 tweet: 100 triệu phép so sánh
• Câu hỏi: Làm sao mở rộng lên hàng triệu tweet trong thảm họa thực?

Vấn đề 3: Cập nhật đồ thị động
──────────────────────────────
• Hiện tại: Đồ thị tĩnh trong quá trình suy luận
• Thảm họa thực: Tweet mới liên tục xuất hiện
• Câu hỏi: Làm sao cập nhật đồ thị tăng dần?

Vấn đề 4: Chuyển giao kiến thức giữa các thảm họa
─────────────────────────────────────────────────
• Hiện tại: Đồ thị được xây cho từng tập dữ liệu
• Câu hỏi: Kiến thức từ thảm họa trước có giúp ích cho thảm họa hiện tại không?
```

### 2.3 Vấn Đề Đặc Trưng Ngữ Cảnh Xã Hội

```
Vấn đề 1: Vấn đề khởi động lạnh người dùng
──────────────────────────────────────────
• UIS yêu cầu lịch sử người dùng
• Người dùng mới không có lịch sử → UIS = 0.5 (mặc định)
• Câu hỏi: Làm sao ước tính độ tin cậy cho người dùng mới/không hoạt động?

Vấn đề 2: Tính đầy đủ của từ điển khủng hoảng
─────────────────────────────────────────────
• Hiện tại: 4.268 thuật ngữ khủng hoảng
• Thảm họa mới có thể đưa vào từ vựng mới
• Câu hỏi: Làm sao mở rộng từ điển tự động?

Vấn đề 3: Tương quan cảm xúc-khủng hoảng
────────────────────────────────────────
• EmoLex được thiết kế cho cảm xúc tổng quát
• Tình huống khủng hoảng có thể có mẫu cảm xúc độc đáo
• Câu hỏi: Có cần từ điển cảm xúc đặc thù khủng hoảng không?

Vấn đề 4: Thao túng tương tác người dùng
────────────────────────────────────────
• Chỉ số tương tác có thể bị thao túng (bot, tài khoản giả)
• Câu hỏi: Làm sao phát hiện và lọc tương tác bị thao túng?
```

---

## 3. Thách Thức Phương Pháp Luận

### 3.1 Vấn Đề Căn Chỉnh Đa Phương Thức

```python
# THÁCH THỨC: Khoảng cách ngữ nghĩa giữa các phương thức

# Biểu diễn văn bản: BERT embeddings (768 chiều)
# Biểu diễn hình ảnh: Đặc trưng ResNet (1024 chiều)

# Vấn đề: Không gian ngữ nghĩa khác nhau
# Giải pháp thử nghiệm: Chiếu embedding dùng chung
H_Text = tanh(BatchNorm(H_t @ W_t))  # Chiếu vào không gian chung

# Câu hỏi mở:
# 1. Chiếu dùng chung có bảo toàn thông tin đặc thù phương thức không?
# 2. Những gì bị mất trong quá trình chiếu?
# 3. Làm sao cân bằng biểu diễn chung và biểu diễn đặc thù phương thức?
```

### 3.2 Thách Thức Kết Hợp Đặc Trưng

```
┌────────────────────────────────────────────────────────────────┐
│              PIPELINE KẾT HỢP ĐA GIAI ĐOẠN                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Giai đoạn 1: Kết hợp IDEA                                     │
│  ─────────────────────────                                     │
│  HAM + CAM → FAV (1792 chiều)                                  │
│  Vấn đề: Nút cổ chai thông tin?                                │
│                                                                │
│  Giai đoạn 2: Kết hợp cấp phương thức                          │
│  ────────────────────────────────────                          │
│  MALN(FAV) → 128 chiều                                         │
│  SHLN(SHV) → 8 chiều                                           │
│  GFLN(Đồ thị) → 128 chiều                                      │
│  Vấn đề: Quy mô và trọng số quan trọng khác nhau?              │
│                                                                │
│  Giai đoạn 3: Kết hợp tổng thể                                 │
│  ─────────────────────────────                                 │
│  Nối(128 + 8 + 128) = 264 chiều → 64 chiều                     │
│  Vấn đề: Nối đơn giản có nắm bắt được các tương tác không?     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Thiết Kế Hàm Mất Mát

```python
# Triển khai hiện tại
if task == 'binary':
    loss = BCELoss(predictions, targets)
elif task == 'multi-class':
    loss = CrossEntropyLoss(predictions, targets)

# Vấn đề nghiên cứu:
# 1. Không có loss explicit cho chất lượng IDEA (cân bằng HAM vs CAM)
# 2. Không có contrastive loss cho chất lượng graph embedding
# 3. Không có tác vụ phụ trợ cho học ngữ cảnh xã hội
# 4. Mất cân bằng lớp không được xử lý rõ ràng

# Cải tiến tiềm năng:
loss_total = (
    λ1 * task_loss +           # Phân loại chính
    λ2 * contrastive_loss +    # Chất lượng graph embedding
    λ3 * modality_align_loss + # Căn chỉnh văn bản-hình ảnh
    λ4 * contrary_detect_loss  # Phát hiện mâu thuẫn
)
```

---

## 4. Vấn Đề Liên Quan Đến Dữ Liệu

### 4.1 Hạn Chế Tập Dữ Liệu

| Tập dữ liệu | Số mẫu | Ngôn ngữ | Thảm họa | Hạn chế |
|-------------|--------|----------|----------|---------|
| **CrisisMMD** | ~18.000 | Tiếng Anh | 7 sự kiện | Đa dạng hạn chế |
| **TSEqD** | 10.352 | Đa ngôn ngữ | 1 sự kiện | Chỉ một loại khủng hoảng |

### 4.2 Thách Thức Gán Nhãn

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       VẤN ĐỀ GÁN NHÃN                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Vấn đề 1: Tính chủ quan                                                        │
│  ───────────────────────                                                        │
│  • "Thông tin" là khái niệm chủ quan                                            │
│  • Các người gán nhãn khác nhau có thể không đồng ý                             │
│  • Độ đồng thuận giữa người gán nhãn: Kappa = 0.73-0.85 (trung bình-tốt)       │
│                                                                                 │
│  Vấn đề 2: Ngữ cảnh thời gian                                                   │
│  ───────────────────────────                                                    │
│  • Cùng nội dung có thể là thông tin lúc khủng hoảng bắt đầu, thừa sau đó      │
│  • Nhãn hiện tại là tĩnh, không nhận biết thời gian                             │
│                                                                                 │
│  Vấn đề 3: Phân loại nhân đạo                                                   │
│  ────────────────────────────                                                   │
│  • Hiện tại: {Cơ sở hạ tầng, Cứu hộ, Quyên góp, Thương vong, ...}              │
│  • Có thể không bao phủ mọi loại thảm họa                                       │
│  • Các danh mục chồng chéo trong một số trường hợp                              │
│                                                                                 │
│  Vấn đề 4: Không khớp hình ảnh-văn bản trong dữ liệu gốc                       │
│  ─────────────────────────────────────────────────────                          │
│  • Một số tweet có hình ảnh không liên quan                                     │
│  • Nên gán nhãn những trường hợp này là "không thông tin" hay danh mục đặc biệt?│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Vấn Đề Chất Lượng Dữ Liệu

```python
# Vấn đề 1: Dữ liệu thiếu
# Nhiều tweet có:
# - Không có hình ảnh (chỉ văn bản)
# - Không có metadata người dùng (tài khoản đã xóa)
# - Văn bản bị cắt ngắn (retweet)

# Vấn đề 2: Nhiễu
# Mạng xã hội chứa:
# - Châm biếm và mỉa mai
# - Hình ảnh không đúng ngữ cảnh
# - Spam và quảng cáo
# - Trộn ngôn ngữ (code-switching)

# Vấn đề 3: Mất cân bằng lớp
# Phân bố điển hình:
# - Không thông tin: 60-70%
# - Thông tin: 30-40%
# - Các phân loại nhân đạo: mất cân bằng cao

# Vấn đề 4: Thiên lệch thời gian
# - Tweet đầu: tin tức mới, thông tin không chắc chắn
# - Tweet sau: cập nhật, trùng lặp
# - Mô hình được huấn luyện trên dữ liệu hỗn hợp, có thể không tổng quát hóa
```

---

## 5. Vấn Đề Đánh Giá Và Xác Thực

### 5.1 Hạn Chế Độ Đo Đánh Giá

```
Độ đo hiện tại:
──────────────
• Accuracy, Precision, Recall, F1-score

Khía cạnh còn thiếu:
────────────────────
• Hiệu suất thời gian thực (độ trễ)
• Độ bền vững với đầu vào đối nghịch
• Công bằng giữa các nhóm người dùng khác nhau
• Khả năng giải thích quyết định
• Tổng quát hóa sang loại thảm họa mới
```

### 5.2 Vấn Đề So Sánh Baseline

```
Vấn đề 1: Lựa chọn baseline
───────────────────────────
• Nhiều baseline đã lỗi thời (2018-2020)
• Thiếu so sánh với mô hình đa phương thức mới nhất (2023+)
• Không so sánh với phương pháp dựa trên LLM (GPT, LLaMA)

Vấn đề 2: Công bằng triển khai
────────────────────────────
• Siêu tham số khác nhau cho mô hình khác nhau
• Một số baseline có thể không được tinh chỉnh tối ưu
• Câu hỏi: So sánh có thực sự công bằng không?

Vấn đề 3: Ý nghĩa thống kê
─────────────────────────
• Không báo cáo khoảng tin cậy
• Không có kiểm định thống kê (t-test, Wilcoxon)
• Kết quả một lần chạy có thể có phương sai
```

### 5.3 Kiểm Tra Tổng Quát Hóa

```
Đã kiểm tra:                          CHƯA kiểm tra:
────────────                          ──────────────
✓ CrisisMMD (7 thảm họa)              ✗ Loại thảm họa mới (đại dịch, tấn công mạng)
✓ TSEqD (Động đất Thổ Nhĩ Kỳ-Syria)   ✗ Ngôn ngữ không phải tiếng Anh
✓ Phân loại nhị phân                  ✗ Dữ liệu streaming thời gian thực
✓ Phân loại đa lớp                    ✗ Đa nền tảng (Facebook, Instagram)
                                      ✗ Tổng quát hóa thời gian dài hạn
```

---

## 6. Câu Hỏi Nghiên Cứu Mở

### 6.1 Câu Hỏi Cơ Bản

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  CÂU HỎI NGHIÊN CỨU MỞ                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  C1: TƯƠNG TÁC PHƯƠNG THỨC                                                      │
│  ─────────────────────────                                                      │
│  • Ngoài HAM+CAM, còn quan hệ nào khác giữa các phương thức?                   │
│  • Có thể mô hình hóa quan hệ phân cấp hoặc thời gian không?                    │
│  • Làm sao xử lý tốt khi thiếu phương thức?                                     │
│                                                                                 │
│  C2: LAN TRUYỀN KIẾN THỨC                                                       │
│  ─────────────────────────                                                      │
│  • Cấu trúc đồ thị tối ưu cho thông tin khủng hoảng là gì?                      │
│  • Kiến thức đồ thị đã huấn luyện có chuyển giao được giữa các thảm họa không? │
│  • Làm sao cân bằng ngữ cảnh cục bộ và mẫu toàn cục?                            │
│                                                                                 │
│  C3: MÔ HÌNH HÓA NGƯỜI DÙNG                                                     │
│  ─────────────────────────                                                      │
│  • Làm sao mô hình thay đổi hành vi người dùng trong khủng hoảng?               │
│  • Cấu trúc mạng xã hội (bạn bè, người theo dõi) có giúp ích không?             │
│  • Làm sao xác định nguồn đáng tin cậy mới nổi trong thảm họa mới?              │
│                                                                                 │
│  C4: THÍCH ỨNG THỜI GIAN THỰC                                                   │
│  ────────────────────────────                                                   │
│  • Làm sao thích ứng mô hình với tình huống khủng hoảng đang tiến triển?        │
│  • Học chủ động có giảm nhu cầu gán nhãn không?                                 │
│  • Làm sao xử lý trôi khái niệm theo thời gian?                                 │
│                                                                                 │
│  C5: KHẢ NĂNG GIẢI THÍCH                                                        │
│  ────────────────────────                                                       │
│  • Tại sao mô hình phân loại tweet này là thông tin?                            │
│  • Phương thức nào đóng góp nhiều nhất vào quyết định?                          │
│  • Làm sao cung cấp hiểu biết hành động được cho người ứng phó khẩn cấp?        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Câu Hỏi Ứng Dụng Cụ Thể

```
Cho Tóm Tắt Khủng Hoảng (MEASF):
────────────────────────────────
1. Phân loại CrisisSpot có giúp ưu tiên tweet để tóm tắt không?
2. Làm sao mở rộng IDEA attention cho tóm tắt trích xuất/trừu tượng?
3. Phân cụm dựa trên đồ thị có xác định được cụm thông tin đa dạng không?
4. Làm sao sử dụng CIS/UIS cho độ tin cậy nội dung trong bản tóm tắt?

Cho Phát Hiện Thông Tin Sai:
────────────────────────────
1. CAM (contrary attention) có phát hiện được tin giả không?
2. UIS hiệu quả như thế nào trong xác định nguồn không đáng tin cậy?
3. Lan truyền đồ thị có giúp xác định chiến dịch thông tin sai phối hợp không?

Cho Phân Bổ Tài Nguyên:
───────────────────────
1. Phân loại nhân đạo có hướng dẫn được phân phối tài nguyên không?
2. Làm sao ưu tiên nội dung khẩn cấp so với thông tin?
3. Trích xuất vị trí từ dữ liệu đa phương thức có giúp logistics không?
```

---

## 7. Hướng Mở Rộng Và Phát Triển

### 7.1 Mở Rộng Ngắn Hạn (3-6 tháng)

```
1. NHIỆT ĐỘ CÓ THỂ HỌC
   ────────────────────
   • Thay T_HAM=1.65, T_CAM=0.75 cố định bằng tham số có thể học
   • Sử dụng attention hoặc cổng để điều chỉnh nhiệt độ động
   
2. CONTRARY ATTENTION ĐA ĐẦU
   ──────────────────────────
   • Nhiều đầu CAM cho các loại mâu thuẫn khác nhau
   • Phát hiện mâu thuẫn {Ngữ nghĩa, Thời gian, Cảm xúc}
   
3. MỞ RỘNG TỪ ĐIỂN KHỦNG HOẢNG
   ────────────────────────────
   • Mở rộng tự động sử dụng word embeddings
   • Thích ứng miền cho loại thảm họa mới
   
4. XÂY DỰNG ĐỒ THỊ HIỆU QUẢ
   ─────────────────────────
   • Tìm kiếm láng giềng gần nhất xấp xỉ (FAISS)
   • Băm nhạy vị trí cho khả năng mở rộng
```

### 7.2 Mở Rộng Trung Hạn (6-12 tháng)

```
1. TÍCH HỢP MÔ HÌNH NGÔN NGỮ LỚN
   ───────────────────────────────
   • Sử dụng LLM (GPT, LLaMA) để hiểu văn bản
   • Kết hợp với IDEA cho suy luận đa phương thức
   • Tuân theo hướng dẫn cho tác vụ đặc thù khủng hoảng
   
2. PHƯƠNG THỨC VIDEO
   ──────────────────
   • Mở rộng từ hình ảnh sang video clip
   • Mô hình hóa thời gian cho nội dung động
   • Tích hợp tín hiệu âm thanh
   
3. HỖ TRỢ ĐA NGÔN NGỮ
   ────────────────────
   • BERT đa ngôn ngữ cho văn bản
   • Đặc trưng hình ảnh không phụ thuộc ngôn ngữ
   • Chuyển giao kiến thức giữa các ngôn ngữ
   
4. XỬ LÝ LUỒNG
   ─────────────
   • Pipeline suy luận thời gian thực
   • Cập nhật đồ thị tăng dần
   • Phát hiện và thích ứng trôi khái niệm
```

### 7.3 Hướng Nghiên Cứu Dài Hạn (1-2 năm)

```
1. MÔ HÌNH NỀN TẢNG CHO KHỦNG HOẢNG
   ─────────────────────────────────
   • Tiền huấn luyện trên dữ liệu khủng hoảng lớn
   • Chuyển giao sang bất kỳ loại thảm họa nào
   • Khả năng thích ứng few-shot
   
2. SUY LUẬN ĐA PHƯƠNG THỨC
   ─────────────────────────
   • Vượt xa phân loại: trả lời câu hỏi
   • Suy luận nhân quả về sự kiện thảm họa
   • Dự đoán tiến triển khủng hoảng
   
3. HỢP TÁC NGƯỜI-AI
   ──────────────────
   • Học chủ động với người ứng phó khẩn cấp
   • Khuyến nghị có thể giải thích
   • Hiệu chỉnh độ tin cậy cho quyết định AI
   
4. AI ĐẠO ĐỨC CHO KHỦNG HOẢNG
   ───────────────────────────
   • Công bằng giữa các quần thể bị ảnh hưởng
   • Bảo vệ quyền riêng tư cho nạn nhân
   • Giảm thiểu tác hại thuật toán
```

---

## 8. So Sánh Với Pipeline MEASF

### 8.1 So Sánh Tác Vụ

| Khía cạnh | CrisisSpot | MEASF |
|-----------|------------|-------|
| **Tác vụ** | Phân loại | Tóm tắt |
| **Đầu vào** | Tweet đơn lẻ + hình ảnh | Nhiều tweet + tin tức + hình ảnh |
| **Đầu ra** | Nhãn lớp | Bản tóm tắt văn bản + hình ảnh |
| **Mục tiêu** | Accuracy, F1 | ROUGE, BLEU, CLIP |

### 8.2 Khả Năng Áp Dụng Kỹ Thuật

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│           KỸ THUẬT CRISISSPOT → ỨNG DỤNG MEASF                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  IDEA Attention → Tóm tắt                                                       │
│  ─────────────────────────                                                      │
│  • HAM: Tìm câu ĐỒNG THUẬN với chủ đề chính                                     │
│  • CAM: Xác định MÂU THUẪN để kiểm tra sự thật                                  │
│  • Sử dụng: Chọn câu trích xuất, phát hiện trùng lặp                            │
│                                                                                 │
│  Học Đồ Thị → Phân cụm nội dung                                                 │
│  ─────────────────────────────                                                  │
│  • Xây đồ thị tương đồng câu/tweet                                              │
│  • GraphSAGE: Lan truyền điểm quan trọng                                        │
│  • Sử dụng: Tóm tắt đa dạng bằng lấy mẫu từ cụm                                 │
│                                                                                 │
│  Ngữ cảnh xã hội → Trọng số nguồn                                               │
│  ─────────────────────────────────                                              │
│  • CIS: Tăng cường câu liên quan khủng hoảng                                    │
│  • UIS: Trọng số nội dung theo độ tin cậy nguồn                                 │
│  • Sử dụng: Ưu tiên nội dung đáng tin cậy, thông tin                            │
│                                                                                 │
│  Từ điển khủng hoảng → Tóm tắt nhận biết từ khóa                                │
│  ─────────────────────────────────────────────────                              │
│  • 4.268 thuật ngữ khủng hoảng cho tăng cường TF-IDF                            │
│  • Sử dụng: Đảm bảo từ vựng khủng hoảng trong bản tóm tắt                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Ưu Tiên Triển Khai Cho MEASF

| Ưu tiên | Kỹ thuật | Công sức | Tác động dự kiến |
|---------|----------|----------|------------------|
| 🔴 Cao | Từ điển khủng hoảng (CIS) | Thấp | +5-10% độ liên quan |
| 🔴 Cao | CLIP theo nhiệt độ | Thấp | +3-5% chọn hình ảnh |
| 🟡 Trung bình | Độ tin cậy người dùng (UIS) | Trung bình | +2-4% chất lượng |
| 🟡 Trung bình | Xếp hạng nhận biết cảm xúc | Trung bình | +3-5% mạch lạc |
| 🟢 Thấp | Phân cụm dựa trên đồ thị | Cao | +5-8% đa dạng |
| 🟢 Thấp | Contrary Attention | Cao | Phát hiện thông tin sai |

---

## 📚 Tổng Kết

### Các Vấn Đề Nghiên Cứu Chính Đã Xác Định

1. **Tương tác phương thức:** Làm sao nắm bắt cả sự đồng thuận và mâu thuẫn giữa văn bản và hình ảnh
2. **Lan truyền kiến thức:** Làm sao tận dụng trí tuệ tập thể từ nội dung tương tự
3. **Ngữ cảnh người dùng:** Làm sao tích hợp độ tin cậy nguồn và tín hiệu tương tác
4. **Hiểu khủng hoảng:** Làm sao sử dụng từ vựng và cảm xúc đặc thù miền

### Thách Thức Kỹ Thuật Chính

1. **Lựa chọn nhiệt độ:** Hiện tại theo kinh nghiệm, cần tối ưu tự động
2. **Khả năng mở rộng:** Xây dựng đồ thị và tính attention cho tập dữ liệu lớn
3. **Tổng quát hóa:** Hiệu suất trên loại thảm họa và ngôn ngữ mới
4. **Xử lý thời gian thực:** Xử lý dữ liệu streaming và cập nhật mô hình

### Cơ Hội Nghiên Cứu Tương Lai

1. Tích hợp với Mô hình Ngôn ngữ Lớn
2. Hỗ trợ phương thức video và âm thanh
3. Khả năng đa ngôn ngữ và đa nền tảng
4. Khung hợp tác người-AI

---

*Tài liệu được tạo: 09/02/2026*
*Mục đích: Phân tích vấn đề nghiên cứu cho dự án CrisisSpot*
