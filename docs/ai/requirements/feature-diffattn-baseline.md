---
title: Differential Attention Baseline Comparison
status: draft
---

# Requirements: Differential Attention Baseline

## 1. Problem Statement
Dự án cốt lõi sử dụng Graph Neural Network (GNN) kết hợp CLIP để phân loại khủng hoảng (Crisis Summarization). Cần có một baseline SOTA (State of the Art) uy tín để so sánh độ hiệu quả. Notebook "Differential Attention for Multimodal Crisis Event Analysis" (CVPRw MMFM 2025) cung cấp một hướng tiếp cận xuất sắc để so sánh trực tiếp, đặc biệt trên Task 3 (Damage Severity). Cần chuẩn bị notebook và script chạy trên Google Colab làm BaseLine.

## 2. Goals
- Cấu hình nguyên vẹn Colab Notebook của CVPRw MMFM 2025 (DiffAttn) để làm baseline chuẩn xác trên tập `CrisisMMD v2.0`.
- Thiết lập quy trình thu thập metrics (Accuracy, F1-Macro, F1-Weighted) để ánh xạ 1-1 với metrics xuất ra từ mô hình GNN hiện hữu (IEEE Access 2025).
- Tích hợp khả năng so sánh tự động trong cùng một biểu đồ (nếu khả thi) hoặc xuất bảng csv chung với GNN.
- Giữ vững môi trường Google Colab để tận dụng GPU.

## 3. Non-Goals
- Không merge trực tiếp code của DiffAttn vào trong luồng xử lý GNN nhằm tránh xung đột code.
- Không sửa đổi lõi kiến trúc tập trung vào Attention của họ trừ đoạn log/metrics để trích xuất số liệu.

## 4. User Stories
- Là một nhà nghiên cứu, tôi muốn chạy Differential Attention framework độc lập trên Google Colab để lấy metrics gốc của tác giả làm benchmark so sánh độ chính xác của bài báo của mình.
- Tôi muốn các file kết quả training (logs, metrics) được thiết kế đồng nhất để dễ dàng import vào file evaluation chung.

## 5. Success Criteria
- Colab Notebook Differential Attention chạy thành công 100% từ đầu đến cuối trên tập dataset có sẵn của người dùng ở Google Drive.
- Metrics báo cáo của DiffAttn được biểu diễn chung một format với kết quả báo cáo của kiến trúc GNN cũ.

## 6. Open Questions
- Người dùng đã cấu hình GNN notebook sinh ra kết quả Evaluation với format nào? (VD: file csv, wandb, bảng markdown?)
