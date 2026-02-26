---
title: Differential Attention Baseline Comparison
status: draft
---

# Planning: Differential Attention Baseline

## 1. Task Breakdown

| Task ID | Task Description | Dependencies | Assignee | Estimate |
|---------|-----------------|--------------|----------|----------|
| `T1-Colab` | Cập nhật cấu hình của `new_colab_notebook.ipynb` (đường dẫn, môi trường) | None | AI | 1h |
| `T2-Logs` | Sửa Notebook phần "Extract Results from Logs" để xuất thông số ra CSV thay vì print dict/string | `T1-Colab` | AI | 1h |
| `T3-Compare` | Tạo thêm 1 Cell vào Notebook ở đoạn cuối để đọc file CSV của GNN Baseline (nếu có) và đối chiếu với CSV của DiffAttn bằng biểu đồ Cột trực quan | `T2-Logs` | AI | 1.5h |
| `T4-Test` | Chạy thử trên Google Colab | `T3-Compare` | User | 2h |

## 2. Implementation Order
Dựa vào Dependency, thực hiện ngay lập tức `T1-Colab` → `T2-Logs` → `T3-Compare`, cập nhật trực tiếp vào file notebook có sẵn `new_colab_notebook.ipynb`. User sẽ thực hiện `T4-Test` sau khi file notebook đã update 100%.

## 3. Risks
- **Mount Google Drive không đồng nhất**: Có thể bị sai mount folder nếu User đổi cấu trúc GNN. Giải pháp: Thêm dòng lệnh verify mount point.
- **Package Conflicts**: `transformers==4.24.0` đã khá cũ, có nguy cơ xung đột trên Colab mới. Giải pháp: Có lệnh pip auto-resolve.
