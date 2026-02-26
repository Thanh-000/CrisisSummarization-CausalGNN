# Hướng Dẫn Biên Dịch Báo Cáo LaTeX

## Cấu Trúc Thư Mục

```
latex_report/
├── main.tex          # File chính của báo cáo
├── appendix.tex      # Phụ lục với hình ảnh và bảng bổ sung
└── README.md         # File hướng dẫn này
```

## Yêu Cầu

### Phần Mềm
- **LaTeX Distribution:** TeX Live, MiKTeX, hoặc MacTeX
- **Editor (tuỳ chọn):** TeXstudio, Overleaf, VS Code + LaTeX Workshop

### Packages Cần Thiết
Các packages sau sẽ tự động được cài đặt nếu sử dụng MiKTeX:
- `babel` (vietnamese)
- `tikz`, `pgfplots`
- `booktabs`, `multirow`
- `hyperref`
- `algorithm`, `algorithmic`

## Biên Dịch

### Cách 1: Command Line

```bash
# Di chuyển vào thư mục
cd docs/latex_report

# Biên dịch main.tex (chạy 2 lần để tạo mục lục)
pdflatex main.tex
pdflatex main.tex

# Biên dịch appendix.tex
pdflatex appendix.tex
```

### Cách 2: Overleaf (Online)

1. Truy cập [overleaf.com](https://overleaf.com)
2. Tạo dự án mới → Upload files
3. Upload `main.tex` và `appendix.tex`
4. Nhấn "Recompile"

### Cách 3: TeXstudio

1. Mở `main.tex` trong TeXstudio
2. Nhấn F5 hoặc Build & View
3. Lặp lại cho `appendix.tex`

## Nội Dung Báo Cáo

### main.tex (Báo cáo chính)
1. **Giới thiệu** - Bối cảnh, vấn đề, đóng góp
2. **Kiến trúc CrisisSpot** - Tổng quan các module
3. **Cơ chế IDEA Attention** - HAM, CAM, công thức toán học
4. **Học đồ thị đa phương thức** - GraphSAGE
5. **Đặc trưng ngữ cảnh xã hội** - 21 chiều
6. **Kết quả thực nghiệm** - Bảng, biểu đồ
7. **Vấn đề nghiên cứu mở** - Thách thức, hướng phát triển
8. **Kết luận**
9. **Tài liệu tham khảo**

### appendix.tex (Phụ lục)
- Sơ đồ chi tiết IDEA Attention
- Biểu đồ so sánh phương pháp
- Bảng siêu tham số đầy đủ
- Chi tiết 21 chiều đặc trưng
- Kết quả trên TSEqD
- Biểu đồ radar
- Sơ đồ mạng MFN
- Timeline phát triển

## Hình Ảnh và Biểu Đồ

Tất cả hình ảnh được tạo bằng TikZ/PGFPlots trong LaTeX:

| Hình | Mô tả |
|------|-------|
| Fig 1 | 4 đóng góp chính của CrisisSpot |
| Fig 2 | Kiến trúc tổng quan |
| Fig 3 | So sánh F1-Score |
| Fig 4 (Appendix) | Sơ đồ IDEA chi tiết |
| Fig 5 (Appendix) | Accuracy comparison |
| Fig 6 (Appendix) | Ablation study chart |
| Fig 7 (Appendix) | Radar comparison |
| Fig 8 (Appendix) | MFN architecture |

## Bảng Biểu

| Bảng | Nội dung |
|------|----------|
| Tab 1 | Các module chính |
| Tab 2 | Ảnh hưởng nhiệt độ |
| Tab 3 | 21 chiều features |
| Tab 4 | Thông tin dataset |
| Tab 5 | Kết quả CrisisMMD |
| Tab 6 | Ablation study |
| Tab 7 | Inference time |
| Tab 8 (Appendix) | Hyperparameters |
| Tab 9 (Appendix) | Social features detail |
| Tab 10 (Appendix) | TSEqD results |

## Tuỳ Chỉnh

### Thay đổi tiêu đề
```latex
\title{
    \textbf{Tiêu đề mới của bạn}
}
```

### Thêm tác giả
```latex
\author{
    Tên Tác Giả\\
    \textit{Đơn vị}
}
```

### Thêm hình ảnh external
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{path/to/image.png}
\caption{Mô tả hình}
\label{fig:label}
\end{figure}
```

## Lỗi Thường Gặp

### 1. Lỗi encoding tiếng Việt
```
! Package inputenc Error: Unicode character ...
```
**Giải pháp:** Đảm bảo file được lưu với encoding UTF-8

### 2. Lỗi package không tìm thấy
```
! LaTeX Error: File `package.sty' not found.
```
**Giải pháp:** Cài đặt package qua package manager

### 3. Lỗi PGFPlots
```
! Package pgfplots Error: ...
```
**Giải pháp:** Cập nhật PGFPlots lên phiên bản mới nhất

## Liên Hệ

Nếu có vấn đề, vui lòng kiểm tra:
- [LaTeX Stack Exchange](https://tex.stackexchange.com)
- [Overleaf Documentation](https://www.overleaf.com/learn)

---
*Tạo ngày: 09/02/2026*
