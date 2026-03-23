# 🎯 Thiết kế: Cross-Modal Attention for Causal GNN v2

**Mục tiêu:** Tăng Weighted F1 của Task 2 từ 0.67 lên **>0.70** thông qua cơ chế hội tụ đặc trưng chéo (Cross-modal interactions).

## 1. Cơ sở lý thuyết
Hiện tại, mô hình sử dụng hàm `Concat` để gộp đặc trưng Hình ảnh và Văn bản. Tuy nhiên, `Concat` không cho phép mô hình học được sự phụ thuộc lẫn nhau giữa các thành phần đặc trưng (ví dụ: một từ khóa cụ thể trong Text tương ứng với một vùng đối tượng trong Image).

Cơ chế **Cross-Modal Attention** cho phép:
- Nhánh Văn bản "truy vấn" (Query) các đặc trưng quan trọng từ nhánh Hình ảnh.
- Nhánh Hình ảnh được lọc lại dựa trên ngữ cảnh của Văn bản.

## 2. Kiến trúc đề xuất

### Module `CrossModalFusion`
```python
class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # Text attends to Image
        self.t_attn_i = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Image attends to Text
        self.i_attn_t = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, img_feat, txt_feat):
        # Giả định input (B, D) -> Cần reshape (B, 1, D) cho MultiheadAttention
        q_img = img_feat.unsqueeze(1)
        q_txt = txt_feat.unsqueeze(1)
        
        # Text guiding Image
        attn_img, _ = self.t_attn_i(q_txt, q_img, q_img)
        # Image guiding Text
        attn_txt, _ = self.i_attn_t(q_img, q_txt, q_txt)
        
        # Residual + Norm
        out_img = self.norm1(q_img + attn_img).squeeze(1)
        out_txt = self.norm2(q_txt + attn_txt).squeeze(1)
        
        return torch.cat([out_img, out_txt], dim=1)
```

## 3. Tác động đến Pipeline
- **Forward Pass**: Thêm 1 bước tính toán trước khi vào `causal_head` và `spurious_head`.
- **Memory**: Tăng nhẹ việc sử dụng VRAM (~200MB).
- **Phức tạp**: Cần cẩn thận với việc khởi tạo trọng số để không làm gẫy quá trình hội tụ đang tốt.

## 4. Kế hoạch thực hiện
1. Định nghĩa lớp `CrossModalFusion` trong Notebook.
2. Cập nhật `CausalCrisisV2Model` để sử dụng module này.
3. Chạy lại Multi-seed experiment để kiểm chứng.

---
**Trạng thái:** Đợi User approve thiết kế để bắt đầu implement.
