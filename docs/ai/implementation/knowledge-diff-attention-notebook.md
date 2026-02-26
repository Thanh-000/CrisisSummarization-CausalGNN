# Knowledge Capture: Differential Attention Notebook

## Overview
- **Path**: `new_colab_notebook.ipynb`
- **Purpose**: Tái tạo kết quả của bài báo "Differential Attention for Multimodal Crisis Event Analysis" (CVPRw MMFM 2025).
- **Core Technology**: Differential Attention, CLIP, PyTorch.
- **Data Input**: Ảnh (từ CrisisMMD v2.0) + LLaVA-generated text descriptions (thay vì simple tweet_text).
- **Last Updated**: 26/02/2026

## Implementation Details
1. **Repository**: Clone từ `Munia03/Multimodal_Crisis_Event`.
2. **Data & Preprocessing**:
   - Khắc phục lỗi `meta-llama/Llama-3.2-1B` và `DeepSeek` tokenizer bằng cách loại bỏ hoàn toàn do model chỉ dùng CLIP.
   - Sửa file `crisismmd_dataset.py` và `main.py` để script chạy ổn định trên Colab.
   - Task 3 (Damage Severity): Ban đầu thử class weights `[3.0, 1.5, 1.0]` nhưng **đã loại bỏ** — sử dụng `nn.CrossEntropyLoss()` mặc định theo đúng paper gốc.
3. **Training Configuration**:
   - `batch_size`: 32
   - `learning_rate`: 0.001
   - `max_iter`: 80 cho Task 3, 50 cho Task 1 & 2.
   - Optimizer: SGD.
   - Early stopping: patience = 5 (theo `trainer.py`).
4. **Model Weights**: Lưu dưới dạng `best.pt` (không phải `.pth`).

## Training Results (3 Tasks)

| Task | Mô tả | Num Classes | Status |
|------|--------|-------------|--------|
| Task 1 | Informative vs Not | 2 | ✅ Trained |
| Task 2 | Information Type | 8 | ✅ Trained |
| Task 3 | Damage Severity | 3 | ✅ Trained |

### Task 3 Notes
- Class distribution: imbalanced (little/no > mild > severe).
- Custom class weights đã được **loại bỏ** vì ảnh hưởng tiêu cực đến performance.
- Performance thấp hơn paper một phần do LLaVA hallucination và class imbalance tự nhiên.

## Dependencies & Architecture
- `models_clip.py`: Định nghĩa `CLIP_CrisiKAN` (chứa Differential Attention mapping).
- `transformers`: Dùng `CLIPTokenizer` + `CLIPModel` (openai/clip-vit-base-patch32).
- `diff_transformer.py`: `MultiheadDiffAttn` — core differential attention module.
- Modality Text đầu vào mặc định được lấy từ file source LLaVA.

## Notebook Structure (new_colab_notebook.ipynb)

| Section | Mô tả |
|---------|--------|
| PHẦN 1 | Clone repo, install dependencies |
| PHẦN 2 | Dataset download & setup |
| PHẦN 3 | Fix code (dataset paths, tokenizer, main.py) |
| PHẦN 4 | Training (Task 1, 2, 3 + unimodal baselines) |
| PHẦN 5 | Comparison with Paper Results |
| 3.4 | Qualitative Analysis (2×3 grid visualization) |
| PHẦN 6 | Interactive Prediction |
| PHẦN 7 | Save Results |

## Additional Insights
- Đây là baseline comparison với pipeline GNN hiện hữu (IEEE Access 2025).
- Model output: `output/task{1,2,3}_*/best.pt`
- Qualitative analysis: hiển thị 6 samples với confidence scores, viền xanh/đỏ cho correct/wrong predictions.

## Lessons Learned
1. **Class weights cho Task 3 không cần thiết** — paper gốc dùng CrossEntropyLoss mặc định.
2. **Model file extension**: `.pt` chứ không phải `.pth`.
3. **args.py**: Hàm là `get_args()` không phải `get_parser()`.
4. **Jupyter compatibility**: Cần mock `sys.argv` khi gọi `argparse` trong notebook.

## Next Steps
- So sánh trực tiếp hiệu năng DiffAttn (CVPRw) với GNN pipeline (IEEE Access).
- Tạo bảng comparison thống nhất giữa 2 phương pháp.
