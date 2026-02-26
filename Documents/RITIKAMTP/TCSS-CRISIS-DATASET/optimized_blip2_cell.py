import torch
import gc
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# 1. Dọn dẹp bộ nhớ GPU từ bước trước (CLIP)
print("🧹 Đang dọn dẹp bộ nhớ VRAM để nhường chỗ cho BLIP-2...")
try:
    if 'model' in globals(): del model
    if 'processor' in globals(): del processor
    if 'image_inputs' in globals(): del image_inputs
    if 'text_inputs' in globals(): del text_inputs
    if 'image_features' in globals(): del image_features
    if 'text_features' in globals(): del text_features
except:
    pass
gc.collect()
torch.cuda.empty_cache()
print("✅ Đã giải phóng bộ nhớ GPU.")

# 2. Lấy danh sách ảnh tốt nhất (Tái tạo từ top_indices của bước CLIP)
# Lưu ý: 'image_files' và 'top_indices' phải có giá trị từ các bước trước
try:
    if 'image_files' not in globals() or 'top_indices' not in globals():
         raise NameError("Thiếu biến image_files hoặc top_indices")
    
    top_images = [image_files[int(i)] for i in top_indices]
    print(f"📸 {len(top_images)} ảnh được chọn để tạo chú thích.")
except NameError as e:
    print(f"❌ Lỗi: {e}. Bạn cần chạy ô CLIP thành công trước khi chạy ô này.")
    raise e

# 3. Tải BLIP-2 Model
print("⏳ Đang tải mô hình BLIP-2 (khoảng 15GB VRAM required for float32, <8GB for float16)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

try:
    # Sử dụng float16 để tiết kiệm bộ nhớ (bắt buộc trên T4 nếu dùng model lớn)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=dtype
    )
    model.to(device)
    print(f"✅ Đã tải model thành công trên {device} ({dtype}).")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    raise e

# 4. Tạo caption tuần tự
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        # Preprocessing image
        inputs = processor(images=image, return_tensors="pt").to(device, dtype)
        
        # Generation
        generated_ids = model.generate(**inputs, max_new_tokens=30)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    except Exception as e:
        print(f"⚠️ Lỗi ảnh {image_path}: {e}")
        return ""

print("🚀 Đang tạo chú thích cho ảnh...")
captions = []
for idx, path in enumerate(top_images):
    print(f"[{idx+1}/{len(top_images)}] Xử lý: {os.path.basename(path)}")
    cap = generate_caption(path)
    if cap:
        print(f"   -> Caption: {cap}")
        captions.append(cap)
    else:
        print("   -> (Không tạo được caption)")

# Kết quả cuối cùng cho bước tiếp theo
concatenated_captions = " ".join(captions)
print(f"\n✅ Dữ liệu đầu vào cho bước tóm tắt cuối (concatenated_captions):\n{concatenated_captions[:200]}...")
