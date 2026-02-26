import os
from PIL import Image, UnidentifiedImageError

images_folder = 'images'

# Lấy danh sách file ảnh
image_files_raw = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

images = []
valid_image_files = []

print(f"Tìm thấy {len(image_files_raw)} file ảnh. Đang tiến hành đọc...")

for img_path in image_files_raw:
    try:
        # Thử mở ảnh và convert sang RGB để tránh lỗi kênh màu sau này
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        valid_image_files.append(img_path)
    except (UnidentifiedImageError, OSError) as e:
        # Nếu lỗi, in ra cảnh báo và bỏ qua file đó
        print(f"⚠️ Bỏ qua ảnh lỗi: {img_path} - {e}")

# Cập nhật lại danh sách image_files chỉ chứa các file hợp lệ
# Điều này cực kỳ quan trọng để đảm bảo index của 'images' và 'image_files' khớp nhau
image_files = valid_image_files

print(f"✅ Đã tải thành công {len(images)} ảnh hợp lệ.")
