import base64

with open('geda_trainer.py', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

script = f'''import base64
import os

b64_data = "{b64}"

with open('/content/geda_trainer.py', 'wb') as f:
    f.write(base64.b64decode(b64_data))

print("\\n✅ Đã chèn hoàn chỉnh code gốc geda_trainer.py từ máy tính lên Colab. Hủy bỏ mọi cache/lỗi cú pháp cũ.")
'''

with open('colab_patch_script.txt', 'w', encoding='utf-8') as f:
    f.write(script)
