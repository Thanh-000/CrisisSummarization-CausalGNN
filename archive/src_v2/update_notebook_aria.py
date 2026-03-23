import json
import os

path = r'c:\Users\Admin\OneDrive\Desktop\New folder\CrisisSummarization\notebooks\Causal_Crisis_Colab_Setup.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        i = 0
        while i < len(source):
            line = source[i]
            if 'wget' in line and 'https://crisisnlp.qcri.org' in line:
                source[i] = '    !apt-get install -y aria2 > /dev/null 2>&1\n'
                source.insert(i+1, '    !aria2c -x 16 -s 16 -q https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz -d /content -o CrisisMMD_v2.0.tar.gz\n')
                print('Replaced wget code')
                i += 1 # skip next iter over inserted line
            if 'Dang tai xuong' in line and 'Tu Server' in line:
                source[i] = '    print("Dang tai xuong CrisisMMD v2.0 dataset dang nen (Khoang 1.8GB) bang cong nghe da luong aria2c t\\u1ed1c \\u0111\\u1ed9 cao...")\n'
                print('Replaced wget string')
            i += 1
            
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if 'wget' in line and 'thanh ti' in line:
                source[i] = 'Sử dụng `aria2c` tải đa luồng tốc độ cao hơn gấp 10 lần (vừa được khôi phục theo yêu cầu).\\n'
                print('Replaced wget markdown')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
