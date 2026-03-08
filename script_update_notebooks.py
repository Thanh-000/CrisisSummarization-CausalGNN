import json
import glob

# Notebooks to update
notebook_files = glob.glob('notebooks/*.ipynb')

for nb_file in notebook_files:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        for cell in data.get('cells', []):
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    # Fix N=50 -> 500
                    if 'sizes=[50]' in line or 'sizes = [50]' in line:
                        line = line.replace('50', '500')
                        modified = True
                    if 'few_shot_sizes=[50]' in line:
                        line = line.replace('50', '500')
                        modified = True
                        
                    # Fix pip install lint
                    if '!pip install' in line:
                        line = line.replace('!pip install', '%pip install')
                        modified = True
                        
                    new_source.append(line)
                cell['source'] = new_source
                
        if modified:
            with open(nb_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
            print(f"Updated {nb_file}")
    except Exception as e:
        print(f"Error on {nb_file}: {e}")
