import json
import os

nb_path = r'c:\Users\Admin\OneDrive\Desktop\New folder\CrisisSummarization\GEDA_Baseline_Comparison.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Phase 4 (Cell 12): Add Drive logic to shared utilities
cell_12_src = [
    "# === 4.0 Shared utilities ===\n",
    "import time, json, os, csv\n",
    "\n",
    "DRIVE_DIR = '/content/drive/MyDrive/CrisisSummarization'\n",
    "RESULTS_CSV = '/content/geda_results/all_results.csv'\n",
    "\n",
    "# Safe Drive setup for GEDA Results\n",
    "if os.path.exists('/content/drive/MyDrive'):\n",
    "    os.makedirs(f'{DRIVE_DIR}/geda_results', exist_ok=True)\n",
    "    RESULTS_CSV = f'{DRIVE_DIR}/geda_results/all_results.csv'\n",
    "    print(f'[SAFE MODE] GEDA results will map to: {RESULTS_CSV}')\n",
    "else:\n",
    "    os.makedirs('/content/geda_results', exist_ok=True)\n",
    "    print(f'[WARNING] Saving locally to {RESULTS_CSV}')\n",
    "\n",
    "def save_row(row_dict, csv_file=RESULTS_CSV):\n",
    "    file_exists = os.path.isfile(csv_file)\n",
    "    os.makedirs(os.path.dirname(csv_file), exist_ok=True)\n",
    "    with open(csv_file, 'a', newline='') as f:\n",
    "        writer = csv.DictWriter(f, fieldnames=row_dict.keys())\n",
    "        if not file_exists:\n",
    "            writer.writeheader()\n",
    "        writer.writerow(row_dict)\n"
]
nb['cells'][12]['source'] = cell_12_src

# 2. Update Phase 5 (Cell 17): Add Dataset Auto-detect
for i, cell in enumerate(nb['cells']):
    if '# === 5.1 Run Paper 1' in ''.join(cell['source']):
        src = cell['source']
        for j, line in enumerate(src):
            if 'DATASET_PATH =' in line:
                drive_detect = (
                    "DATASET_PATH = './data/CrisisMMD_v2.0'\n"
                    "# Auto-detect Dataset in Drive\n"
                    "DRIVE_DIR = '/content/drive/MyDrive/CrisisSummarization'\n"
                    "if os.path.exists('/content/drive/MyDrive'):\n"
                    "    candidates = ['/content/drive/MyDrive/CrisisMMD_v2.0', '/content/drive/MyDrive/datasets/CrisisMMD_v2.0', f'{DRIVE_DIR}/CrisisMMD_v2.0']\n"
                    "    for c in candidates:\n"
                    "        if os.path.exists(c):\n"
                    "            # symlink the drive folder into ./data so Paper 1 script works natively\n"
                    "            os.makedirs('./data', exist_ok=True)\n"
                    "            if not os.path.exists(DATASET_PATH):\n"
                    "                os.symlink(c, DATASET_PATH)\n"
                    "            print(f'[SAFE MODE] Sourced Paper 1 Dataset from Drive')\n"
                    "            break\n"
                )
                src[j] = drive_detect
                break
        nb['cells'][i]['source'] = src
        break

# 3. Update Phase 7 (Cell 28 - geda_phase7_8.py runner):
for i, cell in enumerate(nb['cells']):
    if '# === 7.3 Run ALL GEDA experiments' in ''.join(cell['source']):
        src = [
            "# === 7.3 Run ALL GEDA experiments ===\n",
            "import os\n",
            "\n",
            "# Use auto-detected Drive paths for safe execution of Phase 7\n",
            "RESULTS_CSV = '/content/geda_results/all_results.csv'\n",
            "RESULTS_DIR = '/content/geda_results'\n",
            "DATASET_PATH = '/content/datasets/CrisisMMD_v2.0'\n",
            "\n",
            "DRIVE_DIR = '/content/drive/MyDrive/CrisisSummarization'\n",
            "if os.path.exists('/content/drive/MyDrive'):\n",
            "    RESULTS_CSV = f'{DRIVE_DIR}/geda_results/all_results.csv'\n",
            "    RESULTS_DIR = f'{DRIVE_DIR}/geda_results'\n",
            "    candidates = ['/content/drive/MyDrive/CrisisMMD_v2.0', '/content/drive/MyDrive/datasets/CrisisMMD_v2.0', f'{DRIVE_DIR}/CrisisMMD_v2.0']\n",
            "    for c in candidates:\n",
            "        if os.path.exists(c):\n",
            "            DATASET_PATH = c\n",
            "            break\n",
            "\n",
            "!python /content/geda_phase7_8.py \\\n",
            "    --dataset_path {DATASET_PATH} \\\n",
            "    --results_csv {RESULTS_CSV} \\\n",
            "    --output_dir {RESULTS_DIR} \\\n",
            "    --phase 7\n",
            "\n",
            "print('\\n=== Phase 7 complete! ===')\n"
        ]
        nb['cells'][i]['source'] = src
        break

# 4. Add Phase 9 & 10 at the end if not exists
last_md = ''.join([c.get('source', [''])[0] for c in nb['cells'] if c['cell_type'] == 'markdown'])
if 'Phase 9' not in last_md:
    phase_9_10 = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Phase 9: CausalCrisis Experiments\n", "\n", "Load dataset from Drive if available, save results persistently to Drive, and run 60 rounds of CausalCrisis."]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "# === 9.1 Mount Google Drive (if not already mounted) ===\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    import os\n",
                "    if not os.path.exists('/content/drive/MyDrive'):\n",
                "        drive.mount('/content/drive')\n",
                "        print('Google Drive connected.')\n",
                "    else:\n",
                "        print('Google Drive already connected.')\n",
                "except ImportError:\n",
                "    print('Not in Colab. Skipping Drive mount.')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "# === 9.2 Upload Causal Models ===\n",
                "import os\n",
                "\n",
                "# In case the user uploads them via the sidebar:\n",
                "if not os.path.exists('/content/causal_crisis_model.py'):\n",
                "    print('Please upload causal_crisis_model.py and causal_crisis_trainer.py via Colab sidebar.')\n",
                "else:\n",
                "    print('[OK] causal_crisis files are present.')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "# === 9.3 Run 60 CausalCrisis Experiments ===\n",
                "import os, sys\n",
                "sys.path.insert(0, '/content')\n",
                "\n",
                "if not os.path.exists('/content/causal_crisis_trainer.py'):\n",
                "    print('Missing scripts! Stopping.')\n",
                "else:\n",
                "    from causal_crisis_trainer import run_causal_all_experiments\n",
                "    \n",
                "    # Auto-detect Drive for safe saving & loading\n",
                "    DRIVE_DIR = '/content/drive/MyDrive/CrisisSummarization'\n",
                "    RESULTS_CSV = '/content/causal_results/all_results.csv'\n",
                "    DATASET = '/content/datasets/CrisisMMD_v2.0'\n",
                "    \n",
                "    if os.path.exists('/content/drive/MyDrive'):\n",
                "        os.makedirs(f'{DRIVE_DIR}/causal_results', exist_ok=True)\n",
                "        RESULTS_CSV = f'{DRIVE_DIR}/causal_results/all_results.csv'\n",
                "        print(f'  [SAFE MODE] Saving results directly to Google Drive: {RESULTS_CSV}')\n",
                "        \n",
                "        drive_datasets = [\n",
                "            '/content/drive/MyDrive/CrisisMMD_v2.0',\n",
                "            '/content/drive/MyDrive/datasets/CrisisMMD_v2.0',\n",
                "            f'{DRIVE_DIR}/CrisisMMD_v2.0'\n",
                "        ]\n",
                "        for d_path in drive_datasets:\n",
                "            if os.path.exists(d_path):\n",
                "                DATASET = d_path\n",
                "                print(f'  [SAFE MODE] Loading dataset from Google Drive: {DATASET}')\n",
                "                break\n",
                "    else:\n",
                "        print(f'  [WARNING] Google Drive not mounted. Saving locally.')\n",
                "    \n",
                "    run_causal_all_experiments(\n",
                "        dataset_path=DATASET,\n",
                "        seeds=(42, 123, 456, 789, 1024),\n",
                "        tasks=('task1', 'task2', 'task3'),\n",
                "        few_shot_sizes=(50, 100, 250, 500),\n",
                "        device='cuda',\n",
                "        results_csv=RESULTS_CSV,\n",
                "        variant_name='causal_full',\n",
                "        use_causal=True,\n",
                "        use_intervention=True,\n",
                "        use_causal_graph=True,\n",
                "    )\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Phase 10: Comparison - GEDA vs CausalCrisis\n"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "# === 10.1 CausalCrisis vs GEDA unified analysis ===\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from scipy import stats\n",
                "import os\n",
                "\n",
                "print('\\n' + '='*70)\n",
                "print('  PHASE 10: GEDA vs CausalCrisis COMPARISON')\n",
                "print('='*70)\n",
                "\n",
                "geda_csv = '/content/geda_results/all_results.csv'\n",
                "if not os.path.exists(geda_csv) and os.path.exists('/content/drive/MyDrive'):\n",
                "    drive_geda = '/content/drive/MyDrive/CrisisSummarization/geda_results/all_results.csv'\n",
                "    if os.path.exists(drive_geda):\n",
                "        geda_csv = drive_geda\n",
                "        print(f'  [SAFE MODE] Loading GEDA results from: {geda_csv}')\n",
                "\n",
                "causal_csv = RESULTS_CSV if 'RESULTS_CSV' in locals() else '/content/causal_results/all_results.csv'\n",
                "if not os.path.exists(causal_csv) and os.path.exists('/content/drive/MyDrive'):\n",
                "    drive_causal = '/content/drive/MyDrive/CrisisSummarization/causal_results/all_results.csv'\n",
                "    if os.path.exists(drive_causal):\n",
                "        causal_csv = drive_causal\n",
                "\n",
                "dfs = {}\n",
                "if os.path.exists(geda_csv):\n",
                "    dfs['geda_full'] = pd.read_csv(geda_csv)\n",
                "    print(f'  GEDA: {len(dfs[\"geda_full\"])} runs')\n",
                "\n",
                "if os.path.exists(causal_csv):\n",
                "    dfs['causal_full'] = pd.read_csv(causal_csv)\n",
                "    print(f'  CausalCrisis: {len(dfs[\"causal_full\"])} runs')\n",
                "\n",
                "if len(dfs) < 2:\n",
                "    print('  [WARN] Need both result files to compare!')\n",
                "else:\n",
                "    df_all = pd.concat(dfs.values(), ignore_index=True)\n",
                "    task_names = {'task1': 'Informativeness', 'task2': 'Humanitarian', 'task3': 'Damage'}\n",
                "\n",
                "    print('\\n' + '='*70)\n",
                "    print('  COMPARISON TABLE: Weighted F1 (mean ± std)')\n",
                "    print('='*70)\n",
                "    for task in ['task1', 'task2', 'task3']:\n",
                "        print(f'\\n--- {task_names[task]} ---')\n",
                "        print(f'{\"Model\":20s} {\"N=50\":>12s} {\"N=100\":>12s} {\"N=250\":>12s} {\"N=500\":>12s}')\n",
                "        print('-' * 75)\n",
                "        for model_name in ['geda_full', 'causal_full']:\n",
                "            row = f'{model_name:20s}'\n",
                "            for n in [50, 100, 250, 500]:\n",
                "                mask = (df_all['model'] == model_name) & (df_all['task'] == task) & (df_all['few_shot'] == n)\n",
                "                vals = df_all.loc[mask, 'weighted_f1'].values * 100\n",
                "                if len(vals) > 0:\n",
                "                    row += f'  {vals.mean():.1f}±{vals.std():.1f}'\n",
                "                else:\n",
                "                    row += '         N/A'\n",
                "            print(row)\n",
                "            \n",
                "    print('\\n' + '='*70)\n",
                "    print('  SIGNIFICANCE TESTS')\n",
                "    print('='*70)\n",
                "    for task in ['task1', 'task2', 'task3']:\n",
                "        print(f'\\n--- {task_names[task]} ---')\n",
                "        for n in [50, 100, 250, 500]:\n",
                "            g = df_all.loc[(df_all['model'] == 'geda_full') & (df_all['task'] == task) & (df_all['few_shot'] == n), 'weighted_f1'].values\n",
                "            c = df_all.loc[(df_all['model'] == 'causal_full') & (df_all['task'] == task) & (df_all['few_shot'] == n), 'weighted_f1'].values\n",
                "            if len(g) >= 2 and len(c) >= 2:\n",
                "                t, p = stats.ttest_ind(c, g)\n",
                "                d = (c.mean() - g.mean()) * 100\n",
                "                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'\n",
                "                print(f'  N={n:3d}: diff={d:+.1f}% p={p:.4f} {sig}')\n",
                "\n",
                "    if 'domain_acc_causal' in df_all.columns:\n",
                "        cr = df_all[df_all['model'] == 'causal_full']\n",
                "        if 'domain_acc_causal' in cr.columns:\n",
                "            print('\\n' + '='*70)\n",
                "            print('  DOMAIN INVARIANCE CHECK (Target: ≈ 14.3%)')\n",
                "            for task in ['task1', 'task2', 'task3']:\n",
                "                vals = cr.loc[cr['task'] == task, 'domain_acc_causal'].values\n",
                "                vals = vals[vals >= 0]\n",
                "                if len(vals) > 0:\n",
                "                    print(f'  {task_names[task]:20s}: {vals.mean()*100:.1f}%')\n"
            ]
        }
    ]
    nb['cells'].extend(phase_9_10)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Success! Notebook updated.')
