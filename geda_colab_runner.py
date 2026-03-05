"""
GEDA Colab Runner - Script tong quat chay tren Google Colab
============================================================
Chay ca 2 baselines (Paper1 GNN + Paper2 DiffAttn) va so sanh.
Upload file nay len Colab roi chay tung ham theo notebook.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import torch
import random
from pathlib import Path
from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

class GEDAConfig:
    """Cau hinh trung tam cho toan bo pipeline."""
    
    # Paths
    BASE_DIR = "/content"
    DATASET_DIR = "/content/datasets"
    DATASET_PATH = "/content/datasets/CrisisMMD_v2.0"
    RESULTS_DIR = "/content/geda_results"
    FIGURES_DIR = "/content/geda_results/figures"
    
    # Repos
    PAPER1_REPO = "jdnascim/mm-class-for-disaster-data-with-gnn"
    PAPER1_DIR = "/content/mm-class-for-disaster-data-with-gnn"
    PAPER2_REPO = "Munia03/Multimodal_Crisis_Event"
    PAPER2_DIR = "/content/Multimodal_Crisis_Event"
    
    # Dataset
    DATASET_URL = "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"
    LLAVA_GDRIVE_ID = "1xpYcshz9_KkQqw3tN9E86Df7UgmN6BaY"
    
    # Experiment settings
    RANDOM_SEEDS = [42, 123, 456, 789, 1024]
    TASKS = {
        "task1": {"name": "Informative", "num_classes": 2},
        "task2": {"name": "Humanitarian", "num_classes": 6},
        "task3": {"name": "Damage Severity", "num_classes": 3},
    }
    
    # Paper 1 settings (EXACT from exp_5.sh)
    P1_FEW_SHOT = [50, 100, 250, 500]
    P1_EPOCHS = 2000
    P1_EARLY_STOPPING = 300
    P1_LR = 1e-5
    P1_ARCH = "sage-2l-norm-res"
    P1_EXP_ID = 5
    P1_IMAGEFT = "maxvit"
    P1_FUSION = "late"
    P1_BEST_MODEL = "best_hm"
    
    # Paper 2 settings (from train.sh)
    P2_BATCH_SIZE = 32
    P2_LR = 0.001
    P2_MAX_ITER = 50


def set_seed(seed=42):
    """Dat seed cho reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 2. SETUP FUNCTIONS
# ============================================================

def check_gpu():
    """Kiem tra GPU va in thong tin."""
    print("=" * 50)
    print("  GPU STATUS")
    print("=" * 50)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name}")
        print(f"  VRAM: {mem:.1f} GB")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        return True
    else:
        print("  [!] No GPU detected!")
        return False


def install_dependencies():
    """Cai dat tat ca dependencies."""
    print("\n[1/3] Installing PyG...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"],
                   check=True, capture_output=True)
    
    print("[2/3] Installing ML deps...")
    deps = [
        "numpy", "pandas", "scikit-learn", "tqdm", "matplotlib",
        "pillow", "scikit-image", "requests", "umap-learn", "timm",
        "sentence-transformers", "jsonlines", "psutil", "gdown",
        "transformers", "termcolor", "nltk", "imageio",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps,
                   check=True, capture_output=True)
    
    print("[3/3] NLTK data...")
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    print("[OK] All dependencies installed!")


def clone_repos():
    """Clone ca 2 repos baseline."""
    for url, d in [
        (f"https://github.com/{GEDAConfig.PAPER1_REPO}.git", GEDAConfig.PAPER1_DIR),
        (f"https://github.com/{GEDAConfig.PAPER2_REPO}.git", GEDAConfig.PAPER2_DIR),
    ]:
        if not os.path.isdir(d):
            subprocess.run(["git", "clone", "--depth", "1", url, d], check=True)
            print(f"  Cloned: {os.path.basename(d)}")
        else:
            print(f"  Exists: {os.path.basename(d)}")


def download_dataset():
    """Download va extract CrisisMMD v2.0."""
    cfg = GEDAConfig
    os.makedirs(cfg.DATASET_DIR, exist_ok=True)
    
    if os.path.isdir(cfg.DATASET_PATH):
        print(f"  Dataset exists: {cfg.DATASET_PATH}")
        return
    
    tar_path = f"{cfg.DATASET_DIR}/CrisisMMD_v2.0.tar.gz"
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1.8e9:
        print("  Downloading CrisisMMD v2.0...")
        subprocess.run(["wget", "-q", "--show-progress", "-c", "-O", tar_path, cfg.DATASET_URL])
    
    print("  Extracting...")
    subprocess.run(["tar", "xzf", tar_path, "-C", cfg.DATASET_DIR])
    print("  [OK] Dataset ready!")


def download_llava_data():
    """Download LLaVA descriptions (Paper 2 needs this)."""
    import gdown
    cfg = GEDAConfig
    test_tsv = f"{cfg.DATASET_PATH}/crisismmd_datasplit_settingA/task_damage_text_img_train.tsv"
    
    # Check if already has LLaVA
    if os.path.exists(test_tsv):
        with open(test_tsv, 'r') as f:
            if 'LLaVA_text' in f.readline():
                print("  LLaVA data already present!")
                return
    
    print("  Downloading LLaVA data...")
    gdown.download(id=cfg.LLAVA_GDRIVE_ID, output="llava_data.zip", quiet=False)
    
    import zipfile
    with zipfile.ZipFile("llava_data.zip", 'r') as z:
        z.extractall(cfg.DATASET_PATH)
    os.remove("llava_data.zip")
    print("  [OK] LLaVA data extracted!")


def patch_paper2_code():
    """Fix Paper 2 code issues cho Colab."""
    cfg = GEDAConfig
    
    # Fix paths.py
    with open(f"{cfg.PAPER2_DIR}/paths.py", 'w') as f:
        f.write(f"dataroot = '{cfg.DATASET_DIR}'\n")
    
    # Fix crisismmd_dataset.py - comment out Llama/DeepSeek tokenizers
    ds_file = f"{cfg.PAPER2_DIR}/crisismmd_dataset.py"
    with open(ds_file, 'r') as f:
        content = f.read()
    
    import re
    if 'llama_model_name="meta-llama' in content and '# llama_model_name' not in content:
        fixes = [
            (r'llama_model_name="meta-llama/Llama-3.2-1B"',
             '# llama_model_name="meta-llama/Llama-3.2-1B"'),
            (r"self\.llamma_tokenizer = AutoTokenizer\.from_pretrained\(llama_model_name\)",
             '# self.llamma_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)'),
            (r"self\.llamma_tokenizer\.pad_token = self\.llamma_tokenizer\.eos_token",
             '# self.llamma_tokenizer.pad_token = self.llamma_tokenizer.eos_token'),
            (r'deepseek_model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"',
             '# deepseek_model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"'),
            (r"self\.deepseek_tokenizer = AutoTokenizer\.from_pretrained\(deepseek_model_name\)",
             '# self.deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)'),
            (r"'llamma_tokens': self\.tokenize_llamma\(caption\),",
             "# 'llamma_tokens': self.tokenize_llamma(caption),"),
        ]
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        with open(ds_file, 'w') as f:
            f.write(content)
    
    # Fix verbose kwarg in main.py
    main_file = f"{cfg.PAPER2_DIR}/main.py"
    with open(main_file, 'r') as f:
        mc = f.read()
    mc = mc.replace("verbose=True", "")
    with open(main_file, 'w') as f:
        f.write(mc)
    
    # Fix hardcoded paths
    import glob
    for py_file in glob.glob(f"{cfg.PAPER2_DIR}/*.py"):
        with open(py_file, 'r', encoding='utf-8') as f:
            c = f.read()
        if '/content/datasets/CrisisMMD_v2.0/CrisisMMD_v2.0' in c:
            c = c.replace('/content/datasets/CrisisMMD_v2.0/CrisisMMD_v2.0',
                          '/content/datasets/CrisisMMD_v2.0')
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(c)
    
    print("  [OK] Paper 2 code patched!")


# ============================================================
# 3. EVALUATION METRICS (thong nhat tu ca 2 papers)
# ============================================================

from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred):
    """Tinh TAT CA metrics tu ca 2 papers."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ============================================================
# 4. BASELINE RUNNERS
# ============================================================

def run_paper1_experiment(size=50, split=0, run_id=0):
    """Chay 1 experiment Paper 1 (GNN) -- EXACT exp_5.sh."""
    cfg = GEDAConfig
    datasplit = f"{size}_s{split}"
    
    cmd = [
        sys.executable, "feature_fusion.py",
        "--gpu_id", "0",
        "--arch", cfg.P1_ARCH,
        "--epochs", str(cfg.P1_EPOCHS),
        "--lr", str(cfg.P1_LR),
        "--weight_decay", "1e-3",
        "--dropout", "0.5",
        "--n_neigh_train", "16",
        "--n_neigh_full", "16",
        "--lbl_train_frac", "0.4",
        "--imagepath", f"{cfg.DATASET_PATH}/data_image",
        "--datasplit", datasplit,
        "--reg", "l2",
        "--l2_lambda", "1e-2",
        "--exp_id", str(cfg.P1_EXP_ID),
        "--imageft", cfg.P1_IMAGEFT,
        "--textft", "mpnet",
        "--fusion", cfg.P1_FUSION,
        "--loss", "nll",
        "--shuffle_split",
        "--early_stopping", str(cfg.P1_EARLY_STOPPING),
        "--best_model", cfg.P1_BEST_MODEL,
        "--run_id", str(run_id),
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cfg.PAPER1_DIR,
                           capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    
    # Parse results from JSON
    resf = f"{cfg.PAPER1_DIR}/results/CrisisMMD/gnn/{cfg.P1_EXP_ID}/{size}_s{split}_{run_id}.json"
    if os.path.exists(resf):
        with open(resf, 'r') as f:
            data = json.load(f)
        return {
            "model": "paper1_gnn",
            "task": "task1",
            "seed": split,
            "few_shot": size,
            "weighted_f1": data.get("f1_test", 0),
            "balanced_accuracy": data.get("bacc_test", 0),
            "train_time_s": elapsed,
        }
    return None


def run_paper2_experiment(task="task1", seed=42):
    """Chay 1 experiment Paper 2 (DiffAttn)."""
    cfg = GEDAConfig
    set_seed(seed)
    
    model_name = f"{task}_seed{seed}"
    cmd = [
        sys.executable, "main.py",
        "--model_name", model_name,
        "--mode", "both",
        "--task", task,
        "--batch_size", str(cfg.P2_BATCH_SIZE),
        "--device", "cuda",
        "--max_iter", str(cfg.P2_MAX_ITER),
        "--text_from", "llava",
        "--learning_rate", str(cfg.P2_LR),
        "--num_workers", "0",
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cfg.PAPER2_DIR,
                           capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    
    # Parse from log file
    import glob
    logs = sorted(glob.glob(f"{cfg.PAPER2_DIR}/output/{model_name}/output_*.log"))
    
    metrics = {"model": "paper2_diffattn", "task": task, "seed": seed, "train_time_s": elapsed}
    
    if logs:
        with open(logs[-1], 'r') as f:
            log_content = f.read()
        
        import re
        for m_name, pattern in [
            ("micro_f1", r"Micro F1: ([\d.]+)"),
            ("macro_f1", r"Macro F1: ([\d.]+)"),
            ("weighted_f1", r"Weighted F1: ([\d.]+)"),
            ("accuracy", r"Test set accuracy ([\d.]+)"),
        ]:
            matches = re.findall(pattern, log_content)
            if matches:
                metrics[m_name] = float(matches[-1])
    
    return metrics


# ============================================================
# 5. RESULTS MANAGEMENT
# ============================================================

import csv

RESULT_HEADERS = [
    "model", "task", "seed", "few_shot",
    "accuracy", "balanced_accuracy",
    "micro_f1", "macro_f1", "weighted_f1",
    "train_time_s", "timestamp",
]


def save_result(result_dict, csv_path=None):
    """Luu 1 result vao CSV (append mode)."""
    if csv_path is None:
        csv_path = f"{GEDAConfig.RESULTS_DIR}/all_results.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    result_dict["timestamp"] = datetime.now().isoformat()
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_HEADERS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def load_results(csv_path=None):
    """Load tat ca results tu CSV."""
    if csv_path is None:
        csv_path = f"{GEDAConfig.RESULTS_DIR}/all_results.csv"
    
    if not os.path.exists(csv_path):
        return []
    
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["accuracy", "balanced_accuracy", "micro_f1", "macro_f1",
                        "weighted_f1", "train_time_s"]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            results.append(row)
    return results


# ============================================================
# 6. REPORT GENERATION
# ============================================================

def generate_comparison_table(results, metric="weighted_f1"):
    """Tao bang so sanh giua cac models."""
    import pandas as pd
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON REPORT")
    print("=" * 70)
    
    for task_id, task_info in GEDAConfig.TASKS.items():
        task_results = [r for r in results if r.get("task") == task_id]
        if not task_results:
            continue
        
        print(f"\n--- {task_info['name']} ({task_id}) ---")
        print(f"{'Model':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>4}")
        print("-" * 60)
        
        models = set(r["model"] for r in task_results)
        model_scores = {}
        
        for model in sorted(models):
            scores = [r[metric] for r in task_results
                     if r["model"] == model and metric in r and r[metric]]
            scores = [float(s) for s in scores if s]
            
            if scores:
                mean = np.mean(scores)
                std = np.std(scores, ddof=1) if len(scores) > 1 else 0
                model_scores[model] = scores
                print(f"  {model:<23} {mean*100:>7.2f}% {std*100:>7.2f}% "
                      f"{min(scores)*100:>7.2f}% {max(scores)*100:>7.2f}% {len(scores):>3}")
        
        # Significance test
        model_list = sorted(model_scores.keys())
        if len(model_list) >= 2:
            print(f"\n  Pairwise t-tests (alpha=0.05):")
            for i in range(len(model_list)):
                for j in range(i+1, len(model_list)):
                    s1 = model_scores[model_list[i]]
                    s2 = model_scores[model_list[j]]
                    min_len = min(len(s1), len(s2))
                    if min_len >= 2:
                        t_stat, p_val = stats.ttest_rel(s1[:min_len], s2[:min_len])
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"    {model_list[i]} vs {model_list[j]}: "
                              f"p={p_val:.4f} ({sig})")


def generate_latex_table(results, metric="weighted_f1"):
    """Tao LaTeX table cho bao cao."""
    print("\n% --- LaTeX Table ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Baseline Comparison Results}")
    print("\\begin{tabular}{l" + "c" * len(GEDAConfig.TASKS) + "}")
    print("\\hline")
    
    header = "Model & " + " & ".join(
        t["name"] for t in GEDAConfig.TASKS.values()
    ) + " \\\\"
    print(header)
    print("\\hline")
    
    models = set(r["model"] for r in results)
    for model in sorted(models):
        row = [model.replace("_", "\\_")]
        for task_id in GEDAConfig.TASKS:
            scores = [float(r[metric]) for r in results
                     if r["model"] == model and r.get("task") == task_id
                     and metric in r and r[metric]]
            if scores:
                mean = np.mean(scores)
                std = np.std(scores, ddof=1) if len(scores) > 1 else 0
                row.append(f"{mean*100:.1f} $\\pm$ {std*100:.1f}")
            else:
                row.append("-")
        print(" & ".join(row) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


# ============================================================
# 7. MAIN ORCHESTRATOR
# ============================================================

def run_full_pipeline():
    """Chay toan bo pipeline."""
    cfg = GEDAConfig
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("  GEDA EVALUATION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    
    # Phase 1: Setup
    print("\n>> Phase 1: Setup")
    check_gpu()
    install_dependencies()
    clone_repos()
    download_dataset()
    download_llava_data()
    patch_paper2_code()
    
    # Phase 2: Run Paper 2 baselines (3 tasks x 5 seeds)
    print("\n>> Phase 2: Paper 2 (DiffAttn) Baselines")
    for task in ["task1", "task2", "task3"]:
        for seed in cfg.RANDOM_SEEDS:
            print(f"\n  [{task}] seed={seed}")
            result = run_paper2_experiment(task=task, seed=seed)
            if result:
                save_result(result)
                wf1 = result.get("weighted_f1", "?")
                print(f"    -> weighted_f1={wf1}")
    
    # Phase 3: Run Paper 1 baselines (4 sizes x 10 splits = 40 runs)
    print("\n>> Phase 3: Paper 1 (GNN) Baselines")
    for size in cfg.P1_FEW_SHOT:
        for split in range(10):
            print(f"\n  [task1] size={size}, split={split}")
            result = run_paper1_experiment(size=size, split=split)
            if result:
                save_result(result)
    
    # Phase 4: Report
    print("\n>> Phase 4: Generate Reports")
    results = load_results()
    generate_comparison_table(results)
    generate_latex_table(results)
    
    print(f"\n{'='*50}")
    print(f"  Pipeline complete!")
    print(f"  Results: {cfg.RESULTS_DIR}/all_results.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_full_pipeline()
