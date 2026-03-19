"""
CausalCrisis Colab Experiment Runner
=====================================
File nay duoc paste truc tiep vao Colab notebook cell.
Chay toan bo 60 runs CausalCrisis + so sanh voi GEDA baseline.

Usage (trong Colab):
  1. Upload causal_crisis_model.py va causal_crisis_trainer.py
  2. Copy noi dung file nay vao cell moi
  3. Run cell
"""

# ============================================================
# PHASE 9: RUN CAUSALCRISIS EXPERIMENTS
# ============================================================

import os
import sys
import time

# Dam bao import duoc
sys.path.insert(0, '/content')

# Verify files exist
for f in ['causal_crisis_model.py', 'causal_crisis_trainer.py']:
    path = f'/content/{f}'
    if os.path.exists(path):
        print(f"  [OK] {f}")
    else:
        print(f"  [MISSING] {f} — Please upload!")

print()

# === RUN 60 EXPERIMENTS ===
from causal_crisis_trainer import run_causal_all_experiments

# === Auto-detect Google Drive for safe saving & loading ===
DRIVE_DIR = "/content/drive/MyDrive/CrisisSummarization"
RESULTS_CSV = "/content/causal_results/all_results.csv"
DATASET = "/content/datasets/CrisisMMD_v2.0"

if os.path.exists("/content/drive/MyDrive"):
    # 1. Setup results path in Drive
    os.makedirs(f"{DRIVE_DIR}/causal_results", exist_ok=True)
    RESULTS_CSV = f"{DRIVE_DIR}/causal_results/all_results.csv"
    print(f"  [SAFE MODE] Saving results directly to Google Drive: {RESULTS_CSV}")
    
    # 2. Check if dataset exists in Drive
    drive_datasets = [
        "/content/drive/MyDrive/CrisisMMD_v2.0",
        "/content/drive/MyDrive/datasets/CrisisMMD_v2.0",
        f"{DRIVE_DIR}/CrisisMMD_v2.0"
    ]
    for d_path in drive_datasets:
        if os.path.exists(d_path):
            DATASET = d_path
            print(f"  [SAFE MODE] Loading dataset from Google Drive: {DATASET}")
            break
else:
    print(f"  [WARNING] Google Drive not mounted.")
    print(f"            Saving locally to: {RESULTS_CSV} (Data may be lost if disconnected!)")

if DATASET == "/content/datasets/CrisisMMD_v2.0" and not os.path.exists(DATASET):
    print(f"  [WARNING] Dataset not found at {DATASET}. Please ensure it is downloaded.")

run_causal_all_experiments(
    dataset_path=DATASET,
    seeds=(42, 123, 456, 789, 1024),
    tasks=("task1", "task2", "task3"),
    few_shot_sizes=(50, 100, 250, 500),
    device="cuda",
    results_csv=RESULTS_CSV,
    variant_name="causal_full",
    use_causal=True,
    use_intervention=True,
    use_causal_graph=True,
)


# ============================================================
# PHASE 10: COMPARISON - GEDA vs CausalCrisis
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

print("\n" + "="*70)
print("  PHASE 10: GEDA vs CausalCrisis COMPARISON")
print("="*70)

# Load results
geda_csv = "/content/geda_results/all_results.csv"
if not os.path.exists(geda_csv) and os.path.exists("/content/drive/MyDrive"):
    # Try finding in Drive
    drive_geda = "/content/drive/MyDrive/CrisisSummarization/geda_results/all_results.csv"
    if os.path.exists(drive_geda):
        geda_csv = drive_geda
        print(f"  [SAFE MODE] Loading GEDA results from Google Drive: {geda_csv}")

causal_csv = RESULTS_CSV

dfs = {}
if os.path.exists(geda_csv):
    dfs["geda_full"] = pd.read_csv(geda_csv)
    print(f"  GEDA: {len(dfs['geda_full'])} runs loaded")
else:
    print(f"  [ERROR] GEDA baseline results NOT FOUND at: {geda_csv}")

if os.path.exists(causal_csv):
    dfs["causal_full"] = pd.read_csv(causal_csv)
    print(f"  CausalCrisis: {len(dfs['causal_full'])} runs loaded")
else:
    print(f"  [ERROR] CausalCrisis results NOT FOUND at: {causal_csv}")

if len(dfs) < 2:
    print("  [WARN] Can ca 2 file results de so sanh!")
else:
    df_all = pd.concat(dfs.values(), ignore_index=True)

    task_names = {"task1": "Informativeness", "task2": "Humanitarian", "task3": "Damage"}

    print("\n" + "="*70)
    print("  COMPARISON TABLE: Weighted F1 (mean ± std)")
    print("="*70)

    for task in ["task1", "task2", "task3"]:
        print(f"\n--- {task_names[task]} ---")
        print(f"{'Model':35s} {'N=50':>10s} {'N=100':>10s} {'N=250':>10s} {'N=500':>10s}")
        print("-" * 70)

        for model_name in ["geda_full", "causal_full"]:
            row = f"{model_name:35s}"
            for n in [50, 100, 250, 500]:
                mask = (
                    (df_all['model'] == model_name) &
                    (df_all['task'] == task) &
                    (df_all['few_shot'] == n)
                )
                vals = df_all.loc[mask, 'weighted_f1'].values * 100
                if len(vals) > 0:
                    row += f"  {vals.mean():.1f}±{vals.std():.1f}"
                else:
                    row += f"       N/A"
            print(row)

    # Significance tests
    print("\n" + "="*70)
    print("  SIGNIFICANCE TESTS: CausalCrisis vs GEDA")
    print("="*70)

    for task in ["task1", "task2", "task3"]:
        print(f"\n--- {task_names[task]} ---")
        for n in [50, 100, 250, 500]:
            geda_vals = df_all.loc[
                (df_all['model'] == 'geda_full') &
                (df_all['task'] == task) &
                (df_all['few_shot'] == n),
                'weighted_f1'
            ].values

            causal_vals = df_all.loc[
                (df_all['model'] == 'causal_full') &
                (df_all['task'] == task) &
                (df_all['few_shot'] == n),
                'weighted_f1'
            ].values

            if len(geda_vals) >= 2 and len(causal_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(causal_vals, geda_vals)
                diff = (causal_vals.mean() - geda_vals.mean()) * 100
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  N={n:3d}: diff={diff:+.1f}% p={p_val:.4f} {sig}")

    # Domain invariance analysis
    if 'domain_acc_causal' in df_all.columns:
        print("\n" + "="*70)
        print("  DOMAIN INVARIANCE CHECK")
        print("  (Ly tuong: causal_domain_acc ≈ chance = 1/7 ≈ 14.3%)")
        print("="*70)

        causal_runs = df_all[df_all['model'] == 'causal_full']
        if 'domain_acc_causal' in causal_runs.columns:
            for task in ["task1", "task2", "task3"]:
                vals = causal_runs.loc[
                    causal_runs['task'] == task,
                    'domain_acc_causal'
                ].values
                vals = vals[vals >= 0]  # filter -1 (unavailable)
                if len(vals) > 0:
                    print(f"  {task_names[task]:20s}: "
                          f"domain_acc = {vals.mean()*100:.1f}% ± {vals.std()*100:.1f}%")

print("\n[DONE] Phase 10 complete!")
