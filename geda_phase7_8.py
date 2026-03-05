"""
GEDA Phase 7+8: Run GEDA + 3-Model Comparison
================================================
Script nay duoc goi tu Colab notebook.
Phase 7: Train GEDA model
Phase 8: So sanh 3 models + statistical tests + plots

Usage trong Colab:
  !python /content/GEDA/geda_phase7_8.py \\
      --dataset_path /content/datasets/CrisisMMD_v2.0 \\
      --results_csv /content/geda_results/all_results.csv \\
      --output_dir /content/geda_results
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy import stats

# Ensure geda_model and geda_trainer are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geda_model import GEDAModel, GEDALoss, model_summary, ABLATION_CONFIGS
from geda_trainer import (
    run_geda_experiment,
    run_geda_all_experiments,
    set_seed,
)


# ============================================================
# PHASE 7: Run GEDA experiments
# ============================================================

def phase7_run_geda(args):
    """Chay tat ca GEDA experiments."""
    print("\n" + "=" * 60)
    print("  PHASE 7: GEDA Model Training")
    print("=" * 60)

    # Model info
    model = GEDAModel()
    info = model_summary(model)
    print(f"  Architecture: GNN -> Self-Attn -> GCA -> Adaptive DiffAttn -> MTL")
    print(f"  Trainable params: {info['trainable_params']:,}")
    del model

    seeds = tuple(int(s) for s in args.seeds.split(","))
    tasks = tuple(args.tasks.split(","))
    sizes = tuple(int(s) for s in args.few_shot_sizes.split(","))

    total = len(seeds) * len(tasks) * len(sizes)
    print(f"  Experiments: {len(seeds)} seeds x {len(tasks)} tasks x {len(sizes)} sizes = {total} runs")
    print(f"  Device: {args.device}")

    run_geda_all_experiments(
        dataset_path=args.dataset_path,
        seeds=seeds,
        tasks=tasks,
        few_shot_sizes=sizes,
        device=args.device,
        results_csv=args.results_csv,
    )

    print("\n  Phase 7 complete!")


# ============================================================
# PHASE 8: 3-Model Comparison
# ============================================================

def phase8_compare(args):
    """So sanh 3 models voi statistical tests."""
    print("\n" + "=" * 60)
    print("  PHASE 8: 3-Model Comparison")
    print("=" * 60)

    if not os.path.exists(args.results_csv):
        print(f"  ERROR: {args.results_csv} not found!")
        return

    df = pd.read_csv(args.results_csv)
    METRIC = "weighted_f1"
    TASKS = ["task1", "task2", "task3"]
    TASK_NAMES = {"task1": "Informativeness", "task2": "Humanitarian", "task3": "Damage"}
    MODELS = sorted(df["model"].unique())

    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)

    # ---- 8.1 Summary Table ----
    print("\n" + "-" * 60)
    print("  8.1: Performance Summary")
    print("-" * 60)

    summary_rows = []
    for task_id in TASKS:
        print(f"\n  --- {TASK_NAMES[task_id]} ---")
        print(f"  {'Model':<25} {'wF1':>12} {'bAcc':>10} {'N':>5}")
        print("  " + "-" * 55)

        for m in MODELS:
            m_df = df[(df["model"] == m) & (df["task"] == task_id)]
            if len(m_df) == 0:
                continue

            wf1 = m_df[METRIC].values
            bacc = m_df["balanced_accuracy"].values if "balanced_accuracy" in m_df else []

            if len(wf1) > 1:
                wf1_str = f"{np.mean(wf1)*100:.1f} +/- {np.std(wf1, ddof=1)*100:.1f}"
            else:
                wf1_str = f"{np.mean(wf1)*100:.1f}"

            bacc_str = f"{np.mean(bacc)*100:.1f}" if len(bacc) > 0 else "-"
            print(f"  {m:<25} {wf1_str:>12} {bacc_str:>10} {len(wf1):>5}")

            summary_rows.append({
                "task": task_id,
                "model": m,
                "wf1_mean": np.mean(wf1),
                "wf1_std": np.std(wf1, ddof=1) if len(wf1) > 1 else 0,
                "bacc_mean": np.mean(bacc) if len(bacc) > 0 else None,
                "n_runs": len(wf1),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{args.output_dir}/summary_3models.csv", index=False)
    print(f"\n  Saved: {args.output_dir}/summary_3models.csv")

    # ---- 8.2 Significance Tests ----
    print("\n" + "-" * 60)
    print("  8.2: Statistical Significance (GEDA vs Baselines)")
    print("-" * 60)

    sig_rows = []
    for task_id in TASKS:
        geda = df[(df["model"] == "geda_full") & (df["task"] == task_id)][METRIC].values
        if len(geda) < 2:
            print(f"\n  {TASK_NAMES[task_id]}: Not enough GEDA runs (need >= 2)")
            continue

        print(f"\n  --- {TASK_NAMES[task_id]} ---")
        baselines = [m for m in MODELS if m != "geda_full"]

        for bl in baselines:
            bl_scores = df[(df["model"] == bl) & (df["task"] == task_id)][METRIC].values
            if len(bl_scores) < 2:
                continue

            # Independent t-test (unequal N possible)
            t_stat, p_val = stats.ttest_ind(geda, bl_scores, equal_var=False)

            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(geda, ddof=1) + np.var(bl_scores, ddof=1)) / 2
            )
            cohens_d = (np.mean(geda) - np.mean(bl_scores)) / (pooled_std + 1e-8)

            # Bonferroni
            n_tests = len(baselines) * len(TASKS)
            alpha_bonf = 0.05 / n_tests
            sig = "YES***" if p_val < alpha_bonf else ("yes*" if p_val < 0.05 else "no")

            delta_pp = (np.mean(geda) - np.mean(bl_scores)) * 100

            print(
                f"  GEDA vs {bl:<20} "
                f"delta={delta_pp:+.2f}pp  p={p_val:.4f}  d={cohens_d:.2f}  sig={sig}"
            )

            sig_rows.append({
                "task": task_id,
                "comparison": f"GEDA vs {bl}",
                "delta_pp": round(delta_pp, 2),
                "p_value": round(p_val, 4),
                "cohens_d": round(cohens_d, 2),
                "significant": sig,
            })

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(f"{args.output_dir}/significance_tests.csv", index=False)
        print(f"\n  Saved: {args.output_dir}/significance_tests.csv")

    # ---- 8.3 Comparison Plot ----
    print("\n" + "-" * 60)
    print("  8.3: Generating Comparison Plots")
    print("-" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FEW_SHOTS = [50, 100, 250, 500]
        COLORS = {
            "paper1_gnn": "#2196F3",
            "paper2_diffattn": "#FF9800",
            "geda_full": "#4CAF50",
        }
        LABELS = {
            "paper1_gnn": "Paper 1 (GNN)",
            "paper2_diffattn": "Paper 2 (DiffAttn)",
            "geda_full": "GEDA (Ours)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for ax, task_id in zip(axes, TASKS):
            task_df = df[df["task"] == task_id]
            for model_name in ["paper1_gnn", "paper2_diffattn", "geda_full"]:
                m_df = task_df[task_df["model"] == model_name]
                if len(m_df) == 0:
                    continue

                means, stds, valid_sizes = [], [], []
                for fs in FEW_SHOTS:
                    scores = m_df[m_df["few_shot"] == fs][METRIC].values
                    if len(scores) > 0:
                        means.append(np.mean(scores) * 100)
                        stds.append(np.std(scores, ddof=1) * 100 if len(scores) > 1 else 0)
                        valid_sizes.append(fs)

                if means:
                    c = COLORS.get(model_name, "#999")
                    lbl = LABELS.get(model_name, model_name)
                    ax.errorbar(
                        valid_sizes, means, yerr=stds,
                        marker="o", label=lbl, color=c,
                        linewidth=2, markersize=6, capsize=3,
                    )

            ax.set_title(TASK_NAMES[task_id], fontsize=13, fontweight="bold")
            ax.set_xlabel("Few-shot size")
            ax.set_xscale("log")
            ax.set_xticks(FEW_SHOTS)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Weighted F1 (%)")
        axes[0].legend(loc="lower right")
        fig.suptitle(
            "GEDA vs Baselines — Few-shot Performance",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        fig_path = f"{args.output_dir}/figures/3model_comparison.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fig_path}")
        plt.close()

    except ImportError as e:
        print(f"  Matplotlib not available: {e}")

    # ---- 8.4 LaTeX Table ----
    print("\n" + "-" * 60)
    print("  8.4: LaTeX Tables")
    print("-" * 60)

    latex_lines = []
    latex_lines.append("% === LaTeX: 3-Model Comparison ===")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Weighted F1 (\\%) — 3-Model Comparison on CrisisMMD}")

    FEW_SHOTS = [50, 100, 250, 500]
    MODEL_ORDER = ["paper1_gnn", "paper2_diffattn", "geda_full"]
    MODEL_LATEX = {
        "paper1_gnn": "Paper 1 (GNN)",
        "paper2_diffattn": "Paper 2 (DiffAttn)",
        "geda_full": "\\textbf{GEDA (Ours)}",
    }
    TASK_LATEX = {"task1": "Inform.", "task2": "Human.", "task3": "Damage"}

    n_cols = 1 + len(FEW_SHOTS)
    col_spec = "l" + " c" * len(FEW_SHOTS)

    for task_id in TASKS:
        latex_lines.append(f"% --- {TASK_LATEX[task_id]} ---")
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")
        header = "\\textbf{Model}"
        for fs in FEW_SHOTS:
            header += f" & \\textbf{{{fs}}}"
        latex_lines.append(header + " \\\\")
        latex_lines.append("\\hline")

        for m in MODEL_ORDER:
            row = MODEL_LATEX.get(m, m)
            for fs in FEW_SHOTS:
                scores = df[
                    (df["model"] == m) & (df["task"] == task_id) & (df["few_shot"] == fs)
                ][METRIC].values
                if len(scores) > 1:
                    cell = f"{np.mean(scores)*100:.1f} $\\pm$ {np.std(scores,ddof=1)*100:.1f}"
                elif len(scores) == 1:
                    cell = f"{scores[0]*100:.1f}"
                else:
                    cell = "-"
                row += f" & {cell}"
            latex_lines.append(row + " \\\\")

        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("")

    latex_lines.append("\\end{table}")

    latex_output = "\n".join(latex_lines)
    print(latex_output)

    latex_path = f"{args.output_dir}/latex_3model_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_output)
    print(f"\n  Saved: {latex_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GEDA Phase 7+8")
    parser.add_argument("--dataset_path", default="/content/datasets/CrisisMMD_v2.0")
    parser.add_argument("--results_csv", default="/content/geda_results/all_results.csv")
    parser.add_argument("--output_dir", default="/content/geda_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--tasks", default="task1,task2,task3")
    parser.add_argument("--few_shot_sizes", default="50,100,250,500")
    parser.add_argument("--phase", choices=["7", "8", "both"], default="both",
                        help="Which phase to run: 7=train, 8=compare, both=all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase in ("7", "both"):
        phase7_run_geda(args)

    if args.phase in ("8", "both"):
        phase8_compare(args)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
