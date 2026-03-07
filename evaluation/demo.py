"""
GEDA Evaluation Framework — Demo Script
=========================================
Chạy demo với dữ liệu mô phỏng để kiểm tra framework hoạt động đúng.

Usage:
    python -m evaluation.demo
    
    hoặc trong Google Colab:
    %run evaluation/demo.py
"""

import sys
import os
from pathlib import Path

# Đảm bảo import được từ thư mục gốc
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import (
    ExperimentResult,
    TASKS,
    FEW_SHOT_SETTINGS,
    RANDOM_SEEDS,
    MODELS,
    ProjectPaths,
)
from evaluation.metrics import (
    confidence_interval,
    format_ci,
    paired_ttest,
    bonferroni_correction,
    full_comparison,
)
from evaluation.results_manager import (
    save_experiment_result,
    load_results,
    get_scores,
    generate_all_templates,
)
from evaluation.report_generator import (
    print_comparison_report,
    generate_markdown_table,
    generate_latex_table,
)

import random
import numpy as np


def generate_simulated_data(paths: ProjectPaths) -> None:
    """
    Tạo dữ liệu mô phỏng để demo framework.
    Simulates realistic F1 scores cho 3 models × 3 tasks × 4 settings × 5 seeds.
    """
    print("[DATA] Generating simulated experiment data...")
    
    # Dựa trên kết quả thực tế từ papers
    # Paper 1 (GNN): ~70-85% F1 tùy task/setting
    # Paper 2 (DiffAttn): ~72-87% F1
    # GEDA (expected): ~75-89% F1
    
    base_scores = {
        "paper1_gnn": {
            "task1": {50: 0.72, 100: 0.76, 250: 0.80, 500: 0.83},
            "task2": {50: 0.45, 100: 0.52, 250: 0.58, 500: 0.62},
            "task3": {50: 0.50, 100: 0.55, 250: 0.60, 500: 0.65},
        },
        "paper2_diffattn": {
            "task1": {50: 0.74, 100: 0.78, 250: 0.82, 500: 0.85},
            "task2": {50: 0.47, 100: 0.53, 250: 0.59, 500: 0.63},
            "task3": {50: 0.48, 100: 0.54, 250: 0.60, 500: 0.64},
        },
        "geda": {
            "task1": {50: 0.77, 100: 0.81, 250: 0.85, 500: 0.88},
            "task2": {50: 0.52, 100: 0.58, 250: 0.64, 500: 0.68},
            "task3": {50: 0.55, 100: 0.60, 250: 0.66, 500: 0.70},
        },
    }
    
    np.random.seed(42)
    
    for model, tasks in base_scores.items():
        for task, settings in tasks.items():
            for fs, base_f1 in settings.items():
                for seed in RANDOM_SEEDS:
                    # Thêm random noise ±2%
                    noise = np.random.normal(0, 0.02)
                    f1 = max(0.1, min(0.99, base_f1 + noise))
                    
                    result = ExperimentResult(
                        model=model,
                        task=task,
                        few_shot=fs,
                        seed=seed,
                        accuracy=f1 + np.random.uniform(0.02, 0.05),
                        macro_f1=f1,
                        weighted_f1=f1 + np.random.uniform(0.01, 0.03),
                        train_time_seconds=random.uniform(60, 300),
                        inference_time_seconds=random.uniform(2, 15),
                        notes="simulated",
                    )
                    
                    save_experiment_result(result, paths.baseline_csv)
    
    print(f"  [OK] Saved to: {paths.baseline_csv}")


def run_demo():
    """Chạy demo đầy đủ."""
    
    print("=" * 60)
    print("  [LAB] GEDA Evaluation Framework -- Demo")
    print("=" * 60)
    
    # 1. Setup paths
    paths = ProjectPaths()
    paths.ensure_dirs()
    
    # 2. Generate templates
    print("\n[DIR] Step 1: Generate CSV templates")
    generate_all_templates(paths)
    
    # 3. Generate simulated data
    print("\n[DATA] Step 2: Generate simulated data")
    generate_simulated_data(paths)
    
    # 4. Load and analyze
    print("\n[CHART] Step 3: Load and analyze results")
    results = load_results(paths.baseline_csv)
    print(f"  Loaded {len(results)} experiment results")
    
    # 5. Statistical analysis
    print("\n[STATS] Step 4: Statistical analysis")
    
    for task in TASKS:
        print_comparison_report(results, task, few_shot=100)
    
    # 6. Markdown table
    print("\n\n[MD] Step 5: Generate Markdown table")
    md = generate_markdown_table(results, "task1")
    print(md)
    
    # 7. LaTeX table
    print("\n[FILE] Step 6: Generate LaTeX table")
    latex = generate_latex_table(results, "task1")
    print(latex)
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("  [OK] DEMO COMPLETE")
    print("=" * 60)
    print(f"\n  Files created:")
    print(f"    [DIR] {paths.results_dir}/")
    for f in paths.results_dir.glob("*.csv"):
        size = f.stat().st_size
        print(f"    [FILE] {f.name} ({size:,} bytes)")
    
    print(f"\n  Usage in your code:")
    print(f"    from evaluation.metrics import confidence_interval, paired_ttest")
    print(f"    from evaluation.results_manager import save_experiment_result, load_results")
    print(f"    from evaluation.report_generator import print_comparison_report")


if __name__ == "__main__":
    run_demo()
