"""
GEDA Evaluation Framework — Report Generator
==============================================
Tạo báo cáo tự động từ kết quả thực nghiệm.
Output: Markdown tables, LaTeX tables, console summary.
"""

from typing import List, Dict, Optional
from pathlib import Path

from .config import TASKS, FEW_SHOT_SETTINGS, MODELS
from .metrics import confidence_interval, format_ci, paired_ttest, bonferroni_correction, interpret_cohens_d
from .results_manager import load_results, get_scores


# ============================================================
# Console Report
# ============================================================

def print_comparison_report(
    results: List[Dict],
    task: str,
    few_shot: int,
    models: Optional[List[str]] = None,
) -> None:
    """
    In bảng so sánh cho một task + few_shot setting cụ thể.
    """
    if models is None:
        models = list(MODELS.keys())
    
    task_name = TASKS.get(task, {}).get("name", task)
    
    print(f"\n{'='*60}")
    print(f"  {task_name} -- {few_shot} labeled samples")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Macro F1':>25}")
    print(f"  {'-'*25} {'-'*25}")
    
    model_scores = {}
    for model in models:
        scores = get_scores(results, model, task, few_shot)
        if scores:
            ci_str = format_ci(scores)
            print(f"  {MODELS.get(model, model):<25} {ci_str:>25}")
            model_scores[model] = scores
        else:
            print(f"  {MODELS.get(model, model):<25} {'(no data)':>25}")
    
    # Pairwise comparisons
    model_list = [m for m in models if m in model_scores and len(model_scores[m]) >= 2]
    if len(model_list) >= 2:
        print(f"\n  --- Pairwise Comparisons ---")
        p_values = []
        pairs = []
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                ma, mb = model_list[i], model_list[j]
                result = paired_ttest(model_scores[ma], model_scores[mb])
                p_values.append(result["p_value"])
                pairs.append((ma, mb, result))
        
        bonf = bonferroni_correction(p_values)
        
        for (ma, mb, result), b in zip(pairs, bonf):
            name_a = MODELS.get(ma, ma)[:15]
            name_b = MODELS.get(mb, mb)[:15]
            sig = "YES" if b["significant"] else "NO"
            effect = interpret_cohens_d(result["effect_size_cohens_d"])
            print(
                f"  {name_a} vs {name_b}: "
                f"Diff={result['mean_diff']*100:+.2f}%, "
                f"p={result['p_value']:.4f}, "
                f"sig(Bonf)={sig}, "
                f"d={result['effect_size_cohens_d']:.2f} ({effect})"
            )


def print_full_report(results: List[Dict]) -> None:
    """In báo cáo đầy đủ cho tất cả tasks và settings."""
    print("\n" + "=" * 60)
    print("  GEDA EVALUATION REPORT")
    print("  " + "=" * 56)
    
    for task in TASKS:
        for fs in FEW_SHOT_SETTINGS:
            scores = get_scores(results, list(MODELS.keys())[0], task, fs)
            if scores:
                print_comparison_report(results, task, fs)


# ============================================================
# Markdown Report
# ============================================================

def generate_markdown_table(
    results: List[Dict],
    task: str,
    models: Optional[List[str]] = None,
) -> str:
    """
    Tạo bảng Markdown so sánh models qua các few-shot settings.
    
    Returns:
        Markdown string
    """
    if models is None:
        models = list(MODELS.keys())
    
    task_name = TASKS.get(task, {}).get("name", task)
    
    lines = [
        f"### {task_name}",
        "",
        "| Model | " + " | ".join([f"{fs} shots" for fs in FEW_SHOT_SETTINGS]) + " |",
        "|" + "---|" * (len(FEW_SHOT_SETTINGS) + 1),
    ]
    
    for model in models:
        model_name = MODELS.get(model, model)
        cells = [f"**{model_name}**"]
        
        for fs in FEW_SHOT_SETTINGS:
            scores = get_scores(results, model, task, fs)
            if scores:
                mean, lo, hi = confidence_interval(scores)
                cells.append(f"{mean*100:.1f} ± {(hi-lo)/2*100:.1f}")
            else:
                cells.append("--")
        
        lines.append("| " + " | ".join(cells) + " |")
    
    lines.append("")
    return "\n".join(lines)


def generate_full_markdown_report(
    results: List[Dict],
    output_path: Optional[Path] = None,
) -> str:
    """Tạo báo cáo Markdown đầy đủ."""
    sections = [
        "# GEDA Evaluation Report\n",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Results by Task\n",
    ]
    
    for task in TASKS:
        sections.append(generate_markdown_table(results, task))
    
    report = "\n".join(sections)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[OK] Report saved: {output_path}")
    
    return report


# ============================================================
# LaTeX Table
# ============================================================

def generate_latex_table(
    results: List[Dict],
    task: str,
    models: Optional[List[str]] = None,
) -> str:
    """
    Tạo bảng LaTeX cho paper.
    Bold best result cho mỗi column.
    """
    if models is None:
        models = list(MODELS.keys())
    
    task_name = TASKS.get(task, {}).get("name", task)
    
    ncols = len(FEW_SHOT_SETTINGS)
    col_spec = "l" + "c" * ncols
    
    lines = [
        f"% Table: {task_name}",
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{task_name} — Macro F1 (\\%) across few-shot settings}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "\\textbf{Model} & " + " & ".join([f"\\textbf{{{fs}}}" for fs in FEW_SHOT_SETTINGS]) + " \\\\",
        "\\midrule",
    ]
    
    # Tìm best cho mỗi column
    best_per_col = {}
    for fs in FEW_SHOT_SETTINGS:
        best_mean = -1
        for model in models:
            scores = get_scores(results, model, task, fs)
            if scores:
                mean = sum(scores) / len(scores)
                if mean > best_mean:
                    best_mean = mean
                    best_per_col[fs] = model
    
    for model in models:
        model_name = MODELS.get(model, model)
        cells = [model_name]
        
        for fs in FEW_SHOT_SETTINGS:
            scores = get_scores(results, model, task, fs)
            if scores:
                mean, lo, hi = confidence_interval(scores)
                margin = (hi - lo) / 2
                val = f"{mean*100:.1f} $\\pm$ {margin*100:.1f}"
                
                if best_per_col.get(fs) == model:
                    val = f"\\textbf{{{val}}}"
                
                cells.append(val)
            else:
                cells.append("--")
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)
