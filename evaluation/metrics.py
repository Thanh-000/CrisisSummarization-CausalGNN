"""
GEDA Evaluation Framework — Statistical Metrics
=================================================
Module tính toán thống kê: CI, paired t-test, Bonferroni, effect size.
Được thiết kế để đảm bảo statistical rigor theo MT5.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional


# ============================================================
# Confidence Intervals
# ============================================================

def confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Tính confidence interval cho một tập scores.
    
    Args:
        scores: Danh sách F1 scores từ nhiều seeds
        confidence: Mức tin cậy (default 0.95 cho 95% CI)
    
    Returns:
        (mean, ci_lower, ci_upper)
    
    Example:
        >>> scores = [0.72, 0.75, 0.73, 0.74, 0.71]
        >>> mean, lo, hi = confidence_interval(scores)
        >>> print(f"{mean:.4f} [{lo:.4f}, {hi:.4f}]")
        0.7300 [0.7119, 0.7481]
    """
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    
    if n < 2:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    
    mean = float(np.mean(arr))
    se = float(stats.sem(arr))  # standard error of mean
    
    # t-distribution vì n nhỏ (thường 5 seeds)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    
    return (mean, mean - margin, mean + margin)


def format_ci(
    scores: List[float], 
    confidence: float = 0.95,
    decimals: int = 2,
) -> str:
    """
    Format kết quả với CI dạng: "73.00 ± 1.81 [71.19, 74.81]"
    
    Dùng trong báo cáo và bảng kết quả.
    """
    mean, lo, hi = confidence_interval(scores, confidence)
    margin = (hi - lo) / 2
    
    pct = 100  # convert to percentage
    return (
        f"{mean*pct:.{decimals}f} ± {margin*pct:.{decimals}f} "
        f"[{lo*pct:.{decimals}f}, {hi*pct:.{decimals}f}]"
    )


# ============================================================
# Statistical Significance Tests
# ============================================================

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Paired t-test giữa 2 models trên cùng seeds.
    
    Args:
        scores_a: F1 scores model A (per seed)
        scores_b: F1 scores model B (per seed)
        alpha: Significance level
    
    Returns:
        Dict với t_statistic, p_value, significant, mean_diff, effect_size
    
    Example:
        >>> gnn_scores  = [0.72, 0.75, 0.73, 0.74, 0.71]
        >>> geda_scores = [0.78, 0.80, 0.79, 0.77, 0.76]
        >>> result = paired_ttest(gnn_scores, geda_scores)
        >>> print(f"p={result['p_value']:.4f}, significant={result['significant']}")
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    
    assert len(a) == len(b), f"Số seeds phải bằng nhau: {len(a)} vs {len(b)}"
    
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Cohen's d cho paired samples
    diffs = b - a
    effect_size = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else 0.0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)),
        "effect_size_cohens_d": effect_size,
        "alpha": alpha,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, float]]:
    """
    Bonferroni correction cho multiple comparisons.
    
    Khi so sánh k cặp models, adjusted alpha = alpha / k
    
    Args:
        p_values: Danh sách p-values từ các paired t-tests
        alpha: Family-wise alpha (default 0.05)
    
    Returns:
        List[Dict] với adjusted_alpha, original_p, significant
    
    Example:
        >>> # So sánh 3 cặp: GNN vs GEDA, DiffAttn vs GEDA, GNN vs DiffAttn
        >>> p_vals = [0.012, 0.008, 0.340]
        >>> results = bonferroni_correction(p_vals)
        >>> for r in results:
        ...     print(f"p={r['original_p']:.3f} -> sig={r['significant']}")
    """
    k = len(p_values)
    adjusted_alpha = alpha / k
    
    results = []
    for p in p_values:
        results.append({
            "original_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
            "num_comparisons": k,
        })
    
    return results


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative).
    Dùng khi n nhỏ và không chắc normality assumption.
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    
    try:
        stat, p_value = stats.wilcoxon(a, b, alternative='two-sided')
    except ValueError:
        # Nếu tất cả differences = 0
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }
    
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
    }


# ============================================================
# Effect Size Interpretation
# ============================================================

def interpret_cohens_d(d: float) -> str:
    """
    Giải thích Cohen's d effect size.
    |d| < 0.2: negligible
    0.2 ≤ |d| < 0.5: small
    0.5 ≤ |d| < 0.8: medium
    |d| ≥ 0.8: large
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================
# Comparison Summary
# ============================================================

def full_comparison(
    model_a_name: str,
    model_b_name: str,
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    So sánh đầy đủ giữa 2 models: CI, t-test, effect size.
    
    Returns:
        Dict summary đủ thông tin để báo cáo.
    """
    ci_a = confidence_interval(scores_a)
    ci_b = confidence_interval(scores_b)
    ttest = paired_ttest(scores_a, scores_b, alpha)
    
    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "model_a_mean_ci": f"{ci_a[0]*100:.2f} [{ci_a[1]*100:.2f}, {ci_a[2]*100:.2f}]",
        "model_b_mean_ci": f"{ci_b[0]*100:.2f} [{ci_b[1]*100:.2f}, {ci_b[2]*100:.2f}]",
        "improvement": f"{ttest['mean_diff']*100:+.2f}%",
        "p_value": ttest["p_value"],
        "significant": ttest["significant"],
        "effect_size": ttest["effect_size_cohens_d"],
        "effect_interpretation": interpret_cohens_d(ttest["effect_size_cohens_d"]),
    }


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GEDA Statistical Metrics -- Demo")
    print("=" * 60)
    
    # Simulated results
    gnn_scores = [0.72, 0.75, 0.73, 0.74, 0.71]
    diffattn_scores = [0.74, 0.76, 0.75, 0.73, 0.72]
    geda_scores = [0.78, 0.80, 0.79, 0.77, 0.76]
    
    print("\n--- Confidence Intervals ---")
    print(f"GNN:      {format_ci(gnn_scores)}")
    print(f"DiffAttn: {format_ci(diffattn_scores)}")
    print(f"GEDA:     {format_ci(geda_scores)}")
    
    print("\n--- Pairwise Comparisons ---")
    comparisons = [
        ("GNN", "GEDA", gnn_scores, geda_scores),
        ("DiffAttn", "GEDA", diffattn_scores, geda_scores),
        ("GNN", "DiffAttn", gnn_scores, diffattn_scores),
    ]
    
    p_values = []
    for name_a, name_b, sa, sb in comparisons:
        result = full_comparison(name_a, name_b, sa, sb)
        p_values.append(result["p_value"])
        print(f"\n{name_a} vs {name_b}:")
        print(f"  {name_a}: {result['model_a_mean_ci']}")
        print(f"  {name_b}: {result['model_b_mean_ci']}")
        print(f"  Improvement: {result['improvement']}")
        print(f"  p-value: {result['p_value']:.4f} (sig: {result['significant']})")
        print(f"  Effect size: {result['effect_size']:.3f} ({result['effect_interpretation']})")
    
    print("\n--- Bonferroni Correction ---")
    bonf = bonferroni_correction(p_values)
    for i, (comp, b) in enumerate(zip(comparisons, bonf)):
        name_a, name_b = comp[0], comp[1]
        print(f"  {name_a} vs {name_b}: p={b['original_p']:.4f}, "
              f"adj_alpha={b['adjusted_alpha']:.4f}, sig={b['significant']}")
    
    print("\nDone!")
