# ============================================================================
# CausalCrisis V3 — Evaluation Module
# Metrics, LODO, Ablation, Statistical Testing, Visualization
# ============================================================================

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Comprehensive Metrics
# ============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute full set of classification metrics."""
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    
    metrics = {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(per_class_f1):
        metrics[f"f1_class_{i}"] = f1
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred)
    
    return metrics


# ============================================================================
# Bootstrap Significance Test
# ============================================================================
def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 10000,
    metric_fn=None,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test: Model A vs Model B.
    H0: No difference in performance.
    
    Returns p-value and confidence interval.
    """
    from sklearn.metrics import f1_score
    
    if metric_fn is None:
        metric_fn = lambda y, p: f1_score(y, p, average="weighted")
    
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    # Observed difference
    score_a = metric_fn(y_true, y_pred_a)
    score_b = metric_fn(y_true, y_pred_b)
    observed_diff = score_a - score_b
    
    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])
        diffs.append(diff)
    
    diffs = np.array(diffs)
    
    # P-value (two-tailed)
    p_value = (np.abs(diffs) >= np.abs(observed_diff)).mean()
    
    # Confidence interval
    ci_lower = np.percentile(diffs, alpha/2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha/2) * 100)
    
    return {
        "score_a": score_a,
        "score_b": score_b,
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < alpha,
    }


# ============================================================================
# LODO Evaluation
# ============================================================================
def run_lodo_evaluation(
    model_factory,  # Callable → returns (model, optimizer, scheduler)
    loss_fn_factory,  # Callable → returns loss function
    image_features: np.ndarray,
    text_features: np.ndarray,
    labels: np.ndarray,
    domain_ids: np.ndarray,
    event_names: List[str],
    train_fn,  # train_fn(model, loss_fn, train_loader, val_loader, ...) → metrics
    epochs: int = 80,
    batch_size: int = 128,
    seeds: List[int] = None,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Leave-One-Disaster-Out evaluation.
    Train trên N-1 disasters, test trên 1.
    """
    from .data import create_lodo_splits, CrisisMMDDataset, create_stratified_splits
    from torch.utils.data import DataLoader
    
    if seeds is None:
        seeds = [42, 123, 456]
    
    unique_domains = sorted(np.unique(domain_ids))
    results = defaultdict(list)
    
    print(f"\n{'='*60}")
    print(f"🌍 LODO Evaluation — {len(unique_domains)} folds × {len(seeds)} seeds")
    print(f"{'='*60}")
    
    for domain_id in unique_domains:
        domain_name = event_names[domain_id] if domain_id < len(event_names) else f"Domain_{domain_id}"
        
        for seed in seeds:
            print(f"\n--- LODO: hold-out={domain_name}, seed={seed} ---")
            
            # Get LODO split
            train_idx, test_idx = create_lodo_splits(domain_ids, domain_id)
            
            # Create val split from training data
            train_labels = labels[train_idx]
            train_domains = domain_ids[train_idx]
            sub_train_idx, sub_val_idx, _ = create_stratified_splits(
                train_labels, train_domains,
                test_ratio=0.001,  # tiny, just for splitting
                val_ratio=0.15,
                seed=seed,
            )
            actual_train = train_idx[sub_train_idx]
            actual_val = train_idx[sub_val_idx]
            
            # DataLoaders
            train_ds = CrisisMMDDataset(
                image_features, text_features, labels, domain_ids, actual_train
            )
            val_ds = CrisisMMDDataset(
                image_features, text_features, labels, domain_ids, actual_val
            )
            test_ds = CrisisMMDDataset(
                image_features, text_features, labels, domain_ids, test_idx
            )
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
            test_loader = DataLoader(test_ds, batch_size=batch_size)
            
            # Build model
            torch.manual_seed(seed)
            model, optimizer, scheduler = model_factory()
            loss_fn = loss_fn_factory()
            
            # Train
            history = train_fn(
                model, loss_fn, optimizer, scheduler,
                train_loader, val_loader,
                epochs=epochs, device=device,
            )
            
            # Evaluate on hold-out disaster
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    f_v = batch["image_features"].to(device)
                    f_t = batch["text_features"].to(device)
                    
                    output = model(f_v, f_t, use_ba=True)
                    preds = output["logits"].argmax(-1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(batch["label"].numpy())
            
            metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
            
            results[domain_name].append(metrics["f1_weighted"])
            print(f"   → F1={metrics['f1_weighted']:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 LODO Summary")
    print(f"{'='*60}")
    
    all_f1s = []
    for domain_name, f1_list in results.items():
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        all_f1s.extend(f1_list)
        print(f"   {domain_name}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    
    overall_mean = np.mean(all_f1s)
    overall_std = np.std(all_f1s)
    print(f"\n   OVERALL: F1 = {overall_mean:.4f} ± {overall_std:.4f}")
    
    return {
        "per_domain": dict(results),
        "overall_mean": overall_mean,
        "overall_std": overall_std,
    }


# ============================================================================
# Ablation Study Runner
# ============================================================================
def run_ablation_study(
    variants: Dict[str, dict],  # name → config overrides
    base_train_fn,  # Function that trains and returns metrics
    seeds: List[int] = None,
) -> Dict[str, Dict]:
    """
    Run ablation study across variants.
    
    Args:
        variants: dict mapping variant name to config overrides
        base_train_fn: function(config, seed) → metrics dict
        seeds: list of random seeds
    
    Returns:
        dict mapping variant name to aggregated metrics
    """
    if seeds is None:
        seeds = [42, 123, 456]
    
    results = {}
    
    for name, config in variants.items():
        print(f"\n{'='*40}")
        print(f"⚗️ Ablation: {name}")
        print(f"{'='*40}")
        
        f1_scores = []
        for seed in seeds:
            metrics = base_train_fn(config, seed)
            f1_scores.append(metrics["f1_weighted"])
        
        results[name] = {
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "f1_scores": f1_scores,
        }
        
        print(f"   → F1 = {results[name]['f1_mean']:.4f} ± {results[name]['f1_std']:.4f}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"📊 Ablation Results")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'F1 Mean':>10} {'F1 Std':>10} {'Δ vs Full':>10}")
    print(f"{'-'*60}")
    
    full_f1 = results.get("full", {}).get("f1_mean", 0)
    for name, res in sorted(results.items(), key=lambda x: x[1]["f1_mean"], reverse=True):
        delta = res["f1_mean"] - full_f1
        sign = "+" if delta >= 0 else ""
        print(f"{name:<30} {res['f1_mean']:>10.4f} {res['f1_std']:>10.4f} {sign}{delta:>9.4f}")
    
    return results


# ============================================================================
# Visualization Utilities
# ============================================================================
def plot_training_curves(history: Dict, save_path: str = None):
    """Plot training loss and F1 curves."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(history["train_loss"], label="Train", alpha=0.8)
        axes[0].plot(history["val_loss"], label="Val", alpha=0.8)
        axes[0].set_title("Loss", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1
        axes[1].plot(history["train_f1"], label="Train", alpha=0.8)
        axes[1].plot(history["val_f1"], label="Val", alpha=0.8)
        axes[1].set_title("Weighted F1", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label="Target 90%")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # LR
        if "lr" in history:
            axes[2].plot(history["lr"], alpha=0.8, color="green")
            axes[2].set_title("Learning Rate", fontsize=14, fontweight="bold")
            axes[2].set_xlabel("Epoch")
            axes[2].set_yscale("log")
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib not available")


def plot_tsne_causal_features(
    C_v: np.ndarray,
    C_t: np.ndarray,
    labels: np.ndarray,
    domain_ids: np.ndarray,
    save_path: str = None,
):
    """t-SNE visualization trên causal features."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Concatenate causal features
        features = np.concatenate([C_v, C_t], axis=1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # By class
        scatter1 = axes[0].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=labels, cmap="Set1", alpha=0.6, s=10
        )
        axes[0].set_title("Causal Features (by Class)", fontsize=14, fontweight="bold")
        plt.colorbar(scatter1, ax=axes[0])
        
        # By domain
        scatter2 = axes[1].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=domain_ids, cmap="tab10", alpha=0.6, s=10
        )
        axes[1].set_title("Causal Features (by Domain)", fontsize=14, fontweight="bold")
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.suptitle("t-SNE: Causal features should cluster by class, NOT by domain",
                     fontsize=12, style="italic")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib/sklearn not available")
