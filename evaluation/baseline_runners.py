"""
GEDA Evaluation Framework -- Baseline Runners
===============================================
Tich hop code evaluation tu 2 papers goc vao framework thong nhat.

Paper 1 (GNN): jdnascim/mm-class-for-disaster-data-with-gnn
  - Original: get_f1() + validate_best_model() trong src/gnn/gnn_utils.py
  - Dung weighted F1, balanced accuracy, confusion matrix
  - Output: JSON file voi f1_test, bacc_test

Paper 2 (DiffAttn): Munia03/Multimodal_Crisis_Event
  - Original: trainer.predict() trong trainer.py
  - Dung accuracy, micro/macro/weighted F1, confusion matrix
  - Output: logging statements

Module nay wrap ca 2 de output vao CSV chuan cua framework.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# sklearn metrics -- giong het cach 2 papers dung
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from .config import ExperimentResult, TASKS, RANDOM_SEEDS


# ============================================================
# Shared Evaluation Core
# Hop nhat metrics tu ca 2 papers
# ============================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Tinh tat ca metrics tu CA 2 papers trong 1 ham duy nhat.

    Paper 1 dung:
      - f1_score(..., average='weighted')  [gnn_utils.py:get_f1()]
      - balanced_accuracy_score()          [gnn_utils.py:get_normalized_acc()]
      - confusion_matrix()                 [gnn_utils.py:eval_data()]

    Paper 2 dung:
      - accuracy (correct/total)           [trainer.py:predict()]
      - f1_score(..., average='micro')     [trainer.py:predict()]
      - f1_score(..., average='macro')     [trainer.py:predict()]
      - f1_score(..., average='weighted')  [trainer.py:predict()]
      - confusion_matrix()                [trainer.py:predict()]
      - classification_report()           [trainer.py:predict()]

    Returns:
        Dict voi tat ca metrics hop nhat
    """
    # Paper 2 style: accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Paper 1 style: balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # F1 scores (ca 2 papers deu dung, Paper 2 day du hon)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics (Paper 2 dung classification_report)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix (ca 2 papers)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# Paper 1 (GNN) Baseline Runner
# Reproduce tu: gnn_utils.py -> eval_data() + validate_best_model()
# ============================================================

class Paper1GNNEvaluator:
    """
    Wrapper cho Paper 1 evaluation logic.

    Original code flow (feature_fusion.py):
    1. Train GNN voi run_base_v2()
    2. Evaluate voi validate_best_model() -> goi eval_data(test=True)
    3. eval_data() tinh:
       - f1_labeled = get_f1(y[mask_labeled], pred_labeled)   # weighted F1
       - f1_test = get_f1(y[mask_test], pred_test)            # weighted F1
       - confm = confusion_matrix(y[mask_test], pred_test)
    4. Save results vao JSON file

    Key difference vs Paper 2:
    - Paper 1 dung weighted_f1 lam metric chinh (get_f1 dung average='weighted')
    - Paper 1 KHONG tinh accuracy hay macro_f1
    - Paper 1 dung balanced_accuracy qua get_normalized_acc()
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def evaluate_gnn(
        self,
        pyg_graph,
        task: str,
        few_shot: int,
        seed: int,
    ) -> ExperimentResult:
        """
        Evaluate GNN model -- reproduce Paper 1 logic + bo sung metrics.

        Args:
            pyg_graph: PyG Data object voi test_mask
            task: Task ID ('task1', 'task2', 'task3')
            few_shot: So luong labeled samples
            seed: Random seed

        Returns:
            ExperimentResult tuong thich voi framework
        """
        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            # Paper 1: logits = model(data.x, data.edge_index)
            logits = self.model(pyg_graph.x, pyg_graph.edge_index)

            mask_test = pyg_graph.test_mask
            pred_test = logits[mask_test].max(1)[1]
            y_true = pyg_graph.y[mask_test]

        inference_time = time.time() - start_time

        y_true_np = y_true.cpu().numpy()
        y_pred_np = pred_test.cpu().numpy()

        # Tinh TAT CA metrics (khong chi weighted F1 nhu paper goc)
        metrics = compute_all_metrics(y_true_np, y_pred_np)

        return ExperimentResult(
            model="paper1_gnn",
            task=task,
            few_shot=few_shot,
            seed=seed,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            precision_per_class=metrics["precision_per_class"],
            recall_per_class=metrics["recall_per_class"],
            f1_per_class=metrics["f1_per_class"],
            confusion_matrix=metrics["confusion_matrix"],
            inference_time_seconds=inference_time,
            notes=f"GNN baseline (balanced_acc={metrics['balanced_accuracy']:.4f})",
        )


# ============================================================
# Paper 2 (DiffAttn) Baseline Runner
# Reproduce tu: trainer.py -> predict()
# ============================================================

class Paper2DiffAttnEvaluator:
    """
    Wrapper cho Paper 2 evaluation logic.

    Original code flow (trainer.py -> predict()):
    1. model.eval()
    2. Loop qua test_loader:
       - logits = self.model(x)
       - indices = torch.argmax(logits, dim=1)
       - correct += sum(indices == y)
       - all_predictions.extend(indices)
       - all_labels.extend(y)
    3. Tinh:
       - test_acc = correct / total
       - micro_f1 = f1_score(all_labels, all_predictions, average='micro')
       - macro_f1 = f1_score(all_labels, all_predictions, average='macro')
       - weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
       - cm = confusion_matrix(all_labels, all_predictions)
    4. Per-event classification report (Paper 2 bonus)

    Key difference vs Paper 1:
    - Paper 2 tinh accuracy + 3 loai F1 (day du hon)
    - Paper 2 KHONG tinh balanced_accuracy
    - Paper 2 dung mixed-precision training (AMP)
    - Paper 2 co event-level evaluation (classification_report per event)
    """

    def __init__(self, model, device='cuda', label_key='label'):
        self.model = model
        self.device = device
        self.label_key = label_key

    def evaluate_diffattn(
        self,
        test_loader,
        task: str,
        few_shot: int,
        seed: int,
    ) -> ExperimentResult:
        """
        Evaluate DiffAttn model -- reproduce Paper 2 predict() logic.

        Args:
            test_loader: DataLoader cho test set
            task: Task ID
            few_shot: So luong labeled samples (0 = full supervised)
            seed: Random seed

        Returns:
            ExperimentResult tuong thich voi framework
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        start_time = time.time()

        for data in test_loader:
            with torch.no_grad():
                # Reproduce chinh xac tu trainer.py:
                x = (
                    data['image'].to(self.device),
                    {k: v.to(self.device) for k, v in data['text_tokens'].items()},
                )
                y = data[self.label_key].to(self.device)

                logits = self.model(x)
                indices = torch.argmax(logits, dim=1).to(dtype=torch.int32)

                all_predictions.extend(indices.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        inference_time = time.time() - start_time

        y_true_np = np.array(all_labels)
        y_pred_np = np.array(all_predictions)

        # Tinh TAT CA metrics
        metrics = compute_all_metrics(y_true_np, y_pred_np)

        return ExperimentResult(
            model="paper2_diffattn",
            task=task,
            few_shot=few_shot,
            seed=seed,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            precision_per_class=metrics["precision_per_class"],
            recall_per_class=metrics["recall_per_class"],
            f1_per_class=metrics["f1_per_class"],
            confusion_matrix=metrics["confusion_matrix"],
            inference_time_seconds=inference_time,
            notes=f"DiffAttn baseline (micro_f1={metrics['micro_f1']:.4f})",
        )

    def evaluate_per_event(
        self,
        test_loader,
    ) -> Dict[str, Dict]:
        """
        Reproduce Paper 2 event-level evaluation.
        Paper 2 trainer.py tinh classification_report cho tung event rieng.

        Returns:
            Dict[event_name] -> metrics dict
        """
        self.model.eval()
        event_labels = {}
        event_predictions = {}

        for data in test_loader:
            with torch.no_grad():
                x = (
                    data['image'].to(self.device),
                    {k: v.to(self.device) for k, v in data['text_tokens'].items()},
                )
                y = data[self.label_key].to(self.device)
                logits = self.model(x)
                indices = torch.argmax(logits, dim=1)

                events = data['event']
                for i, event in enumerate(events):
                    if event not in event_labels:
                        event_labels[event] = []
                        event_predictions[event] = []
                    event_predictions[event].append(indices[i].item())
                    event_labels[event].append(y[i].item())

        results = {}
        for event, preds in event_predictions.items():
            y_true = np.array(event_labels[event])
            y_pred = np.array(preds)
            results[event] = compute_all_metrics(y_true, y_pred)

        return results


# ============================================================
# Unified Multi-Seed Runner
# ============================================================

def run_multi_seed_evaluation(
    evaluator_fn,
    seeds: List[int] = None,
    task: str = "task1",
    few_shot: int = 100,
    csv_path: Optional[Path] = None,
) -> List[ExperimentResult]:
    """
    Chay evaluation voi nhieu seeds.

    Paper 1 goc: 10 random sets, report mean (khong std)
    Paper 2 goc: 3 runs, report mean (khong std)

    Framework moi: 5 seeds + CI + significance tests

    Args:
        evaluator_fn: Callable(seed) -> ExperimentResult
        seeds: Danh sach seeds (default: RANDOM_SEEDS = [42, 123, 456, 789, 1024])
        task: Task ID
        few_shot: So luong labeled samples
        csv_path: Neu co, save tung result vao CSV (append mode)

    Returns:
        List[ExperimentResult]
    """
    if seeds is None:
        seeds = RANDOM_SEEDS

    results = []

    for seed in seeds:
        print(f"  Running seed {seed}...")

        result = evaluator_fn(seed)
        results.append(result)

        # Save ngay lap tuc (chong mat data khi Colab disconnect)
        if csv_path:
            from .results_manager import save_experiment_result
            save_experiment_result(result, csv_path)
            print(f"    -> Saved to CSV (F1={result.macro_f1:.4f})")

    # In summary
    f1_scores = [r.macro_f1 for r in results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0

    print(f"  Summary: Macro F1 = {mean_f1*100:.2f} +/- {std_f1*100:.2f}%")
    print(f"  Seeds: {[f'{s:.4f}' for s in f1_scores]}")

    return results


# ============================================================
# Comparison: Original vs Framework Metrics
# ============================================================

def compare_with_paper_reported(
    our_results: List[ExperimentResult],
    paper_reported: Dict[str, float],
    paper_name: str,
) -> None:
    """
    So sanh ket qua reproduce voi ket qua bao cao trong paper.

    Args:
        our_results: Ket qua tu multi-seed run
        paper_reported: Dict voi metrics bao cao trong paper goc
            e.g. {"weighted_f1": 0.83, "accuracy": 0.85}
        paper_name: Ten paper
    """
    print(f"\n{'='*50}")
    print(f"  Reproduction Check: {paper_name}")
    print(f"{'='*50}")

    for metric_name, reported_value in paper_reported.items():
        our_values = []
        for r in our_results:
            if hasattr(r, metric_name):
                our_values.append(getattr(r, metric_name))

        if our_values:
            our_mean = np.mean(our_values)
            our_std = np.std(our_values, ddof=1) if len(our_values) > 1 else 0
            diff = (our_mean - reported_value) * 100

            status = "MATCH" if abs(diff) < 2.0 else "DIFF"
            print(
                f"  {metric_name:20s}: "
                f"Paper={reported_value*100:.1f}% | "
                f"Ours={our_mean*100:.1f}% +/- {our_std*100:.1f}% | "
                f"Gap={diff:+.1f}% [{status}]"
            )
