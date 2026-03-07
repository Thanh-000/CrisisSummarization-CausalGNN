"""
GEDA Evaluation Framework — Results Manager
=============================================
Quản lý lưu/đọc kết quả thực nghiệm dưới dạng CSV chuẩn hóa.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .config import (
    ExperimentResult,
    AblationResult,
    CrossTypeResult,
    TASKS,
    FEW_SHOT_SETTINGS,
    RANDOM_SEEDS,
    MODELS,
    ProjectPaths,
)


# ============================================================
# CSV Headers
# ============================================================

BASELINE_HEADERS = [
    "model", "task", "task_name", "few_shot", "seed",
    "accuracy", "macro_f1", "weighted_f1",
    "precision_per_class", "recall_per_class", "f1_per_class",
    "train_time_s", "inference_time_s",
    "timestamp", "notes",
]

ABLATION_HEADERS = [
    "variant_id", "variant_name", "has_graph", "attention_type", "has_mtl",
    "task", "task_name", "few_shot", "seed",
    "macro_f1",
    "timestamp",
]

CROSS_TYPE_HEADERS = [
    "experiment_id", "train_types", "test_type",
    "model", "task", "task_name", "seed",
    "macro_f1",
    "timestamp",
]


# ============================================================
# Save Functions
# ============================================================

def save_experiment_result(
    result: ExperimentResult,
    csv_path: Path,
) -> None:
    """
    Lưu một kết quả experiment vào CSV (append mode).
    Tự tạo header nếu file chưa tồn tại.
    """
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(BASELINE_HEADERS)
        
        task_name = TASKS.get(result.task, {}).get("name", result.task)
        
        writer.writerow([
            result.model,
            result.task,
            task_name,
            result.few_shot,
            result.seed,
            f"{result.accuracy:.6f}",
            f"{result.macro_f1:.6f}",
            f"{result.weighted_f1:.6f}",
            json.dumps(result.precision_per_class),
            json.dumps(result.recall_per_class),
            json.dumps(result.f1_per_class),
            f"{result.train_time_seconds:.2f}",
            f"{result.inference_time_seconds:.2f}",
            datetime.now().isoformat(),
            result.notes,
        ])


def save_ablation_result(
    result: AblationResult,
    csv_path: Path,
) -> None:
    """Lưu kết quả ablation vào CSV."""
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(ABLATION_HEADERS)
        
        task_name = TASKS.get(result.task, {}).get("name", result.task)
        
        writer.writerow([
            result.variant_id,
            result.variant_name,
            result.has_graph,
            result.has_attention,
            result.has_mtl,
            result.task,
            task_name,
            result.few_shot,
            result.seed,
            f"{result.macro_f1:.6f}",
            datetime.now().isoformat(),
        ])


def save_cross_type_result(
    result: CrossTypeResult,
    csv_path: Path,
) -> None:
    """Lưu kết quả cross-type vào CSV."""
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(CROSS_TYPE_HEADERS)
        
        task_name = TASKS.get(result.task, {}).get("name", result.task)
        
        writer.writerow([
            result.experiment_id,
            json.dumps(result.train_types),
            result.test_type,
            result.model,
            result.task,
            task_name,
            result.seed,
            f"{result.macro_f1:.6f}",
            datetime.now().isoformat(),
        ])


# ============================================================
# Load Functions
# ============================================================

def load_results(csv_path: Path) -> List[Dict]:
    """
    Đọc kết quả từ CSV file.
    
    Returns:
        List[Dict] — mỗi row là một dict
    """
    if not csv_path.exists():
        return []
    
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Chuyển đổi types cơ bản
            for key in ["accuracy", "macro_f1", "weighted_f1", "train_time_s", "inference_time_s"]:
                if key in row and row[key]:
                    row[key] = float(row[key])
            
            for key in ["few_shot", "seed"]:
                if key in row and row[key]:
                    row[key] = int(row[key])
            
            results.append(row)
    
    return results


def filter_results(
    results: List[Dict],
    model: Optional[str] = None,
    task: Optional[str] = None,
    few_shot: Optional[int] = None,
) -> List[Dict]:
    """Filter kết quả theo model, task, few_shot."""
    filtered = results
    
    if model:
        filtered = [r for r in filtered if r.get("model") == model]
    if task:
        filtered = [r for r in filtered if r.get("task") == task]
    if few_shot is not None:
        filtered = [r for r in filtered if r.get("few_shot") == few_shot]
    
    return filtered


def get_scores(
    results: List[Dict],
    model: str,
    task: str,
    few_shot: int,
    metric: str = "macro_f1",
) -> List[float]:
    """
    Lấy danh sách scores (qua các seeds) cho một cấu hình cụ thể.
    Dùng để truyền vào confidence_interval() hoặc paired_ttest().
    """
    filtered = filter_results(results, model=model, task=task, few_shot=few_shot)
    return [r[metric] for r in filtered if metric in r]


# ============================================================
# Template CSV Generator
# ============================================================

def generate_template_csv(csv_path: Path, experiment_type: str = "baseline") -> None:
    """
    Tạo CSV template trống với headers chuẩn.
    
    Args:
        csv_path: Đường dẫn file CSV
        experiment_type: "baseline", "ablation", hoặc "cross_type"
    """
    headers_map = {
        "baseline": BASELINE_HEADERS,
        "ablation": ABLATION_HEADERS,
        "cross_type": CROSS_TYPE_HEADERS,
    }
    
    headers = headers_map.get(experiment_type, BASELINE_HEADERS)
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    print(f"[OK] Created template: {csv_path}")
    print(f"   Headers: {', '.join(headers)}")


def generate_all_templates(paths: ProjectPaths) -> None:
    """Tạo tất cả CSV templates."""
    paths.ensure_dirs()
    
    generate_template_csv(paths.baseline_csv, "baseline")
    generate_template_csv(paths.ablation_csv, "ablation")
    generate_template_csv(paths.cross_type_csv, "cross_type")
    generate_template_csv(paths.geda_csv, "baseline")
    
    print(f"\n[TARGET] All templates created in: {paths.results_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    paths = ProjectPaths()
    generate_all_templates(paths)
