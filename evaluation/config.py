"""
GEDA Evaluation Framework — Configuration
==========================================
Cấu hình trung tâm cho toàn bộ evaluation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

# ============================================================
# Constants
# ============================================================

TASKS = {
    "task1": {
        "name": "Informativeness",
        "classes": ["not_informative", "informative"],
        "num_classes": 2,
        "metric": "macro_f1",
    },
    "task2": {
        "name": "Humanitarian",
        "classes": [
            "infrastructure_and_utility_damage",
            "not_humanitarian",
            "other_relevant_information",
            "rescue_volunteering_or_donation_effort",
            "vehicle_damage",
            "affected_individuals",
            "injured_or_dead_people",
            "missing_or_found_people",
        ],
        "num_classes": 8,
        "metric": "macro_f1",
    },
    "task3": {
        "name": "Damage Severity",
        "classes": ["little_or_no_damage", "mild_damage", "severe_damage"],
        "num_classes": 3,
        "metric": "macro_f1",
    },
}

FEW_SHOT_SETTINGS = [50, 100, 250, 500]

RANDOM_SEEDS = [42, 123, 456, 789, 1024]

DISASTER_TYPES = {
    "hurricane": ["hurricane_harvey", "hurricane_irma", "hurricane_maria"],
    "earthquake": ["mexico_earthquake", "iraq_iran_earthquake"],
    "wildfire": ["california_wildfires"],
    "flood": ["sri_lanka_floods"],
}

MODELS = {
    "paper1_gnn": "Paper 1 (GNN Semi-Supervised)",
    "paper2_diffattn": "Paper 2 (Differential Attention)",
    "geda": "GEDA (Ours)",
}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ExperimentResult:
    """Kết quả một lần chạy thực nghiệm."""
    model: str
    task: str
    few_shot: int
    seed: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision_per_class: List[float] = field(default_factory=list)
    recall_per_class: List[float] = field(default_factory=list)
    f1_per_class: List[float] = field(default_factory=list)
    confusion_matrix: Optional[List[List[int]]] = None
    train_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    notes: str = ""


@dataclass
class AblationResult:
    """Kết quả một variant trong ablation study."""
    variant_id: str  # e.g., "A1", "A2", ...
    variant_name: str
    has_graph: bool
    has_attention: str  # "none", "SA", "SA+GCA", "Full"
    has_mtl: bool
    task: str
    few_shot: int
    seed: int
    macro_f1: float


@dataclass
class CrossTypeResult:
    """Kết quả cross-disaster-type experiment."""
    experiment_id: str  # "5A", "5B", "5C", "5D"
    train_types: List[str]
    test_type: str
    model: str
    task: str
    seed: int
    macro_f1: float


# ============================================================
# Paths
# ============================================================

@dataclass
class ProjectPaths:
    """Các đường dẫn quan trọng trong project."""
    root: Path = Path(".")
    
    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"
    
    @property
    def results_dir(self) -> Path:
        return self.root / "evaluation" / "results"
    
    @property
    def figures_dir(self) -> Path:
        return self.root / "evaluation" / "figures"
    
    @property
    def baseline_csv(self) -> Path:
        return self.results_dir / "baseline_5seeds.csv"
    
    @property
    def ablation_csv(self) -> Path:
        return self.results_dir / "ablation_results.csv"
    
    @property
    def cross_type_csv(self) -> Path:
        return self.results_dir / "cross_type_results.csv"
    
    @property
    def geda_csv(self) -> Path:
        return self.results_dir / "geda_fewshot_results.csv"
    
    def ensure_dirs(self):
        """Tạo tất cả directories cần thiết."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
