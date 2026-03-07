"""
GEDA Evaluation Framework
==========================
Framework đánh giá cho dự án Natural Disaster Classification.

Usage:
    from evaluation import metrics, results_manager, report_generator
    from evaluation.config import TASKS, FEW_SHOT_SETTINGS, RANDOM_SEEDS
"""

from . import config
from . import metrics
from . import results_manager
from . import report_generator
from . import baseline_runners

__version__ = "1.0.0"
__all__ = ["config", "metrics", "results_manager", "report_generator", "baseline_runners"]
