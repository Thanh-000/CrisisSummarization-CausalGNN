# CausalCrisis V3 — Source Package
from .config import CausalCrisisConfig, AdapterConfig, get_config
from .models import CausalCrisisV3, CLIPMLPBaseline, CLIPTaskAdapter
from .losses import CausalCrisisLoss, FocalLoss, SupConLoss, AdaptiveLossWeighting
from .data import (
    load_crisismmd_annotations,
    extract_and_cache_clip_features,
    CrisisMMDDataset,
    create_stratified_splits,
    create_lodo_splits,
    create_dataloaders,
    compute_class_weights,
)
from .trainer import CausalCrisisTrainer, BaselineTrainer
from .evaluate import (
    compute_metrics,
    paired_bootstrap_test,
    plot_training_curves,
    plot_tsne_causal_features,
)

__version__ = "3.0.0"
__all__ = [
    "CausalCrisisConfig", "AdapterConfig", "get_config",
    "CausalCrisisV3", "CLIPMLPBaseline", "CLIPTaskAdapter",
    "CausalCrisisLoss", "FocalLoss", "SupConLoss", "AdaptiveLossWeighting",
    "load_crisismmd_annotations", "extract_and_cache_clip_features",
    "CrisisMMDDataset", "create_stratified_splits", "create_lodo_splits",
    "create_dataloaders", "compute_class_weights",
    "CausalCrisisTrainer", "BaselineTrainer",
    "compute_metrics", "paired_bootstrap_test",
    "plot_training_curves", "plot_tsne_causal_features",
]
