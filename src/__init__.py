# ============================================================================
# CausalCrisis — Package Entry Point
# V4 components are primary; V3 components available via models_v3/losses_v3
# ============================================================================

__version__ = "4.0.0"

# ----- V4-active Modules -----
from .config import (
    V4Config,
    CLIPConfig,
    TrainingConfig,
    EvalConfig,
    GuidedCAConfig,
    LLaVAConfig,
    get_config,
)

from .models import (
    CLIPTaskAdapter,
    CLIPMLPBaseline,
    GuidedCrossAttention,
    ThreeModalityClassifier,
)

from .losses import FocalLoss

from .data import (
    load_crisismmd_annotations,
    extract_clip_features,
    CrisisMMDDataset,
    CrisisMMD3ModalDataset,
    create_dataloaders,
    create_3modal_loaders,
    create_stratified_splits,
    create_lodo_splits,
    compute_class_weights,
)

from .trainer import (
    GenericTrainer,
    BaselineTrainer,
    EarlyStopping,
)

from .evaluate import compute_metrics

# ----- V3 Legacy (explicit import when needed) -----
# from .models_v3 import CausalCrisisV3, HybridDisentangler, BackdoorAdjustment
# from .losses_v3 import CausalCrisisLoss, OrthogonalLoss, SupConLoss
# from .config import CausalCrisisConfig, get_v3_config

__all__ = [
    # Config
    "V4Config", "CLIPConfig", "TrainingConfig", "EvalConfig",
    "GuidedCAConfig", "LLaVAConfig", "get_config",
    # Models
    "CLIPTaskAdapter", "CLIPMLPBaseline",
    "GuidedCrossAttention", "ThreeModalityClassifier",
    # Losses
    "FocalLoss",
    # Data
    "load_crisismmd_annotations", "extract_clip_features",
    "CrisisMMDDataset", "CrisisMMD3ModalDataset",
    "create_dataloaders", "create_3modal_loaders",
    "create_stratified_splits", "create_lodo_splits",
    "compute_class_weights",
    # Training
    "GenericTrainer", "BaselineTrainer", "EarlyStopping",
    # Evaluation
    "compute_metrics",
]
