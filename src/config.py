# ============================================================================
# CausalCrisis — Configuration Module
# V4-active config + V3 legacy config (for backward compat)
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ============================================================================
# Shared Configs
# ============================================================================
@dataclass
class CLIPConfig:
    """CLIP Feature Extraction configuration."""
    model_name: str = "ViT-L/14"
    pretrained: str = "openai"
    image_dim: int = 768
    text_dim: int = 768
    max_text_length: int = 77
    cache_dir: str = "cached_features"


@dataclass
class TrainingConfig:
    """Training configuration (shared V3/V4)."""
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01

    # Scheduler
    scheduler: str = "cosine"
    T_max: int = 100
    eta_min: float = 1e-6

    # Training
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    early_stop_patience: int = 20

    # Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    test_split: float = 0.15
    val_split: float = 0.15

    # LODO
    lodo_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Bootstrap significance
    n_bootstrap: int = 10000
    alpha: float = 0.05


# ============================================================================
# V4 Configs
# ============================================================================
@dataclass
class GuidedCAConfig:
    """Guided Cross-Attention configuration."""
    d_model: int = 768
    dropout: float = 0.1


@dataclass
class LLaVAConfig:
    """LLaVA caption generation configuration."""
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    encoding_strategy: str = "combined"  # "combined" | "separate"
    max_new_tokens: int = 200
    checkpoint_interval: int = 500
    prompt: str = (
        "Describe this image in detail for crisis analysis. "
        "Focus on: visible damage, affected infrastructure, "
        "people's activities, environmental conditions, "
        "and any signs of emergency response."
    )


@dataclass
class V4Config:
    """CausalCrisis V4 — Master configuration."""
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    guided_ca: GuidedCAConfig = field(default_factory=GuidedCAConfig)
    llava: LLaVAConfig = field(default_factory=LLaVAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # V4 feature flags
    num_modalities: int = 3
    use_guided_ca: bool = True
    use_llava: bool = True

    # Experiment tracking
    experiment_name: str = "causalcrisis_v4"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


# ============================================================================
# V3 Legacy Configs (for backward compatibility)
# ============================================================================
@dataclass
class AdapterConfig:
    """CLIP Task Adapter configuration."""
    use_adapter: bool = True
    bottleneck: int = 128
    residual_ratio: float = 0.2
    dropout: float = 0.1


@dataclass
class DisentangleConfig:
    """Per-Modality Causal Disentanglement (V3, archived)."""
    input_dim: int = 768
    hidden_dim: int = 512
    causal_dim: int = 384
    spurious_dim: int = 384
    dropout: float = 0.1
    use_ica_init: bool = True
    ica_whiten: bool = True
    adversarial_refine: bool = True


@dataclass
class FusionConfig:
    """Cross-Modal Fusion (V3, archived)."""
    d_model: int = 384
    nhead: int = 8
    fusion_type: str = "cross_attention"
    use_gate: bool = True


@dataclass
class ClassifierConfig:
    """Classification Head (V3)."""
    input_dim: int = 1152
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_classes: int = 2
    dropout: float = 0.2


@dataclass
class CausalCrisisConfig:
    """V3 Master configuration (legacy, for old notebooks)."""
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    disentangle: DisentangleConfig = field(default_factory=DisentangleConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    experiment_name: str = "causalcrisis_v3"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


# ============================================================================
# Config Factory
# ============================================================================
def get_config(task: str = "task1", version: str = "v4") -> V4Config:
    """Get configuration for a specific task and version.

    Args:
        task: "task1" (informative), "task2" (humanitarian), "task3" (damage)
        version: "v4" (default) or "v3" (legacy)

    Returns:
        V4Config or CausalCrisisConfig depending on version
    """
    if version == "v3":
        config = CausalCrisisConfig()
        if task == "task1":
            config.classifier.num_classes = 2
        elif task == "task2":
            config.classifier.num_classes = 8
        elif task == "task3":
            config.classifier.num_classes = 3
        return config

    # V4 (default)
    config = V4Config()

    # V4 uses simpler training defaults
    config.training.epochs = 50
    config.training.batch_size = 32
    config.training.early_stop_patience = 7
    config.training.label_smoothing = 0.05

    return config


def get_v3_config(task: str = "task1") -> CausalCrisisConfig:
    """Convenience alias for V3 config."""
    return get_config(task=task, version="v3")
