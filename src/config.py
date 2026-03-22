# ============================================================================
# CausalCrisis V3 — Configuration Module
# Tất cả hyperparameters tập trung tại đây
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
import torch


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
class AdapterConfig:
    """ð CLIP Task Adapter configuration."""
    use_adapter: bool = True          # Enable/disable adapter
    bottleneck: int = 128             # Bottleneck dimension (768 → 128 → 768)
    residual_ratio: float = 0.2      # Blend ratio: (1-r)*CLIP + r*adapted
    dropout: float = 0.1


@dataclass
class DisentangleConfig:
    """Per-Modality Causal Disentanglement configuration."""
    input_dim: int = 768
    hidden_dim: int = 512
    causal_dim: int = 384
    spurious_dim: int = 384
    dropout: float = 0.1
    # Hybrid ICA-Adversarial settings
    use_ica_init: bool = True         # 🆕 ICA initialization (CCA-inspired)
    ica_whiten: bool = True
    adversarial_refine: bool = True   # Adversarial fine-tuning after ICA


@dataclass
class FusionConfig:
    """Cross-Modal Causal Fusion configuration."""
    d_model: int = 384
    nhead: int = 8
    fusion_type: str = "cross_attention"  # "cross_attention" | "bilinear" | "both"
    use_gate: bool = True


@dataclass
class ClassifierConfig:
    """Classification Head configuration."""
    input_dim: int = 1152  # 384 * 3 (C_v + C_t + C_vt)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_classes: int = 2   # Task 1: binary (informative vs not)
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler: str = "cosine_warm_restarts"  # 🆕 warm restarts
    T_0: int = 30
    T_mult: int = 2
    eta_min: float = 1e-6
    
    # Training
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    early_stop_patience: int = 15
    
    # 2-Phase Training Protocol (with gradual loss activation)
    warmup_epochs: int = 20           # ð§ Extended from 10 for adapter convergence
    
    # Loss weights (dùng adaptive nếu use_adaptive_weights=True)
    focal_gamma: float = 2.0
    alpha_adv: float = 0.1
    alpha_ortho: float = 0.05
    alpha_supcon: float = 0.1     # 🆕 Supervised Contrastive
    alpha_recon: float = 0.0      # Disabled by default (ortho + adv đủ)
    use_adaptive_weights: bool = True  # 🆕 Adaptive Weighting
    adaptive_init_logvar: float = 2.0  # ð§ Start with low precision for stability
    
    # GRL settings
    grl_lambda_max: float = 0.5       # ð§ Reduced from 1.0 for gentler adversarial
    grl_warmup_epochs: int = 20        # ð§ Extended from 10
    
    # Early stopping
    early_stop_patience: int = 25      # ð§ Increased from 15
    
    # Backdoor Adjustment
    memory_bank_size: int = 1000   # Giảm từ 2000 (simplicity test)
    ba_n_samples: int = 30         # Giảm từ 50 (simplicity test)
    ba_start_epoch: int = 50       # Bắt đầu BA từ epoch 50
    
    # Domain config
    num_domains: int = 7           # 7 disasters trong CrisisMMD
    
    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # In-domain
    test_split: float = 0.15
    val_split: float = 0.15
    
    # LODO
    lodo_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Ablation variants
    ablation_variants: List[str] = field(default_factory=lambda: [
        "full",
        "no_causal",           # MLP baseline
        "no_cvt",              # Without cross-modal factor
        "no_ica",              # Without ICA (adversarial only)
        "no_adversarial",      # Without adversarial (ICA only)
        "no_supcon",           # Without SupCon loss
        "no_ortho",            # Without orthogonal constraint
        "no_backdoor",         # Without Backdoor Adjustment
        "bilinear_fusion",     # Bilinear instead of cross-attention
        "joint_disentangle",   # CAMO-style joint (not per-modality)
    ])
    
    # Bootstrap significance
    n_bootstrap: int = 10000
    alpha: float = 0.05


@dataclass
class CausalCrisisConfig:
    """Master configuration."""
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)  # ð
    disentangle: DisentangleConfig = field(default_factory=DisentangleConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Experiment tracking
    experiment_name: str = "causalcrisis_v3"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def get_config(task: str = "task1") -> CausalCrisisConfig:
    """Get configuration for a specific task."""
    config = CausalCrisisConfig()
    
    if task == "task1":
        config.classifier.num_classes = 2   # informative vs not
    elif task == "task2":
        config.classifier.num_classes = 8   # humanitarian categories
        config.classifier.input_dim = 1152
    elif task == "task3":
        config.classifier.num_classes = 3   # damage severity
    
    return config
