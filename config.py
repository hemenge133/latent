"""
Configuration parameters for training transformer models
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    # Data parameters
    total_samples: int = 51000
    batch_size: int = 64
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, val, test
    
    # Model parameters
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    num_latent: int = 8  # Only for LatentTransformer
    
    # Optimization parameters
    base_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 0.5
    
    # Training parameters
    max_epochs: int = 100
    patience: int = 5  # Early stopping patience
    test_every: int = 5  # Evaluate on test set every N epochs
    save_every: int = 10  # Save checkpoints every N epochs
    
    # Paths
    simple_transformer_logdir: str = "runs/baseline_transformer"
    latent_transformer_logdir: str = "runs/latent_transformer"
    checkpoints_dir: str = "checkpoints"
    
    # Names
    simple_transformer_name: str = "simple_transformer"
    latent_transformer_name: str = "latent_transformer" 
