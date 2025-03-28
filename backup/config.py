"""
Configuration parameters for training transformer models
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    # Data parameters
    total_samples: int = 51000
    batch_size: int = 128  # Increased batch size for better hardware utilization
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, val, test
    min_value: int = 100  # Minimum value for multiplication (3-digit numbers)
    max_value: int = 999  # Maximum value for multiplication (3-digit numbers)

    # Model parameters - balanced for speed vs capacity
    d_model: int = 320  # Reduced from 384 for speed, but still larger than original 256
    nhead: int = 8  # Keep at 8 heads for good attention coverage
    num_layers: int = 5  # Reduced from 6 for speed, but still more than original 4
    num_latent: int = 10  # Reduced from 12 for speed, but still more than original 8

    # Optimization parameters
    base_lr: float = 3e-4  # Increased slightly for faster early training
    weight_decay: float = 0.01
    warmup_steps: int = 1000  # Reduced for faster warmup
    max_grad_norm: float = 1.0

    # Training parameters
    max_epochs: int = 2000  # Increased from 200 to 2000 (10x)
    patience: int = 100  # Increased from 10 to 100 (10x)
    test_every: int = 50  # Increased from 5 to 50 (10x)
    save_every: int = 100  # Increased from 10 to 100 (10x)

    # Step-based evaluation parameters - more frequent for faster feedback
    validate_every_n_steps: int = 50  # Keep the same for frequent validation
    test_every_n_steps: int = 250  # Keep the same for frequent testing

    # Paths
    simple_transformer_logdir: str = (
        "runs/baseline_transformer_3digit_fast"  # New name for the optimized version
    )
    latent_transformer_logdir: str = (
        "runs/latent_transformer_3digit_fast"  # New name for the optimized version
    )
    checkpoints_dir: str = "checkpoints"

    # Names
    simple_transformer_name: str = (
        "simple_transformer_3digit_fast"  # New name for the optimized version
    )
    latent_transformer_name: str = (
        "latent_transformer_3digit_fast"  # New name for the optimized version
    )
