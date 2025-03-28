"""
Configuration parameters for training transformer models.

This module defines the configuration dataclasses used throughout the training process,
ensuring consistent parameter usage across different components of the system.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Union, Dict, Any


@dataclass
class TrainingConfig:
    """
    Configuration class for transformer model training parameters.
    
    This class contains all parameters required for training both SimpleTransformer
    and LatentTransformer models, including data, model architecture, optimization,
    and training control parameters.
    """
    # Data parameters
    total_samples: int = 51000  # Total number of samples to generate for the dataset
    batch_size: int = 128  # Number of samples per batch during training
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # Ratio for train, val, test split
    min_value: int = 100  # Minimum value for multiplication (inclusive)
    max_value: int = 999  # Maximum value for multiplication (inclusive)

    # Model parameters
    d_model: int = 320  # Hidden dimension size of the transformer model
    nhead: int = 8  # Number of attention heads in the transformer
    num_layers: int = 5  # Number of transformer encoder/decoder layers
    num_latent: int = 10  # Number of latent tokens for LatentTransformer

    # Optimization parameters
    base_lr: float = 3e-4  # Base learning rate for optimizer
    weight_decay: float = 0.01  # Weight decay coefficient for L2 regularization
    warmup_steps: int = 1000  # Number of steps for learning rate warmup
    max_grad_norm: float = 1.0  # Maximum norm for gradient clipping

    # Training parameters
    max_epochs: int = 2000  # Maximum number of training epochs
    patience: int = 100  # Patience for early stopping (epochs without improvement)
    test_every: int = 50  # Frequency of testing in epochs
    save_every: int = 100  # Frequency of checkpoint saving in epochs

    # Step-based evaluation parameters
    validate_every_n_steps: int = 50  # Frequency of validation in steps
    test_every_n_steps: int = 250  # Frequency of testing in steps

    # Paths
    simple_transformer_logdir: str = "runs/baseline_transformer_3digit_fast"  # Log directory for SimpleTransformer
    latent_transformer_logdir: str = "runs/latent_transformer_3digit_fast"  # Log directory for LatentTransformer
    checkpoints_dir: str = "checkpoints"  # Directory for saving model checkpoints

    # Names
    simple_transformer_name: str = "simple_transformer_3digit_fast"  # Name identifier for SimpleTransformer
    latent_transformer_name: str = "latent_transformer_3digit_fast"  # Name identifier for LatentTransformer
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        
        This method ensures that all configuration parameters are within valid ranges
        and consistent with each other.
        """
        # Validate split ratio sums to 1.0
        if sum(self.split_ratio) != 1.0:
            raise ValueError(f"Split ratio must sum to 1.0, got {self.split_ratio}")
        
        # Validate min_value is less than max_value
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value ({self.min_value}) must be less than max_value ({self.max_value})")
