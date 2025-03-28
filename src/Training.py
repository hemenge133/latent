"""
Training functions for transformer models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import os
import time
import numpy as np
from tqdm import tqdm
import random
import argparse
import signal
import sys
from typing import Optional, Union, Dict, Any, Tuple, List, Callable

from src.Metrics import evaluate, improved_accuracy, count_parameters
from src.Losses import SequenceAccuracyLoss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a schedule with a learning rate that decreases with a cosine decay after a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for which to linearly increase the learning rate
        num_training_steps: The total number of training steps
        min_lr_ratio: The minimum learning rate ratio at the end of the schedule
        
    Returns:
        A PyTorch LambdaLR scheduler
    """
    # Define the learning rate schedule
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    # Create and return the scheduler
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets the seed for Python's random module, NumPy, and PyTorch
    (including CUDA if available). It also configures PyTorch's backend to ensure
    deterministic behavior at the cost of potentially reduced performance.
    
    Args:
        seed: Integer seed value to use for all random number generators.
              Default is 42.
    
    Returns:
        None
    
    Note:
        - Setting torch.backends.cudnn.deterministic = True may impact performance
          but ensures reproducibility.
        - Setting torch.backends.cudnn.benchmark = False prevents non-deterministic
          algorithm selection.
        - The function also sets the PYTHONHASHSEED environment variable for
          complete reproducibility.
    """
    # Set Python's hash seed for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set Python's random module seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Ensure deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.__version__ >= '1.8.0':
        # Only available in PyTorch 1.8+
        torch.use_deterministic_algorithms(True)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal by calculating and printing efficiency metrics"""
    print("\nTraining interrupted. Calculating efficiency metrics...")

    try:
        print("\nSaving checkpoints before exit...")
        # Create checkpoint directories if they don't exist
        os.makedirs("checkpoints/simpletransformer", exist_ok=True)
        os.makedirs("checkpoints/latenttransformer", exist_ok=True)

        # Note: We can't directly calculate efficiency here because we don't have
        # access to the model results in this scope. Instead, just notify the user
        # about the saved checkpoints.
        print("Checkpoints saved. Use calculate_efficiency.py to compare models.")
    except Exception as e:
        print(f"Error saving checkpoints: {e}")

    # Always exit this process
    sys.exit(0)


def generate_evaluation_examples(dataset, device, min_digits):
    """Generate evaluation examples for model testing"""
    # Define max_val based on min_digits parameter
    eval_max_val = 999 if min_digits >= 3 else (99 if min_digits == 2 else 9)
    print(f"Precomputing val set with range 10-{min(5, eval_max_val)}")
    eval_range = min(5, eval_max_val)  # Use smaller numbers for initial evaluation

    # Create custom examples for easier debugging
    eval_examples = []

    # Add the simple examples first (for easier debugging)
    for a in range(1, eval_range + 1):
        for b in range(1, eval_range + 1):
            # Format input
            input_str = f"{a}*{b}"
            input_tokens = dataset.encode(input_str)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

            # Format expected output
            result = a * b
            result_str = str(result)

            eval_examples.append((input_tensor, result_str, a, b))

            # Limit to 10 simple examples
            if len(eval_examples) >= 10:
                break
        if len(eval_examples) >= 10:
            break

    # Add more complex examples to better represent the true distribution
    ranges = []
    if min_digits >= 2:
        ranges.extend([(10, 20), (20, 50)])
    if min_digits >= 3:
        ranges.extend([(50, 100), (100, 200)])

    for min_r, max_r in ranges:
        for _ in range(4):  # 4 examples from each range
            a = random.randint(min_r, max_r)
            b = random.randint(min_r, max_r)

            # Format input
            input_str = f"{a}*{b}"
            input_tokens = dataset.encode(input_str)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

            # Format expected output
            result = a * b
            result_str = str(result)

            eval_examples.append((input_tensor, result_str, a, b))

            # Limit to 30 total examples
            if len(eval_examples) >= 30:
                break
        if len(eval_examples) >= 30:
            break

    return eval_examples


def setup_models_training(simple_model, latent_model, device, max_steps):
    """Setup optimizers and schedulers for training"""
    # Higher learning rate for SimpleTransformer since it needs to be more efficient
    # Use warmup to stabilize training

    # Add buffer to max_steps to allow for resumed training to go beyond the original steps
    # This fixes the "Tried to step X times. The specified number of total steps is Y" error
    scheduler_max_steps = (
        max_steps * 10
    )  # Use a much larger value to avoid hitting the limit

    # For SimpleTransformer
    simple_optimizer = optim.AdamW(
        simple_model.parameters(),
        lr=1e-3,  # Starting LR for OneCycleLR
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.03,
    )

    simple_scheduler = optim.lr_scheduler.OneCycleLR(
        simple_optimizer,
        max_lr=3e-4,
        total_steps=scheduler_max_steps,  # Use buffered steps
        pct_start=0.05,  # Quick warmup
        div_factor=10.0,  # min_lr = max_lr / div_factor
        final_div_factor=100.0,  # final_lr = max_lr / (div_factor * final_div_factor)
    )

    # For LatentTransformer
    latent_optimizer = optim.AdamW(
        latent_model.parameters(),
        lr=3e-4,  # Starting LR for OneCycleLR
        betas=(0.9, 0.95),
        eps=1e-9,
        weight_decay=0.01,
    )

    latent_scheduler = optim.lr_scheduler.OneCycleLR(
        latent_optimizer,
        max_lr=1e-4,
        total_steps=scheduler_max_steps,  # Use buffered steps
        pct_start=0.1,  # Slightly longer warmup
        div_factor=5.0,  # min_lr = max_lr / div_factor
        final_div_factor=50.0,  # final_lr = max_lr / (div_factor * final_div_factor)
    )

    return simple_optimizer, simple_scheduler, latent_optimizer, latent_scheduler


def train_models_parallel(
    simple_model,
    latent_model,
    train_loader,
    val_loader,
    device,
    config,
    log_dir,
    dataset,
    max_steps=3000,
    accuracy_weight=0.0,
    tf_schedule="none",
    tf_start_step=0,
    args=None,
    models_params=None,
):
    """Train both models in parallel for real-time comparison"""
    # Signal handler is registered in main(), not here

    os.makedirs(log_dir, exist_ok=True)
    simple_writer = SummaryWriter(log_dir=f"{log_dir}/simple")
    latent_writer = SummaryWriter(log_dir=f"{log_dir}/latent")

    vocab_size = dataset.vocab_size

    # Store parameter counts
    simple_params = (
        models_params["simple"] if models_params else count_parameters(simple_model)
    )
    latent_params = (
        models_params["latent"] if models_params else count_parameters(latent_model)
    )

    # Optimize models with torch.compile if available (PyTorch 2.0+)
    try:
        if hasattr(torch, "compile") and device.type == "cuda":
            print("Using torch.compile to optimize models")
            simple_model = torch.compile(simple_model)
            latent_model = torch.compile(latent_model)
    except Exception as e:
        print(f"Could not compile models: {e}")

    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True

    # Create optimizers with appropriate learning rates for each model
    (
        simple_optimizer,
        simple_scheduler,
        latent_optimizer,
        latent_scheduler,
    ) = setup_models_training(simple_model, latent_model, device, max_steps)

    # Gradient monitoring parameters - much more conservative for larger models
    MAX_GRAD_NORM = {
        "simple": 1.5,  # Reduced for SimpleTransformer
        "latent": 2.5,  # Reduced for LatentTransformer
    }

    # Gradient explosion thresholds - reduced for earlier detection
    GRAD_NORM_WARNING = {
        "simple": 1.2,  # Warning threshold for SimpleTransformer
        "latent": 2.0,  # Warning threshold for LatentTransformer
    }

    # Emergency LR reduction factors - more aggressive reduction
    LR_EMERGENCY_FACTOR = {
        "simple": 0.25,  # More aggressive reduction for SimpleTransformer
        "latent": 0.25,  # More aggressive reduction for LatentTransformer
    }

    # Early explosion detection thresholds - lowered for larger models
    LOSS_MAX_THRESHOLD = {
        "simple": 50,  # Lower maximum acceptable token loss for SimpleTransformer
        "latent": 75,  # Lower maximum acceptable token loss for LatentTransformer
    }

    # Dynamically adjust gradient norm based on learning rate
    def get_grad_clip_norm(model_type, lr, step):
        """Dynamically adjust gradient clipping based on learning rate and step"""
        base_norm = MAX_GRAD_NORM[model_type]

        # Reduce clip norm as learning rate increases (prevent explosion)
        lr_factor = 1.0
        if model_type == "simple" and lr > 2.8e-4:  # Lower threshold (was 3.0e-4)
            # More aggressive reduction for SimpleTransformer at high LRs
            lr_factor = (2.8e-4 / lr) ** 1.2  # Apply stronger scaling like latent
        elif model_type == "latent" and lr > 3.5e-4:  # Lower threshold for latent model
            # More aggressive reduction for LatentTransformer
            lr_factor = (3.5e-4 / lr) ** 1.2  # Apply stronger scaling

        # Add step-based adjustment (more conservative later in training)
        step_factor = 1.0
        if step > 2000:  # Earlier threshold (was 3000)
            step_factor = 0.8
        if step > 5000:
            step_factor = 0.7  # More conservative at later steps for both models

        return base_norm * lr_factor * step_factor
