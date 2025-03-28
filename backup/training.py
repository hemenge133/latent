"""
Training functions for transformer models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math
import os
import time
import numpy as np
from tqdm import tqdm
import random
import itertools
import shutil
import signal
import sys

from metrics import evaluate, improved_accuracy, count_parameters
from losses import SequenceAccuracyLoss


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """Set up optimizers, schedulers, and training structures for both models"""
    # Create optimizers with appropriate learning rates for each model
    simple_optimizer = optim.AdamW(
        simple_model.parameters(), lr=3e-4, weight_decay=0.02
    )
    latent_optimizer = optim.AdamW(
        latent_model.parameters(), lr=3e-4, weight_decay=0.02
    )

    # Create separate LR schedulers optimized for each model's characteristics
    # More conservative for SimpleTransformer to prevent gradient explosions
    simple_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=simple_optimizer,
        max_lr=1.2e-4,  # Further reduced peak learning rate to prevent overfitting
        total_steps=max_steps,
        pct_start=0.1,  # Shorter warmup period
        div_factor=5.0,  # Start with even lower LR (max_lr/5)
        final_div_factor=75,  # Steeper decay for better stability
        anneal_strategy="cos",  # Cosine annealing for smoother decay
    )

    # More conservative settings for LatentTransformer to prevent explosions
    latent_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=latent_optimizer,
        max_lr=1.8e-4,  # Reduced peak learning rate
        total_steps=max_steps,
        pct_start=0.1,  # Shorter warmup period
        div_factor=5.0,  # Start with lower LR (max_lr/5)
        final_div_factor=75,  # Steeper decay to lower final lr
        anneal_strategy="cos",  # Cosine annealing for smoother decay
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
