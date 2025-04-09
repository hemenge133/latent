"""
Main entry point for parallel comparison of SimpleTransformer and LatentTransformer models.
"""

import os
import time
import random
import shutil
import signal
import sys
import traceback
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import glob
import logging
from loguru import logger  # Replace standard logging with loguru
import itertools # Add import for product

from src.Dataset import MultiplicationDataset
from src.Utils import collate_fn
from src.Config import TrainingConfig
from src.Models import StableSimpleTransformer, StableLatentTransformer
from src.Training import setup_models_training, set_seed
from src.TrainingLoop import train_models_parallel
from src.RunManagement import register_run, get_run_info, get_run_config_from_id


# Signal handler for graceful interruption
def signal_handler(sig, frame):
    print("\nInterrupted by user, shutting down...")
    sys.exit(0)


# Initialize logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
# Add enqueue=True for better handling in subprocesses/interrupts
logger.add("training.log", rotation="100 MB", enqueue=True)


def fix_state_dict(state_dict):
    """Fix state dict keys from older checkpoints"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def filter_state_dict_for_model(state_dict, model):
    """
    Filter a state dict to only include keys that are in the model.
    This helps when loading a checkpoint with more layers than the current model.
    """
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    # First check if we need to remove _orig_mod prefix
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    
    for k, v in state_dict.items():
        # Remove _orig_mod prefix if present
        if has_orig_mod and k.startswith("_orig_mod."):
            k = k[10:]  # Remove '_orig_mod.' prefix
        
        # Only include keys that are in the model's state dict
        if k in model_state_dict:
            # Check for shape compatibility
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
    
    return filtered_state_dict


def check_model_dimensions(checkpoint, d_model):
    """Check if the checkpoint dimensions match the requested model dimensions"""
    # Check if the checkpoint has embed.weight
    if isinstance(checkpoint, dict) and "embed.weight" in checkpoint:
        checkpoint_d_model = checkpoint["embed.weight"].size(1)
        if checkpoint_d_model != d_model:
            return False, checkpoint_d_model
    # If checkpoint has model_state_dict key (older format)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        for k, v in checkpoint["model_state_dict"].items():
            if k.endswith("embed.weight") or k == "embed.weight":
                checkpoint_d_model = v.size(1)
                if checkpoint_d_model != d_model:
                    return False, checkpoint_d_model
                break
    return True, d_model


def extract_model_dimensions(checkpoint):
    """Extract model dimensions from checkpoint for model recreation"""
    dimensions = {
        'd_model': None,
        'num_layers': None,
        'num_latent': None,
        'vocab_size': None,
        'max_len': None
    }
    
    # Try to extract from config
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config = checkpoint['config']
        dimensions['d_model'] = config.get('d_model')
        dimensions['num_layers'] = config.get('num_layers')
        dimensions['num_latent'] = config.get('num_latent')
        dimensions['vocab_size'] = config.get('vocab_size')
        dimensions['max_len'] = config.get('max_len')
    
    # Direct extraction as fallback
    if dimensions['d_model'] is None and 'd_model' in checkpoint:
        dimensions['d_model'] = checkpoint['d_model']
    
    # Try to get num_latent directly from checkpoint
    if dimensions['num_latent'] is None and 'num_latent' in checkpoint:
        dimensions['num_latent'] = checkpoint['num_latent']
    
    # Handle case when checkpoint is already a state_dict
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'embed.weight' in checkpoint:
        # This is already a state dict
        state_dict = checkpoint
    
    # Extract dimensions from state_dict if available
    if state_dict is not None:
        # Extract d_model from embedding weights
        if dimensions['d_model'] is None and 'embed.weight' in state_dict:
            embed_weight = state_dict['embed.weight']
            dimensions['d_model'] = embed_weight.shape[1]
            # Also extract vocab_size from embedding weights
            if dimensions['vocab_size'] is None:
                dimensions['vocab_size'] = embed_weight.shape[0]
        
        # Extract max_len from pos_encoder
        if dimensions['max_len'] is None and 'pos_encoder' in state_dict:
            pos_encoder = state_dict['pos_encoder']
            dimensions['max_len'] = pos_encoder.shape[0]
        
        # Extract num_layers by counting encoder layers
        if dimensions['num_layers'] is None:
            # Count number of encoder layers by looking for layer pattern
            layer_count = 0
            for key in state_dict:
                if 'encoder.layers.' in key:
                    layer_idx = int(key.split('encoder.layers.')[1].split('.')[0])
                    layer_count = max(layer_count, layer_idx + 1)
            
            if layer_count > 0:
                dimensions['num_layers'] = layer_count
        
        # Try to extract num_latent from latent_tokens if present
        if dimensions['num_latent'] is None:
            # Check for latent tokens in state dict
            for key, value in state_dict.items():
                if 'latent_tokens' in key:
                    # The first dimension of latent_tokens is the number of latent tokens
                    dimensions['num_latent'] = value.shape[0]
                    logger.info(f"Extracted num_latent={dimensions['num_latent']} from latent_tokens tensor")
                    break
    
    logger.info(f"Extracted dimensions from checkpoint: {dimensions}")
    return dimensions


def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Enable MPS fallback
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Enable TF32 on Ampere GPUs for faster training with minimal precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Default to mixed precision for training speed
    torch.set_float32_matmul_precision("high")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a SimpleTransformer and LatentTransformer in parallel."
    )

    # Model architecture parameters
    parser.add_argument("--d-model", type=int, default=384, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads (must divide d_model)"
    )
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num-latent", type=int, default=8, help="Number of latent tokens"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.25, help="Dropout probability"
    )

    # Dataset parameters
    parser.add_argument(
        "--min-digits", type=int, default=1, help="Minimum number of digits"
    )
    parser.add_argument(
        "--max-digits", type=int, default=2, help="Maximum number of digits"
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum training steps"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200, help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--bottleneck-factor", type=float, default=1.0, 
        help="Factor for LatentTransformer bottleneck (0.0-1.0, where 1.0 is pure latent)"
    )
    parser.add_argument(
        "--accuracy-weight", type=float, default=0.5, help="Weight for accuracy in loss"
    )
    parser.add_argument(
        "--tf-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "step"],
        help="Teacher forcing schedule",
    )
    parser.add_argument(
        "--tf-start-step",
        type=int,
        default=5000,
        help="Teacher forcing reduction start step",
    )
    parser.add_argument(
        "--use-checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N steps (overrides default behavior)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Run management
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument("--run-id", type=str, help="Run ID to resume from")
    parser.add_argument(
        "--force-config",
        action="store_true",
        help="Force using command-line config instead of checkpoint config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )

    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Perform grid search over specified hyperparameters.",
    )

    args = parser.parse_args()

    # Set seed for reproducibility and ensure Dataset uses the same seed
    seed = args.seed
    set_seed(seed)
    # Explicitly set the dataset seed for consistency
    MultiplicationDataset.set_fixed_seed(seed)

    # Load configuration
    config = TrainingConfig()

    # Override config for stability and to reduce overfitting
    config.base_lr = 3e-4  # Standard learning rate (will be overridden in grid search if enabled)
    config.max_grad_norm = 0.5  # Reduced gradient clipping for stability
    config.warmup_steps = 200  # Extended warmup period
    config.weight_decay = 0.04  # Increased weight decay (will be overridden in grid search if enabled)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set loss function based on args
    criterion_type = "SequenceAccuracyLoss" # Example, adjust as needed
    accuracy_weight = args.accuracy_weight
    logger.info(f"Using {criterion_type} with accuracy weight: {accuracy_weight}")

    # --- Define Hyperparameter Grid ---
    # Smaller grid for 1-digit proof-of-concept
    param_grid = {
        # Optimizer/Scheduler HPs (fixed for this small test)
        "simple_lr": [3e-4],
        "simple_wd": [0.03],
        "latent_lr": [1e-4],
        "latent_wd": [0.01],
        # Model Architecture HPs (small configurations)
        "d_model": [128],
        "nhead": [4],              # Fixed, divides 64 and 128
        "num_layers": [2],         # Fixed at 2 layers
        "dropout": [0.1],          # Fixed low dropout
        "num_latent": [2, 4],      # Two options for latent tokens
        # New parameters to test
        "batch_size": [32, 64],    # Batch size options
        "bottleneck_factor": [0.8, 1.0],  # Bottleneck factor options
        "warmup_steps": [100, 200],  # Warmup steps options
    }
    logger.info(f"Using grid search with extended parameters: {param_grid}")

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    # Filter combinations where d_model % nhead != 0
    hyperparameter_combinations = []
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        if params["d_model"] % params["nhead"] == 0:
            hyperparameter_combinations.append(params)
        else:
            logger.warning(f"Skipping invalid combination: d_model={params['d_model']}, nhead={params['nhead']}")

    if not hyperparameter_combinations:
        logger.error("No valid hyperparameter combinations found after filtering (d_model % nhead). Check param_grid.")
        sys.exit(1)

    # Initialize trackers for best params independently
    best_simple_metric = float("inf") 
    best_simple_params = None
    best_latent_metric = float("inf") # Renamed from best_metric
    best_latent_params = None # Renamed from best_params
    all_results = []

    # --- Checkpoint Loading and Parameter Determination (modified for grid search) ---
    # Checkpoint loading logic might need adjustment if searching d_model/num_layers
    # For now, assume we resume with the *first* parameter set or CLI defaults
    # This part needs careful consideration if resuming within a grid search
    simple_checkpoint = None
    latent_checkpoint = None
    simple_checkpoint_path = "checkpoints/simpletransformer/simpletransformer_latest.pt"
    latent_checkpoint_path = "checkpoints/latenttransformer/latenttransformer_latest.pt"
    start_step = 0
    resume_run = args.resume # Store original resume flag

    # Initial parameter determination (primarily for non-grid search or first run)
    final_d_model = args.d_model
    final_num_layers = args.num_layers
    final_num_latent = args.num_latent

    # >>> MOVE DATASET LOADING HERE <<< (Already moved in original code)
    # Dataset configuration
    min_digits = args.min_digits
    max_digits = args.max_digits
    min_val = 10 ** (min_digits - 1)
    max_val = 10**max_digits - 1

    logger.info(f"Loading datasets...")
    train_dataset = MultiplicationDataset(
        num_samples=20000, # Consider making this configurable
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )
    val_dataset = MultiplicationDataset(
        num_samples=2000, # Consider making this configurable
        split="val",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )
    vocab_size = train_dataset.vocab_size
    logger.info(f"Datasets loaded. Vocabulary size: {vocab_size}")


    # --- Grid Search Loop or Single Run ---
    if args.grid_search:
        logger.info(f"Starting grid search over {len(hyperparameter_combinations)} combinations...")
        # Disable resuming within the grid search loop for simplicity
        # Individual runs within the search start from scratch
        if resume_run:
             logger.warning("Resuming is disabled when performing grid search. Each combination will start training from scratch.")
             resume_run = False # Disable resume for the loop

        for i, params in enumerate(hyperparameter_combinations):
            logger.info(f"--- Grid Search Run {i+1}/{len(hyperparameter_combinations)} ---")
            logger.info(f"Parameters: {params}")

            # Set seed for this specific run
            run_seed = seed + i # Offset seed for different runs
            set_seed(run_seed)
            MultiplicationDataset.set_fixed_seed(run_seed) # Reset dataset seed too

            # Extract model and training params from the grid
            current_d_model = params['d_model'] # Use directly from grid
            current_nhead = params['nhead']
            current_num_layers = params['num_layers']
            current_dropout = params['dropout']
            current_num_latent = params['num_latent']
            current_batch_size = params.get('batch_size', args.batch_size)  # New parameter
            current_bottleneck_factor = params.get('bottleneck_factor', 1.0)  # New parameter
            current_warmup_steps = params.get('warmup_steps', 200)  # New parameter

            # Create models for this run
            logger.info(f"Creating models for this run: d_model={current_d_model}, nhead={current_nhead}, "
                       f"layers={current_num_layers}, dropout={current_dropout}, latent={current_num_latent}, "
                       f"bottleneck={current_bottleneck_factor}, batch_size={current_batch_size}, "
                       f"warmup_steps={current_warmup_steps}")
            
            simple_transformer = StableSimpleTransformer(
                vocab_size=vocab_size,
                d_model=current_d_model,
                nhead=current_nhead,
                num_layers=current_num_layers,
                dropout=current_dropout,
            ).to(device)

            latent_transformer = StableLatentTransformer(
                vocab_size=vocab_size,
                d_model=current_d_model,
                nhead=current_nhead,
                num_layers=current_num_layers,
                num_latent=current_num_latent,
                dropout=current_dropout,
                bottleneck_factor=current_bottleneck_factor  # Use from grid
            ).to(device)

            simple_params_count = sum(p.numel() for p in simple_transformer.parameters())
            latent_params_count = sum(p.numel() for p in latent_transformer.parameters())
            logger.info(f"Simple Params: {simple_params_count:,}, Latent Params: {latent_params_count:,}")

            # Define unique log directory for this run
            param_str = "_".join([f"{k}{v}" for k, v in params.items()])
            log_dir = f"runs/grid_search/{param_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Logging to: {log_dir}")

            # Train with current hyperparameters
            try:
                 # Pass hyperparameters directly to train_models_parallel
                 # Also pass the current model parameters
                 results = train_models_parallel(
                     models={"simple": simple_transformer, "latent": latent_transformer},
                     dataset=train_dataset,
                     dataset_val=val_dataset,
                     vocab_size=vocab_size,
                     device=device,
                     max_steps=args.max_steps,
                     batch_size=current_batch_size,  # Use from grid
                     config=config,
                     args=args,
                     models_params={"simple": simple_params_count, "latent": latent_params_count},
                     start_step=0,
                     simple_checkpoint=None,
                     latent_checkpoint=None,
                     log_dir=log_dir,
                     # Pass the hyperparameters from the grid
                     simple_lr=params.get("simple_lr", config.base_lr),
                     simple_wd=params.get("simple_wd", config.weight_decay),
                     latent_lr=params.get("latent_lr", config.base_lr),
                     latent_wd=params.get("latent_wd", config.weight_decay),
                     warmup_steps=current_warmup_steps,  # New parameter
                     bottleneck_factor=current_bottleneck_factor,  # New parameter
                 )

                 # Store results (e.g., final validation loss and accuracy for BOTH models)
                 # First, log the full results structure for debugging purposes
                 logger.info(f"Results structure: {results.keys()}")
                 
                 # The train_models_parallel function returns a dictionary with the
                 # structure: { "simple": {...}, "latent": {...}, "training_time": value }
                 
                 # For SimpleTransformer:
                 simple_final_val_loss = results.get("simple", {}).get("loss", float('inf'))
                 simple_final_val_acc = results.get("simple", {}).get("sequence_accuracy", 0.0)
                 
                 # For LatentTransformer:
                 latent_final_val_loss = results.get("latent", {}).get("loss", float('inf'))
                 latent_final_val_acc = results.get("latent", {}).get("sequence_accuracy", 0.0)
                 
                 # Log detailed structure if we couldn't find the expected metrics
                 if simple_final_val_loss == float('inf') or latent_final_val_loss == float('inf'):
                     logger.warning("Could not find expected metrics in the standard structure!")
                     logger.warning(f"Full results object: {results}")

                 current_run_result = {
                     "params": params,
                     "simple_final_val_loss": simple_final_val_loss,
                     "simple_final_val_accuracy": simple_final_val_acc,
                     "latent_final_val_loss": latent_final_val_loss,
                     "latent_final_val_accuracy": latent_final_val_acc,
                     "log_dir": log_dir
                 }
                 all_results.append(current_run_result)
                 logger.info(f"Run {i+1} finished. Simple [Loss: {simple_final_val_loss:.6f}, Acc: {simple_final_val_acc:.2%}], Latent [Loss: {latent_final_val_loss:.6f}, Acc: {latent_final_val_acc:.2%}]")

                 # Update best parameters independently
                 if simple_final_val_loss < best_simple_metric:
                     best_simple_metric = simple_final_val_loss
                     best_simple_params = params
                     logger.info(f"*** New best SimpleTransformer validation loss found: {best_simple_metric:.6f} with params: {best_simple_params} ***")

                 if latent_final_val_loss < best_latent_metric:
                     best_latent_metric = latent_final_val_loss
                     best_latent_params = params
                     logger.info(f"*** New best LatentTransformer validation loss found: {best_latent_metric:.6f} with params: {best_latent_params} ***")

            except Exception as e:
                 logger.error(f"Error during grid search run {i+1} with params {params}: {e}")
                 logger.error(traceback.format_exc())
                 all_results.append({
                     "params": params,
                     "simple_final_val_loss": float('inf'), # Mark error runs
                     "simple_final_val_accuracy": 0.0,
                     "latent_final_val_loss": float('inf'),
                     "latent_final_val_accuracy": 0.0,
                     "log_dir": log_dir,
                     "error": str(e)
                 })

        # --- End of Grid Search Loop ---
        logger.info("--- Grid Search Complete ---")
        logger.info(f"Best SimpleTransformer validation loss found: {best_simple_metric:.6f}")
        logger.info(f"Best SimpleTransformer hyperparameters: {best_simple_params}")
        logger.info(f"Best LatentTransformer validation loss found: {best_latent_metric:.6f}")
        logger.info(f"Best LatentTransformer hyperparameters: {best_latent_params}")

        # Save results
        results_file = f"grid_search_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Grid search results saved to {results_file}")

    else:
        # --- Standard Single Run ---
        logger.info("Starting standard single training run...")

        # Handle checkpoint loading for single run
        if resume_run:
             logger.info("Attempting to resume training...")
             # (Existing checkpoint loading logic based on resume_run flag)
             # Try loading both checkpoints first
             if os.path.exists(simple_checkpoint_path):
                 try:
                     simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
                     logger.info(f"Successfully loaded SimpleTransformer checkpoint from {simple_checkpoint_path}")
                 except Exception as e:
                     logger.error(f"Failed to load SimpleTransformer checkpoint from {simple_checkpoint_path}: {e}")
                     simple_checkpoint = None # Ensure it's None if loading failed
             else:
                 logger.warning(f"SimpleTransformer checkpoint not found at {simple_checkpoint_path}")
                 
             if os.path.exists(latent_checkpoint_path):
                 try:
                     latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)
                     logger.info(f"Successfully loaded LatentTransformer checkpoint from {latent_checkpoint_path}")
                 except Exception as e:
                     logger.error(f"Failed to load LatentTransformer checkpoint from {latent_checkpoint_path}: {e}")
                     latent_checkpoint = None
             else:
                 logger.warning(f"LatentTransformer checkpoint not found at {latent_checkpoint_path}")

             # Extract dimensions and determine final parameters based on Simple checkpoint
             if simple_checkpoint:
                 simple_dims = extract_model_dimensions(simple_checkpoint)
                 start_step = max(start_step, simple_checkpoint.get('step', 0))
                 
                 # Override d_model and num_layers ONLY from simple checkpoint if valid
                 if simple_dims['d_model'] is not None:
                     if args.force_config:
                         logger.warning(f"--force-config is set. Ignoring d_model={simple_dims['d_model']} from simple checkpoint, using CLI value {args.d_model}.")
                         # Keep final_d_model as args.d_model
                     elif simple_dims['d_model'] != args.d_model:
                         logger.warning(f"Overriding CLI d_model={args.d_model} with value from simple checkpoint: {simple_dims['d_model']}")
                         final_d_model = simple_dims['d_model']
                     # else: keep final_d_model as args.d_model
                 else:
                     logger.warning("Could not extract d_model from simple checkpoint, using CLI value.")
                     # Keep final_d_model as args.d_model

                 if simple_dims['num_layers'] is not None:
                     if args.force_config:
                         logger.warning(f"--force-config is set. Ignoring num_layers={simple_dims['num_layers']} from simple checkpoint, using CLI value {args.num_layers}.")
                         # Keep final_num_layers as args.num_layers
                     elif simple_dims['num_layers'] != args.num_layers:
                         logger.warning(f"Overriding CLI num_layers={args.num_layers} with value from simple checkpoint: {simple_dims['num_layers']}")
                         final_num_layers = simple_dims['num_layers']
                     # else: keep final_num_layers as args.num_layers
                 else:
                     logger.warning("Could not extract num_layers from simple checkpoint, using CLI value.")
                     # Keep final_num_layers as args.num_layers
             # else: Keep final_d_model/final_num_layers as args.d_model/args.num_layers if no simple checkpoint

             # Extract dimensions and determine final num_latent based on Latent checkpoint
             if latent_checkpoint:
                 latent_dims = extract_model_dimensions(latent_checkpoint)
                 start_step = max(start_step, latent_checkpoint.get('step', 0))

                 # Override num_latent ONLY from latent checkpoint if valid
                 if latent_dims['num_latent'] is not None:
                     if args.force_config:
                         logger.warning(f"--force-config is set. Ignoring num_latent={latent_dims['num_latent']} from latent checkpoint, using CLI value {args.num_latent}.")
                         # Keep final_num_latent as args.num_latent
                     elif latent_dims['num_latent'] != args.num_latent:
                         logger.warning(f"Overriding CLI num_latent={args.num_latent} with value from latent checkpoint: {latent_dims['num_latent']}")
                         final_num_latent = latent_dims['num_latent']
                     # else: keep final_num_latent as args.num_latent
                 else:
                     logger.warning("Could not extract num_latent from latent checkpoint, using CLI value.")
                     # Keep final_num_latent as args.num_latent
                 
                 # Check for d_model consistency (Latent vs Final determined above)
                 if latent_dims['d_model'] is not None and latent_dims['d_model'] != final_d_model:
                     logger.warning(f"d_model mismatch! Latent checkpoint suggests {latent_dims['d_model']}, but using {final_d_model} (determined by simple checkpoint/CLI).")
                 # Check for num_layers consistency
                 if latent_dims['num_layers'] is not None and latent_dims['num_layers'] != final_num_layers:
                     logger.warning(f"num_layers mismatch! Latent checkpoint suggests {latent_dims['num_layers']}, but using {final_num_layers} (determined by simple checkpoint/CLI).")
             # else: Keep final_num_latent as args.num_latent if no latent checkpoint
             
             logger.info(f"Resuming with parameters: d_model={final_d_model}, layers={final_num_layers}, latent={final_num_latent}, start_step={start_step}")

        else: # if not resume_run
             logger.info("Starting training from scratch (no resume).")
             start_step = 0
             # Use CLI args directly
             final_d_model = args.d_model
             final_num_layers = args.num_layers
             final_num_latent = args.num_latent


        # Log the final parameters that will be used for model creation
        logger.info(f"Final model creation parameters (standard run): d_model={final_d_model}, nhead={args.nhead}, layers={final_num_layers}, dropout={args.dropout}")

        # Create the models with the determined parameters
        logger.info(f"Creating SimpleTransformer with d_model={final_d_model}, nhead={args.nhead}, num_layers={final_num_layers}, dropout={args.dropout}")
        simple_transformer = StableSimpleTransformer(
             vocab_size=vocab_size,
             d_model=final_d_model,
             nhead=args.nhead, # Use nhead from args
             num_layers=final_num_layers,
             dropout=args.dropout, # Use dropout from args
        ).to(device)

        logger.info(f"Creating LatentTransformer with d_model={final_d_model}, nhead={args.nhead}, num_layers={final_num_layers}, "
                   f"dropout={args.dropout}, num_latent={final_num_latent}, bottleneck_factor={args.bottleneck_factor}")
        latent_transformer = StableLatentTransformer(
             vocab_size=vocab_size,
             d_model=final_d_model,
             nhead=args.nhead, # Use nhead from args
             num_layers=final_num_layers,
             num_latent=final_num_latent,
             dropout=args.dropout, # Use dropout from args
             bottleneck_factor=args.bottleneck_factor  # Use from args
        ).to(device)

        simple_params_count = sum(p.numel() for p in simple_transformer.parameters())
        latent_params_count = sum(p.numel() for p in latent_transformer.parameters())
        logger.info(f"SimpleTransformer has {simple_params_count:,} parameters")
        logger.info(f"LatentTransformer has {latent_params_count:,} parameters")

        # Define log directory for TensorBoard (standard run)
        log_dir = "runs/parallel_comparison"
        logger.info(f"Logging to: {log_dir}")

        # --- Pass Checkpoints and Start Step to Training Loop ---
        logger.info(f"Passing start_step={start_step} to training loop.")

        # Flush logs before training
        logger.info("Flushing logs before starting training loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Call training loop for the single run
        results = train_models_parallel(
            models={"simple": simple_transformer, "latent": latent_transformer},
            dataset=train_dataset,
            dataset_val=val_dataset,
            vocab_size=vocab_size,
            device=device,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            config=config,
            args=args,
            models_params={"simple": simple_params_count, "latent": latent_params_count},
            start_step=start_step, # Use the determined start step
            simple_checkpoint=simple_checkpoint if resume_run else None,
            latent_checkpoint=latent_checkpoint if resume_run else None,
            log_dir=log_dir,
            # Pass default hyperparameters (from config or args if overridden)
            simple_lr=config.base_lr, # Use config or potentially args if you add LR/WD args later
            simple_wd=config.weight_decay,
            latent_lr=config.base_lr,
            latent_wd=config.weight_decay,
        )

        # Flush logs after training
        logger.info("Flushing logs after training loop completion...")
        sys.stdout.flush()
        sys.stderr.flush()

        # Print comparison (assuming results dict contains necessary info)
        # This part might need adjustment based on the return value of train_models_parallel
        # logger.info("\n--- Training Complete ---")
        # if results:
        #      simple_final_loss = results.get("final_metrics", {}).get("simple", {}).get("val_loss", "N/A")
        #      latent_final_loss = results.get("final_metrics", {}).get("latent", {}).get("val_loss", "N/A")
        #      simple_final_acc = results.get("final_metrics", {}).get("simple", {}).get("val_accuracy", "N/A")
        #      latent_final_acc = results.get("final_metrics", {}).get("latent", {}).get("val_accuracy", "N/A")
        #      logger.info(f"SimpleTransformer Final Val Loss: {simple_final_loss}, Accuracy: {simple_final_acc}")
        #      logger.info(f"LatentTransformer Final Val Loss: {latent_final_loss}, Accuracy: {latent_final_acc}")
        # else:
        #      logger.warning("Training function did not return results.")


    # Finalize logging
    logger.info("Finalizing logging before exit...")
    logger.complete()

if __name__ == "__main__":
    main()
