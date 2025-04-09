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
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num-latent", type=int, default=8, help="Number of latent tokens"
    )

    # Dataset parameters
    parser.add_argument(
        "--min-digits", type=int, default=1, help="Minimum number of digits"
    )
    parser.add_argument(
        "--max-digits", type=int, default=2, help="Maximum number of digits"
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum training steps"
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

    args = parser.parse_args()

    # Set seed for reproducibility and ensure Dataset uses the same seed
    seed = args.seed
    set_seed(seed)
    # Explicitly set the dataset seed for consistency
    MultiplicationDataset.set_fixed_seed(seed)

    # Load configuration
    config = TrainingConfig()

    # Override config for stability and to reduce overfitting
    config.base_lr = 3e-4  # Standard learning rate
    config.max_grad_norm = 0.5  # Reduced gradient clipping for stability
    config.warmup_steps = 200  # Extended warmup period
    config.weight_decay = 0.04  # Increased weight decay to fight overfitting

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

    # --- Checkpoint Loading and Parameter Determination --- 
    simple_checkpoint = None
    latent_checkpoint = None
    simple_checkpoint_path = "checkpoints/simpletransformer/simpletransformer_latest.pt"
    latent_checkpoint_path = "checkpoints/latenttransformer/latenttransformer_latest.pt"
    start_step = 0

    # Initialize final parameters with command-line args
    final_d_model = args.d_model
    final_num_layers = args.num_layers
    final_num_latent = args.num_latent

    if args.resume:
        logger.info("Resume flag is set. Attempting to load checkpoints...")
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
                elif simple_dims['d_model'] != args.d_model:
                     logger.warning(f"Overriding CLI d_model={args.d_model} with value from simple checkpoint: {simple_dims['d_model']}")
                     final_d_model = simple_dims['d_model']
                else:
                     final_d_model = args.d_model # Keep CLI value if consistent
            else:
                 logger.warning("Could not extract d_model from simple checkpoint, using CLI value.")
                 final_d_model = args.d_model

            if simple_dims['num_layers'] is not None:
                 if args.force_config:
                      logger.warning(f"--force-config is set. Ignoring num_layers={simple_dims['num_layers']} from simple checkpoint, using CLI value {args.num_layers}.")
                 elif simple_dims['num_layers'] != args.num_layers:
                      logger.warning(f"Overriding CLI num_layers={args.num_layers} with value from simple checkpoint: {simple_dims['num_layers']}")
                      final_num_layers = simple_dims['num_layers']
                 else:
                      final_num_layers = args.num_layers
            else:
                 logger.warning("Could not extract num_layers from simple checkpoint, using CLI value.")
                 final_num_layers = args.num_layers
        else:
             # If no simple checkpoint, use CLI args for d_model and num_layers
             logger.info("No simple checkpoint loaded. Using CLI values for d_model and num_layers.")
             final_d_model = args.d_model
             final_num_layers = args.num_layers

        # Extract dimensions and determine final num_latent based on Latent checkpoint
        if latent_checkpoint:
             latent_dims = extract_model_dimensions(latent_checkpoint)
             start_step = max(start_step, latent_checkpoint.get('step', 0))

             # Override num_latent ONLY from latent checkpoint if valid
             if latent_dims['num_latent'] is not None:
                  if args.force_config:
                       logger.warning(f"--force-config is set. Ignoring num_latent={latent_dims['num_latent']} from latent checkpoint, using CLI value {args.num_latent}.")
                  elif latent_dims['num_latent'] != args.num_latent:
                       logger.warning(f"Overriding CLI num_latent={args.num_latent} with value from latent checkpoint: {latent_dims['num_latent']}")
                       final_num_latent = latent_dims['num_latent']
                  else:
                       final_num_latent = args.num_latent
             else:
                  logger.warning("Could not extract num_latent from latent checkpoint, using CLI value.")
                  final_num_latent = args.num_latent
             
             # Check for d_model consistency (Latent vs Final determined above)
             if latent_dims['d_model'] is not None and latent_dims['d_model'] != final_d_model:
                 logger.warning(f"d_model mismatch! Latent checkpoint suggests {latent_dims['d_model']}, but using {final_d_model} (from simple checkpoint/CLI).")
             # Check for num_layers consistency
             if latent_dims['num_layers'] is not None and latent_dims['num_layers'] != final_num_layers:
                  logger.warning(f"num_layers mismatch! Latent checkpoint suggests {latent_dims['num_layers']}, but using {final_num_layers} (from simple checkpoint/CLI).")
        else:
             # If no latent checkpoint, use CLI arg for num_latent
             logger.info("No latent checkpoint loaded. Using CLI value for num_latent.")
             final_num_latent = args.num_latent
             
    else:
         logger.info("Resume flag not set. Starting training from scratch.")
         # Use CLI args directly when not resuming
         final_d_model = args.d_model
         final_num_layers = args.num_layers
         final_num_latent = args.num_latent

    # --- End Checkpoint Loading and Parameter Determination ---

    # >>> MOVE DATASET LOADING HERE <<<
    # Dataset configuration
    min_digits = args.min_digits
    max_digits = args.max_digits
    min_val = 10 ** (min_digits - 1)
    max_val = 10**max_digits - 1

    # Create datasets
    logger.info(f"Using train dataset with range {min_val}-{max_val}")
    # Ensure dataset seed is set correctly if needed by MultiplicationDataset
    # MultiplicationDataset.set_fixed_seed(args.seed)
    train_dataset = MultiplicationDataset(
        num_samples=20000, # Consider making this configurable
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )

    logger.info(f"Using val dataset with range {min_val}-{max_val}")
    val_dataset = MultiplicationDataset(
        num_samples=2000, # Consider making this configurable
        split="val",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )
    # >>> END MOVE DATASET LOADING <<<

    # Get vocab size after dataset loading
    vocab_size = train_dataset.vocab_size
    logger.info(f"Using vocabulary size: {vocab_size}")
    
    # Set seed for reproducibility before creating models
    set_seed(args.seed)

    # Log the final parameters that will be used for model creation
    logger.info(f"Final model creation parameters: d_model={final_d_model}, num_layers={final_num_layers}, num_latent={final_num_latent}")

    # Create the models with the determined parameters
    logger.info(f"Creating SimpleTransformer with d_model={final_d_model}, num_layers={final_num_layers}")
    simple_transformer = StableSimpleTransformer(
        vocab_size=vocab_size,
        d_model=final_d_model,
        nhead=8,
        num_layers=final_num_layers,
        dropout=0.25,
    ).to(device)
    
    logger.info(f"Creating LatentTransformer with d_model={final_d_model}, num_layers={final_num_layers}, num_latent={final_num_latent}")
    latent_transformer = StableLatentTransformer(
        vocab_size=vocab_size,
        d_model=final_d_model,
        nhead=8,
        num_layers=final_num_layers,
        num_latent=final_num_latent,
        dropout=0.25,
    ).to(device)

    # Compile models if possible
    # ... (existing compile logic) ...

    # Print model parameter counts
    simple_params_count = sum(p.numel() for p in simple_transformer.parameters())
    latent_params_count = sum(p.numel() for p in latent_transformer.parameters())
    logger.info(f"SimpleTransformer has {simple_params_count:,} parameters")
    logger.info(f"LatentTransformer has {latent_params_count:,} parameters")

    # --- Pass Checkpoints and Start Step to Training Loop --- 
    # Ensure the correct start_step (max of loaded checkpoints) is used
    logger.info(f"Passing start_step={start_step} to training loop.")

    # Flush logs before training
    logger.info("Flushing logs before starting training loop...")
    sys.stdout.flush()
    sys.stderr.flush()

    # Define log directory for TensorBoard
    log_dir = "runs/parallel_comparison"
    
    results = train_models_parallel(
        models={"simple": simple_transformer, "latent": latent_transformer},
        dataset=train_dataset,
        dataset_val=val_dataset,
        vocab_size=vocab_size,
        criterion=None, # Let TrainingLoop handle criterion creation
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=config.base_lr if config else 0.001, # Pass a base LR
        writer=None, # Let TrainingLoop handle writers
        config=config, # Pass config object
        args=args, # Pass args
        models_params={"simple": simple_params_count, "latent": latent_params_count},
        start_step=start_step, # Use the determined start step
        simple_checkpoint=simple_checkpoint, # Pass loaded checkpoint data
        latent_checkpoint=latent_checkpoint, # Pass loaded checkpoint data
        log_dir=log_dir # Pass log directory
    )

    # Flush logs after training
    logger.info("Flushing logs after training loop completion...")
    sys.stdout.flush()
    sys.stderr.flush()

    # Print comparison
    # ... (existing comparison logic) ...

    # Finalize logging
    logger.info("Finalizing logging before exit...")
    logger.complete()

if __name__ == "__main__":
    main()
