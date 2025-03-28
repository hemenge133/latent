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
logger.add("training.log", rotation="100 MB")  # Add file handler with rotation


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
    """
    Extract important model dimensions from a checkpoint file.
    This is used to ensure that models are created with the correct dimensions
    when loading from checkpoints, regardless of command-line defaults.
    """
    dimensions = {
        'd_model': None,       # Will be set later if detected
        'num_layers': None,    # Will be set later if detected
        'vocab_size': None,    # Will be set later if detected
        'num_latent': None,    # Will be set later if detected (latent models only)
        'max_len': 20          # Default max sequence length
    }
    
    # Determine if we have a model_state_dict or direct state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Look for config in the checkpoint first
    if 'config' in checkpoint:
        config = checkpoint['config']
        # Get d_model from config
        if 'd_model' in config:
            dimensions['d_model'] = config['d_model']
        
        # Get num_layers from config
        if 'num_layers' in config:
            dimensions['num_layers'] = config['num_layers']
            
        # Get vocab_size from config
        if 'vocab_size' in config:
            dimensions['vocab_size'] = config['vocab_size']
            
        # Get num_latent from config (for latent models)
        if 'num_latent' in config:
            dimensions['num_latent'] = config['num_latent']
            
        # Get max_len from config if available
        if 'max_len' in config:
            dimensions['max_len'] = config['max_len']
    
    # Try to extract dimensions from state_dict structure if not found in config
    # Check embedding weights for vocab size and d_model
    for k, v in state_dict.items():
        if k.endswith("embed.weight") or k == "embed.weight":
            if dimensions['vocab_size'] is None:
                dimensions['vocab_size'] = v.size(0)
            if dimensions['d_model'] is None:
                dimensions['d_model'] = v.size(1)
            break
    
    # Check positional encoder for max_len and d_model
    for k, v in state_dict.items():
        if k.endswith("pos_encoder") or k == "pos_encoder":
            dimensions['max_len'] = v.size(0)
            if dimensions['d_model'] is None:
                dimensions['d_model'] = v.size(1)
            break
    
    # Extract layer numbers from keys if not found in config
    if dimensions['num_layers'] is None:
        layer_indices = []
        for key in state_dict.keys():
            if "encoder.layers." in key:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i+1 < len(parts):
                        try:
                            layer_idx = int(parts[i+1])
                            layer_indices.append(layer_idx)
                        except ValueError:
                            continue
        
        if layer_indices:
            max_layer_idx = max(layer_indices)
            dimensions['num_layers'] = max_layer_idx + 1  # +1 because indexing starts at 0
    
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

    # Setup device based on arguments or auto-detect
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif args.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Display info about accuracy-aware loss if enabled
    if args.accuracy_weight > 0:
        logger.info(f"Using accuracy-aware loss with weight: {args.accuracy_weight}")
        logger.info("This reduces loss for correct token predictions during training")

    # Dataset configuration
    min_digits = args.min_digits
    max_digits = args.max_digits
    min_val = 10 ** (min_digits - 1)
    max_val = 10**max_digits - 1

    # Import run_management utilities
    try:
        from src.RunManagement import get_run_info, get_run_config_from_id, register_run
    except ImportError:
        logger.warning(
            "src/RunManagement.py not found, run ID functionality will be limited"
        )
        get_run_info = lambda run_id: None
        get_run_config_from_id = lambda run_id: {}
        register_run = lambda run_id, config: None

    # If resuming from a run ID, get the config from that run
    run_config = {}
    if args.run_id:
        logger.info(f"Attempting to resume from run ID: {args.run_id}")
        run_info = get_run_info(args.run_id)

        if run_info:
            logger.info(f"Found run: {run_info['id']}")
            run_config = run_info.get("config", {})
        else:
            # Try to extract config from the run ID format
            logger.warning(
                f"Run ID {args.run_id} not found in index, trying to parse from ID format"
            )
            run_config = get_run_config_from_id(args.run_id)

        # Apply config values if not forcing command-line config
        if run_config and not args.force_config:
            if "seed" in run_config:
                logger.info(
                    f"Using seed={run_config['seed']} from run config (override {args.seed})"
                )
                args.seed = run_config["seed"]
                # Re-seed with the resumed seed for consistency
                set_seed(args.seed)
                MultiplicationDataset.set_fixed_seed(args.seed)

            if "d_model" in run_config:
                logger.info(
                    f"Using d_model={run_config['d_model']} from run config (override {args.d_model})"
                )
                args.d_model = run_config["d_model"]

            if "num_layers" in run_config:
                logger.info(
                    f"Using num_layers={run_config['num_layers']} from run config (override {args.num_layers})"
                )
                args.num_layers = run_config["num_layers"]

            if "num_latent" in run_config:
                logger.info(
                    f"Using num_latent={run_config['num_latent']} from run config (override {args.num_latent})"
                )
                args.num_latent = run_config["num_latent"]

            if "min_digits" in run_config:
                logger.info(
                    f"Using min_digits={run_config['min_digits']} from run config (override {args.min_digits})"
                )
                args.min_digits = run_config["min_digits"]
                min_digits = args.min_digits
                min_val = 10 ** (min_digits - 1)

            if "max_digits" in run_config:
                logger.info(
                    f"Using max_digits={run_config['max_digits']} from run config (override {args.max_digits})"
                )
                args.max_digits = run_config["max_digits"]
                max_digits = args.max_digits
                max_val = 10**max_digits - 1

    # If resuming training, check checkpoints FIRST before creating models to get the correct dimensions
    if args.resume:
        # Default checkpoint paths
        simple_checkpoint_path = (
            "checkpoints/simpletransformer/simpletransformer_latest.pt"
        )
        latent_checkpoint_path = (
            "checkpoints/latenttransformer/latenttransformer_latest.pt"
        )

        # If run ID is specified, check if it has different checkpoint paths
        if args.run_id:
            run_info = get_run_info(args.run_id)
            if run_info and "checkpoint_paths" in run_info:
                checkpoint_paths = run_info["checkpoint_paths"]
                simple_checkpoint_path = checkpoint_paths.get(
                    "simple_latest", simple_checkpoint_path
                )
                latent_checkpoint_path = checkpoint_paths.get(
                    "latent_latest", latent_checkpoint_path
                )
                logger.info(f"Using checkpoint paths from run {args.run_id}")
                
                # Also get the batch size if available to ensure consistency
                if "batch_size" in run_info:
                    logger.info(f"Using batch size {run_info['batch_size']} from run to ensure consistency")
                    args.batch_size = run_info["batch_size"]
                    batch_size = args.batch_size

        if not os.path.exists(simple_checkpoint_path) or not os.path.exists(
            latent_checkpoint_path
        ):
            raise FileNotFoundError(
                f"Checkpoint files not found: {simple_checkpoint_path} or {latent_checkpoint_path}. Cannot resume training."
            )

        try:
            # Load checkpoints to extract model dimensions
            logger.info("Loading checkpoints to extract model dimensions")
            simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
            latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

            # Extract model dimensions from checkpoints using our improved function
            simple_dimensions = extract_model_dimensions(simple_checkpoint)
            latent_dimensions = extract_model_dimensions(latent_checkpoint)

            # Log the extracted dimensions for debugging
            logger.info(f"Dimensions from SimpleTransformer checkpoint: {simple_dimensions}")
            logger.info(f"Dimensions from LatentTransformer checkpoint: {latent_dimensions}")

            # Update dimensions from checkpoints if they are valid
            if simple_dimensions['d_model'] is not None:
                logger.info(f"Setting d_model to {simple_dimensions['d_model']} from checkpoint")
                args.d_model = simple_dimensions['d_model']
            
            if simple_dimensions['num_layers'] is not None:
                logger.info(f"Setting num_layers to {simple_dimensions['num_layers']} from checkpoint")
                args.num_layers = simple_dimensions['num_layers']

            if latent_dimensions['num_latent'] is not None:
                logger.info(f"Setting num_latent to {latent_dimensions['num_latent']} from checkpoint")
                args.num_latent = latent_dimensions['num_latent']

            # If vocab_size is found, use it for both models
            checkpoint_vocab_size = None
            if simple_dimensions['vocab_size'] is not None:
                checkpoint_vocab_size = simple_dimensions['vocab_size']
                logger.info(f"Using vocab_size={checkpoint_vocab_size} from checkpoint")
            
            # Store checkpoint data for later use
            simple_start_step = simple_checkpoint.get("step", 0)
            latent_start_step = latent_checkpoint.get("step", 0)

        except Exception as e:
            logger.error(f"Error loading checkpoints for dimension check: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    else:
        # Initialize to 0 if not resuming
        simple_start_step = 0
        latent_start_step = 0
        simple_checkpoint = None
        latent_checkpoint = None
        checkpoint_vocab_size = None

    # Update training configuration
    config.min_digits = min_digits
    config.max_digits = max_digits

    # Create datasets
    logger.info(f"Using train dataset with range {min_val}-{max_val}")
    train_dataset = MultiplicationDataset(
        num_samples=20000,  # Increased dataset size further
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )

    logger.info(f"Using val dataset with range {min_val}-{max_val}")
    val_dataset = MultiplicationDataset(
        num_samples=2000,  # Increased validation set
        split="val",
        split_ratio=(0.8, 0.1, 0.1),
        min_value=min_val,
        max_value=max_val,
    )

    # Create data loaders with larger batch size to fully utilize GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # Use batch size from command-line arguments
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,  # More workers for faster data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between batches
        prefetch_factor=3,  # Prefetch more batches
        worker_init_fn=lambda worker_id: np.random.seed(
            args.seed + worker_id
        ),  # Set worker seeds
        generator=torch.Generator().manual_seed(
            args.seed
        ),  # Ensure consistent shuffling
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # Use same batch size for validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,  # More workers for faster data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between batches
        worker_init_fn=lambda worker_id: np.random.seed(
            args.seed + worker_id
        ),  # Set worker seeds
        generator=torch.Generator().manual_seed(
            args.seed
        ),  # Ensure consistent shuffling
    )

    # Create model configuration dictionary
    model_config = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_latent": args.num_latent,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }

    # Create base models
    # Use the vocabulary size from the checkpoint if available, otherwise use default 12
    vocab_size = checkpoint_vocab_size if checkpoint_vocab_size is not None else 12
    logger.info(f"Using vocabulary size: {vocab_size}")
    
    # Create SimpleTransformer (stable version)
    simple_transformer = StableSimpleTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        dropout=0.25,
    )
    
    # Create LatentTransformer (stable version)
    latent_transformer = StableLatentTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        num_latent=args.num_latent,
        dropout=0.25,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if args.use_checkpointing and torch.cuda.is_available():
        logger.info("Enabling gradient checkpointing for memory efficiency")
        if hasattr(simple_transformer.encoder, 'layers'):
            simple_transformer.encoder.layers.apply(
                lambda m: setattr(m, "_checkpoint", True)
                if hasattr(m, "_checkpoint")
                else None
            )
        if hasattr(simple_transformer.decoder, 'layers'):
            simple_transformer.decoder.layers.apply(
                lambda m: setattr(m, "_checkpoint", True)
                if hasattr(m, "_checkpoint")
                else None
            )
        if hasattr(latent_transformer.encoder, 'layers'):
            latent_transformer.encoder.layers.apply(
                lambda m: setattr(m, "_checkpoint", True)
                if hasattr(m, "_checkpoint")
                else None
            )
        if hasattr(latent_transformer.decoder, 'layers'):
            latent_transformer.decoder.layers.apply(
                lambda m: setattr(m, "_checkpoint", True)
                if hasattr(m, "_checkpoint")
                else None
            )
    
    # Compile for performance (PyTorch 2.0 feature)
    if torch.cuda.is_available() and not getattr(args, "disable_compile", False):
        try:
            simple_transformer = torch.compile(simple_transformer)
            latent_transformer = torch.compile(latent_transformer)
            logger.info("Models compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"Model compilation failed: {str(e)}")
    
    # Move models to device
    simple_transformer = simple_transformer.to(device)
    latent_transformer = latent_transformer.to(device)

    # Count parameters
    simple_params = sum(p.numel() for p in simple_transformer.parameters())
    latent_params = sum(p.numel() for p in latent_transformer.parameters())

    logger.info("\nModel Parameters:")
    logger.info(f"SimpleTransformer: {simple_params:,}")
    logger.info(f"LatentTransformer: {latent_params:,}")
    logger.info(f"Parameter ratio: {latent_params/simple_params:.2f}x")
    logger.info(
        f"Difference: {latent_params-simple_params:,} parameters ({(latent_params-simple_params)/simple_params:.1%})"
    )

    # Register run with run management
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (
        f"{current_timestamp}_d{args.d_model}_l{args.num_layers}_n{args.num_latent}"
    )
    checkpoint_paths = {
        "simple_latest": f"checkpoints/simpletransformer/simpletransformer_latest.pt",
        "simple_best": f"checkpoints/simpletransformer/simpletransformer_best.pt",
        "latent_latest": f"checkpoints/latenttransformer/latenttransformer_latest.pt",
        "latent_best": f"checkpoints/latenttransformer/latenttransformer_best.pt",
    }

    # Register the current run with all the configuration
    register_run(
        run_id,
        {
            "id": run_id,
            "timestamp": current_timestamp,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_latent": args.num_latent,
            "min_digits": min_digits,
            "max_digits": max_digits,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "checkpoint_paths": checkpoint_paths,
        },
    )
    logger.info(f"\nTraining both models in parallel...")

    # For resumed training, always use the original log directory to maintain continuous TensorBoard logs
    if args.resume and args.run_id:
        # Create a consistent directory path based on the run_id
        log_dir = os.path.join("runs/parallel_comparison", args.run_id)
        logger.info(f"Using consistent log directory for resumed training: {log_dir}")
        
        # Ensure it exists
        os.makedirs(log_dir, exist_ok=True)
    else:
        # Original format for new runs
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(
            "runs/parallel_comparison", 
            f"{timestamp}_d{args.d_model}_l{args.num_layers}_n{args.num_latent}"
        )
    
    # Register the run for future reference
    run_id = os.path.basename(log_dir)
    register_run(run_id, {
        "id": run_id, 
        "log_dir": log_dir,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_latent": args.num_latent,
        "seed": args.seed,
        "checkpoint_paths": {
            "simple_latest": f"checkpoints/simpletransformer/simpletransformer_latest.pt",
            "latent_latest": f"checkpoints/latenttransformer/latenttransformer_latest.pt",
        }
    })

    # Step counting for resumption
    start_step = 0
    
    if args.resume and simple_checkpoint and 'step' in simple_checkpoint:
        start_step = simple_checkpoint['step']
    
    # Load checkpoints if available
    if args.resume:
        # Load SimpleTransformer checkpoint
        if os.path.exists(simple_checkpoint_path):
            logger.info(f"Loading SimpleTransformer checkpoint from {simple_checkpoint_path}")
            simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
            
            # Apply the checkpoint
            try:
                simple_transformer.load_state_dict(simple_checkpoint['model_state_dict'], strict=False)
                logger.info(f"Loaded SimpleTransformer model weights")
            except RuntimeError as e:
                logger.error(f"Error loading SimpleTransformer weights: {str(e)}")
                # Error might be due to dimension mismatch which shouldn't happen anymore
                # since we extract dimensions before creating models, but just in case:
                logger.error("This might be due to dimension mismatch. Check model configuration.")
                
                # As a safety measure, extract dimensions again and recreate if needed
                dimensions = extract_model_dimensions(simple_checkpoint)
                if (dimensions['d_model'] is not None and dimensions['d_model'] != args.d_model) or \
                   (dimensions['num_layers'] is not None and dimensions['num_layers'] != args.num_layers):
                    logger.warning(f"Recreating model with dimensions from checkpoint: {dimensions}")
                    
                    # Need to modify Models.py to accept max_len parameter, or we need to hack it
                    # after creation to match the checkpoint's max_len
                    simple_transformer = StableSimpleTransformer(
                        vocab_size=vocab_size,
                        d_model=dimensions['d_model'] or args.d_model,
                        nhead=8,
                        num_layers=dimensions['num_layers'] or args.num_layers,
                        dropout=0.25,
                    )
                    
                    # If the max_len in the checkpoint is different from the default in the model,
                    # we need to replace the pos_encoder Parameter with one of the correct size
                    if 'pos_encoder' in simple_checkpoint['model_state_dict']:
                        pos_encoder_tensor = simple_checkpoint['model_state_dict']['pos_encoder']
                        if simple_transformer.pos_encoder.shape != pos_encoder_tensor.shape:
                            logger.warning(f"Fixing pos_encoder shape mismatch: model has {simple_transformer.pos_encoder.shape} but checkpoint has {pos_encoder_tensor.shape}")
                            # Replace the pos_encoder with one that matches the checkpoint
                            simple_transformer.max_len = pos_encoder_tensor.shape[0]
                            simple_transformer.pos_encoder = nn.Parameter(torch.zeros_like(pos_encoder_tensor))
                        
                        # Move model to device
                        simple_transformer = simple_transformer.to(device)
                        
                        # Try loading again
                        simple_transformer.load_state_dict(simple_checkpoint['model_state_dict'], strict=False)
                        logger.info("Successfully loaded weights after model recreation")
                    else:
                        sys.exit(1)
        
        # Load LatentTransformer checkpoint
        if os.path.exists(latent_checkpoint_path):
            logger.info(f"Loading LatentTransformer checkpoint from {latent_checkpoint_path}")
            latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)
            
            # Apply the checkpoint
            try:
                latent_transformer.load_state_dict(latent_checkpoint['model_state_dict'], strict=False)
                logger.info(f"Loaded LatentTransformer model weights")
            except RuntimeError as e:
                logger.error(f"Error loading LatentTransformer weights: {str(e)}")
                
                # As a safety measure, extract dimensions again and recreate if needed
                dimensions = extract_model_dimensions(latent_checkpoint)
                if (dimensions['d_model'] is not None and dimensions['d_model'] != args.d_model) or \
                   (dimensions['num_layers'] is not None and dimensions['num_layers'] != args.num_layers) or \
                   (dimensions['num_latent'] is not None and dimensions['num_latent'] != args.num_latent):
                    logger.warning(f"Recreating model with dimensions from checkpoint: {dimensions}")
                    latent_transformer = StableLatentTransformer(
                        vocab_size=vocab_size,
                        d_model=dimensions['d_model'] or args.d_model,
                        nhead=8,
                        num_layers=dimensions['num_layers'] or args.num_layers,
                        num_latent=dimensions['num_latent'] or args.num_latent,
                        dropout=0.25,
                    )
                    
                    # If the max_len in the checkpoint is different from the default in the model,
                    # we need to replace the pos_encoder Parameter with one of the correct size
                    if 'pos_encoder' in latent_checkpoint['model_state_dict']:
                        pos_encoder_tensor = latent_checkpoint['model_state_dict']['pos_encoder']
                        if latent_transformer.pos_encoder.shape != pos_encoder_tensor.shape:
                            logger.warning(f"Fixing pos_encoder shape mismatch: model has {latent_transformer.pos_encoder.shape} but checkpoint has {pos_encoder_tensor.shape}")
                            # Replace the pos_encoder with one that matches the checkpoint
                            latent_transformer.max_len = pos_encoder_tensor.shape[0]
                            latent_transformer.pos_encoder = nn.Parameter(torch.zeros_like(pos_encoder_tensor))
                        
                        # Move model to device
                        latent_transformer = latent_transformer.to(device)
                        
                        # Try loading again
                        latent_transformer.load_state_dict(latent_checkpoint['model_state_dict'], strict=False)
                        logger.info("Successfully loaded weights after model recreation")
                    else:
                        sys.exit(1)

    results = train_models_parallel(
        models={"simple": simple_transformer, "latent": latent_transformer},
        dataset=train_dataset,
        dataset_val=val_dataset,
        vocab_size=vocab_size,
        criterion=None,
        device=device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=None,
        writer=None,
        config=config,
        args=args,
        models_params={"simple": simple_params, "latent": latent_params},
        start_step=start_step,
        simple_checkpoint=simple_checkpoint,
        latent_checkpoint=latent_checkpoint,
        log_dir=log_dir
    )

    # Print comparison
    logger.info("\nFinal Comparison:")
    logger.info("=" * 50)

    simple_results = results["simple"]
    latent_results = results["latent"]
    training_time = results["training_time"]

    logger.info(f"Training time: {training_time:.2f}s")
    logger.info(
        f"SimpleTransformer: {simple_results['loss']:.6f} loss, {simple_results['sequence_accuracy']:.2%} sequence accuracy, {simple_results['digit_accuracy']:.2%} digit accuracy, {simple_results['params']:,} parameters"
    )
    logger.info(
        f"LatentTransformer: {latent_results['loss']:.6f} loss, {latent_results['sequence_accuracy']:.2%} sequence accuracy, {latent_results['digit_accuracy']:.2%} digit accuracy, {latent_results['params']:,} parameters"
    )

    # Calculate efficiency metrics if possible
    if (
        simple_results["loss"] != float("inf")
        and latent_results["loss"] != float("inf")
        and not np.isnan(simple_results["loss"])
        and not np.isnan(latent_results["loss"])
        and simple_results["loss"] > 0
        and latent_results["loss"] > 0
    ):
        # Parameter efficiency (lower is better: loss * num_params)
        param_efficiency_simple = simple_results["loss"] * simple_results["params"]
        param_efficiency_latent = latent_results["loss"] * latent_results["params"]

        # Accuracy efficiency (higher is better: accuracy / num_params)
        acc_param_efficiency_simple = (
            simple_results["sequence_accuracy"] / simple_results["params"]
            if simple_results["params"] > 0 and simple_results["sequence_accuracy"] > 0
            else 0
        )
        acc_param_efficiency_latent = (
            latent_results["sequence_accuracy"] / latent_results["params"]
            if latent_results["params"] > 0 and latent_results["sequence_accuracy"] > 0
            else 0
        )

        logger.info("\nEfficiency Metrics:")

        # Prevent division by zero
        if param_efficiency_simple == 0 or param_efficiency_latent == 0:
            logger.info(
                "Cannot calculate loss efficiency metrics: at least one model has 0 loss"
            )
        else:
            if param_efficiency_latent < param_efficiency_simple:
                ratio = param_efficiency_simple / param_efficiency_latent
                logger.info(
                    f"LatentTransformer is {ratio:.2f}x more parameter-efficient (loss*params)"
                )
            else:
                ratio = param_efficiency_latent / param_efficiency_simple
                logger.info(
                    f"SimpleTransformer is {ratio:.2f}x more parameter-efficient (loss*params)"
                )

        # Only show accuracy efficiency if both models have non-zero accuracy
        if (
            simple_results["sequence_accuracy"] > 0
            and latent_results["sequence_accuracy"] > 0
        ):
            if acc_param_efficiency_simple == 0 or acc_param_efficiency_latent == 0:
                logger.info(
                    "Cannot calculate accuracy efficiency metrics: efficiency calculation resulted in zero"
                )
            else:
                if acc_param_efficiency_latent > acc_param_efficiency_simple:
                    ratio = acc_param_efficiency_latent / acc_param_efficiency_simple
                    logger.info(
                        f"LatentTransformer is {ratio:.2f}x more accuracy-per-parameter efficient"
                    )
                else:
                    ratio = acc_param_efficiency_simple / acc_param_efficiency_latent
                    logger.info(
                        f"SimpleTransformer is {ratio:.2f}x more accuracy-per-parameter efficient"
                    )
        else:
            logger.info(
                "Cannot calculate accuracy efficiency metrics: at least one model has 0% accuracy"
            )
    else:
        logger.info(
            "Cannot calculate efficiency metrics due to invalid or zero loss values"
        )

    logger.info("\nTo view parallel training curves, run:")
    logger.info(f"tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
