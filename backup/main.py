"""
Main entry point for parallel comparison of SimpleTransformer and LatentTransformer models.
"""
import os
import torch
import torch.backends.cudnn
import torch.backends.mps
from torch.utils.data import DataLoader
import time
import argparse
import signal
import shutil
import random
import numpy as np
import logging
import json

from src.Dataset import MultiplicationDataset
from src.Utils import collate_fn
from src.Config import TrainingConfig
from src.Models import StableSimpleTransformer, StableLatentTransformer
from src.Training import set_seed, signal_handler
from src.TrainingLoop import train_models_parallel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("training.log"),  # Also save to file
    ],
)
logger = logging.getLogger(__name__)


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

    # Set seed for reproducibility
    set_seed(42)

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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # Use same batch size for validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,  # More workers for faster data loading
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between batches
    )

    # Initialize models
    d_model = args.d_model
    num_layers = args.num_layers
    num_latent = args.num_latent

    # Create SimpleTransformer (stable version)
    simple_transformer = StableSimpleTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=num_layers,
        dropout=0.25,
    ).to(device)

    # For 3-digit multiplication, we need more latent tokens
    latent_transformer = StableLatentTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=num_layers,
        num_latent=num_latent,  # Increased to handle 3-digit complexity
        dropout=0.25,
        bottleneck_factor=1.0,  # Standard bottleneck constraint
    ).to(device)

    # Enable gradient checkpointing for memory efficiency
    if args.use_checkpointing and device.type == "cuda":
        logger.info("Enabling gradient checkpointing for memory efficiency")
        simple_transformer.encoder.layers.apply(
            lambda m: setattr(m, "_checkpoint", True)
            if hasattr(m, "_checkpoint")
            else None
        )
        simple_transformer.decoder.layers.apply(
            lambda m: setattr(m, "_checkpoint", True)
            if hasattr(m, "_checkpoint")
            else None
        )
        latent_transformer.encoder.layers.apply(
            lambda m: setattr(m, "_checkpoint", True)
            if hasattr(m, "_checkpoint")
            else None
        )
        latent_transformer.decoder.layers.apply(
            lambda m: setattr(m, "_checkpoint", True)
            if hasattr(m, "_checkpoint")
            else None
        )

    # Print parameter counts to help with comparison
    simple_params = sum(p.numel() for p in simple_transformer.parameters())
    latent_params = sum(p.numel() for p in latent_transformer.parameters())
    logger.info(f"\nModel Parameters:")
    logger.info(f"SimpleTransformer: {simple_params:,}")
    logger.info(f"LatentTransformer: {latent_params:,}")
    logger.info(f"Parameter ratio: {latent_params/simple_params:.2f}x")
    logger.info(
        f"Difference: {latent_params-simple_params:,} parameters ({((latent_params-simple_params)/simple_params)*100:.1f}%)"
    )

    # Train both models in parallel
    logger.info("\nTraining both models in parallel...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_d{args.d_model}_l{args.num_layers}_n{args.num_latent}"
    log_dir = f"runs/parallel_comparison/{run_id}"

    # Register the run in the index
    current_config = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_latent": args.num_latent,
        "min_digits": args.min_digits,
        "max_digits": args.max_digits,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "accuracy_weight": args.accuracy_weight,
        "tf_schedule": args.tf_schedule,
        "tf_start_step": args.tf_start_step,
        "use_checkpointing": args.use_checkpointing,
    }

    # Save config to TensorBoard directory for easier recovery
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(current_config, f, indent=2)

    register_run(run_id, current_config)

    # Delete existing runs directory if it exists to avoid TensorBoard confusion
    if os.path.exists(log_dir):
        logger.info(f"Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # Load checkpoints if resuming
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

        if not os.path.exists(simple_checkpoint_path) or not os.path.exists(
            latent_checkpoint_path
        ):
            raise FileNotFoundError(
                f"Checkpoint files not found: {simple_checkpoint_path} or {latent_checkpoint_path}. Cannot resume training."
            )

        # Load checkpoints
        simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
        latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

        # Extract config from checkpoints if available and not already set from run ID
        if not args.run_id or args.force_config:
            # Check if the checkpoint has config data
            if "config" in simple_checkpoint:
                checkpoint_config = simple_checkpoint["config"]
                logger.info("Found config in simple_checkpoint")

                # Only update values not explicitly set by command line or run ID
                if "d_model" in checkpoint_config and not args.force_config:
                    logger.info(
                        f"Using d_model={checkpoint_config['d_model']} from checkpoint config"
                    )
                    args.d_model = checkpoint_config["d_model"]

                if "num_layers" in checkpoint_config and not args.force_config:
                    logger.info(
                        f"Using num_layers={checkpoint_config['num_layers']} from checkpoint config"
                    )
                    args.num_layers = checkpoint_config["num_layers"]

            if "config" in latent_checkpoint:
                checkpoint_config = latent_checkpoint["config"]
                logger.info("Found config in latent_checkpoint")

                # Only update latent-specific values not explicitly set
                if "num_latent" in checkpoint_config and not args.force_config:
                    logger.info(
                        f"Using num_latent={checkpoint_config['num_latent']} from checkpoint config"
                    )
                    args.num_latent = checkpoint_config["num_latent"]

        # Fix state dict keys by removing _orig_mod prefix
        def fix_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict["model_state_dict"].items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
                else:
                    new_state_dict[k] = v
            return new_state_dict

        # Load model states with fixed keys
        simple_transformer.load_state_dict(fix_state_dict(simple_checkpoint))
        latent_transformer.load_state_dict(fix_state_dict(latent_checkpoint))

        # Get steps from checkpoints
        simple_step = simple_checkpoint.get("step", 0)
        latent_step = latent_checkpoint.get("step", 0)

        # Use the minimum step as starting point to ensure both models train the same amount
        start_step = min(simple_step, latent_step)

        # If steps are unequal, log a warning
        if simple_step != latent_step:
            logger.warning(
                f"Model steps are unequal: SimpleTransformer: {simple_step}, LatentTransformer: {latent_step}"
            )
            logger.warning(
                f"Using minimum step {start_step} for resuming. This may cause imbalanced training."
            )

        # Calculate continuation steps needed when resuming
        continuation_steps = 0
        if start_step > 0:
            logger.info(f"Calculating continuation steps for resumed training")
            logger.info(
                f"Original max_steps: {args.max_steps}, start_step: {start_step}"
            )
            if args.max_steps <= start_step:
                continuation_steps = args.max_steps  # Run for the full max_steps again
                logger.info(
                    f"Setting continuation_steps to {continuation_steps} to allow full additional training"
                )
            else:
                logger.info(
                    f"No continuation steps needed, continuing to target {args.max_steps}"
                )

        logger.info(f"Resuming from step {start_step}/{args.max_steps}")
    else:
        start_step = 0
        simple_checkpoint = None
        latent_checkpoint = None
        continuation_steps = 0

    results = train_models_parallel(
        simple_model=simple_transformer,
        latent_model=latent_transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        log_dir=log_dir,
        dataset=train_dataset,
        max_steps=args.max_steps,
        accuracy_weight=args.accuracy_weight,
        tf_schedule=args.tf_schedule,
        tf_start_step=args.tf_start_step,
        args=args,
        models_params={"simple": simple_params, "latent": latent_params},
        start_step=start_step,
        simple_checkpoint=simple_checkpoint,
        latent_checkpoint=latent_checkpoint,
        continuation_steps=continuation_steps,
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
