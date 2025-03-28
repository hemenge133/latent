# Import the original module to avoid duplication
from src.TrainingLoop import *
import logging
import torch.nn as nn


# Override the train_models_parallel function
def debug_train_models_parallel(*args, **kwargs):
    # Get the logger
    logger = logging.getLogger(__name__)

    # Extract checkpoints from kwargs
    simple_checkpoint = kwargs.get("simple_checkpoint")
    latent_checkpoint = kwargs.get("latent_checkpoint")
    start_step = kwargs.get("start_step", 0)

    # Print debug info
    logger.info("=" * 50)
    logger.info("DEBUG RESUME INFORMATION")
    logger.info("=" * 50)
    logger.info(f"start_step: {start_step}")

    if simple_checkpoint:
        logger.info("Simple checkpoint info:")
        logger.info(f"  step: {simple_checkpoint.get('step', 'N/A')}")
        logger.info(f"  val_loss: {simple_checkpoint.get('val_loss', 'N/A')}")
        logger.info(
            f"  has_model_state: {'Yes' if 'model_state_dict' in simple_checkpoint else 'No'}"
        )
        logger.info(
            f"  has_optimizer_state: {'Yes' if 'optimizer_state_dict' in simple_checkpoint else 'No'}"
        )
        logger.info(
            f"  has_scheduler_state: {'Yes' if 'scheduler_state_dict' in simple_checkpoint else 'No'}"
        )
        logger.info(
            f"  seed: {simple_checkpoint.get('seed', simple_checkpoint.get('config', {}).get('seed', 'Not found'))}"
        )

        # Print first few model parameters to verify they're loaded correctly
        if "model_state_dict" in simple_checkpoint:
            logger.info("  Sample model state values:")
            for i, (name, param) in enumerate(
                simple_checkpoint["model_state_dict"].items()
            ):
                if i < 3:  # Just show first 3 parameters
                    if isinstance(param, torch.Tensor):
                        logger.info(
                            f"    {name}: shape={param.shape}, mean={param.float().mean().item():.6f}, std={param.float().std().item():.6f}"
                        )
                else:
                    break
    else:
        logger.info("No simple checkpoint provided")

    if latent_checkpoint:
        logger.info("Latent checkpoint info:")
        logger.info(f"  step: {latent_checkpoint.get('step', 'N/A')}")
        logger.info(f"  val_loss: {latent_checkpoint.get('val_loss', 'N/A')}")
        logger.info(
            f"  has_model_state: {'Yes' if 'model_state_dict' in latent_checkpoint else 'No'}"
        )
        logger.info(
            f"  has_optimizer_state: {'Yes' if 'optimizer_state_dict' in latent_checkpoint else 'No'}"
        )
        logger.info(
            f"  has_scheduler_state: {'Yes' if 'scheduler_state_dict' in latent_checkpoint else 'No'}"
        )
        logger.info(
            f"  seed: {latent_checkpoint.get('seed', latent_checkpoint.get('config', {}).get('seed', 'Not found'))}"
        )

        # Print first few model parameters
        if "model_state_dict" in latent_checkpoint:
            logger.info("  Sample model state values:")
            for i, (name, param) in enumerate(
                latent_checkpoint["model_state_dict"].items()
            ):
                if i < 3:  # Just show first 3 parameters
                    if isinstance(param, torch.Tensor):
                        logger.info(
                            f"    {name}: shape={param.shape}, mean={param.float().mean().item():.6f}, std={param.float().std().item():.6f}"
                        )
                else:
                    break
    else:
        logger.info("No latent checkpoint provided")

    # Log information about the dataset state
    try:
        from src.Dataset import MultiplicationDataset

        logger.info(f"Dataset fixed seed: {MultiplicationDataset._fixed_seed}")
        logger.info(
            f"Number of cached problem sets: {len(MultiplicationDataset._all_problems)}"
        )
    except Exception as e:
        logger.info(f"Error checking dataset state: {e}")

    logger.info("Calling original train_models_parallel function...")
    logger.info("=" * 50)

    # Call the original function with debug info in first batch
    result = train_models_parallel(*args, **kwargs)

    return result
