"""
Main training loop for parallel model training.
"""
import os
import sys
import math
import time
import random
import traceback
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.SummaryWriter import SummaryWriter
from tqdm import tqdm
import itertools
from contextlib import nullcontext
from transformers import get_cosine_schedule_with_warmup
from src.Metrics import evaluate
from src.Losses import SequenceAccuracyLoss
from src.Training import generate_evaluation_examples, setup_models_training, set_seed

# Get the logger
logger = logging.getLogger(__name__)

# Constants for loss monitoring and emergency handling
LOSS_MAX_THRESHOLD = {
    "simple": 20.0,  # Maximum loss allowed for SimpleTransformer before intervention
    "latent": 25.0,  # Maximum loss allowed for LatentTransformer before intervention
}

LR_EMERGENCY_FACTOR = {
    "simple": 0.5,  # LR reduction factor when SimpleTransformer loss explodes
    "latent": 0.5,  # LR reduction factor when LatentTransformer loss explodes
}

# Max gradient norm for clipping
MAX_GRAD_NORM = {
    "simple": 0.5,  # Base norm for SimpleTransformer
    "latent": 0.4,  # Base norm for LatentTransformer (lower for stability)
}

class Config:
    """Default configuration if none is provided"""
    def __init__(self):
        self.base_lr = 3e-4
        self.warmup_steps = 100
        self.weight_decay = 0.01

def collate_fn(batch):
    """Collate function for DataLoader that handles variable length sequences"""
    # Separate input sequences, targets, and their lengths
    inp_seqs = [item[0] for item in batch]
    tgt_seqs = [item[1] for item in batch]
    
    # Convert to tensors if they aren't already
    if not isinstance(inp_seqs[0], torch.Tensor):
        inp_seqs = [torch.tensor(seq) for seq in inp_seqs]
    if not isinstance(tgt_seqs[0], torch.Tensor):
        tgt_seqs = [torch.tensor(seq) for seq in tgt_seqs]
    
    # Get lengths
    inp_lens = [len(seq) for seq in inp_seqs]
    tgt_lens = [len(seq) for seq in tgt_seqs]
    
    # Find maximum lengths for padding
    max_inp_len = max(inp_lens)
    max_tgt_len = max(tgt_lens)
    
    # Pad sequences to the maximum length
    inp_seqs_padded = []
    tgt_seqs_padded = []
    
    for inp_seq, tgt_seq in zip(inp_seqs, tgt_seqs):
        # Pad input sequence
        if len(inp_seq) < max_inp_len:
            padding = torch.zeros(max_inp_len - len(inp_seq), dtype=inp_seq.dtype)
            inp_seq = torch.cat([inp_seq, padding])
        inp_seqs_padded.append(inp_seq)
        
        # Pad target sequence
        if len(tgt_seq) < max_tgt_len:
            padding = torch.zeros(max_tgt_len - len(tgt_seq), dtype=tgt_seq.dtype)
            tgt_seq = torch.cat([tgt_seq, padding])
        tgt_seqs_padded.append(tgt_seq)
    
    # Stack padded sequences into batches
    inp_tensor = torch.stack(inp_seqs_padded)
    tgt_tensor = torch.stack(tgt_seqs_padded)
    inp_lens = torch.tensor(inp_lens)
    tgt_lens = torch.tensor(tgt_lens)
    
    return inp_tensor, tgt_tensor, inp_lens, tgt_lens

def get_grad_clip_norm(model_type, lr, step):
    """Get gradient clipping norm based on model type, learning rate and step"""
    if model_type == 'simple':
        base_clip = 1.0
    else:
        base_clip = 0.8  # Slightly more aggressive for latent model
    
    # Scale with learning rate
    lr_scale = min(1.0, lr / 3e-4)
    
    # Reduce clipping as training progresses (allow more exploration)
    step_scale = min(1.0, 1000 / (step + 100))
    
    return base_clip * lr_scale * step_scale

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

def train_models_parallel(
    models,
    dataset,
    dataset_val,
    vocab_size,
    criterion,
    device,
    max_steps,
    batch_size,
    learning_rate,
    writer=None,
    config=None,
    args=None,
    models_params=None,
    start_step=0,
    simple_checkpoint=None,
    latent_checkpoint=None,
    log_dir="runs/parallel_comparison"
):
    """
    Train both models in parallel and compare performance
    
    Parameters:
    models (dict): Dictionary with 'simple' and 'latent' keys containing transformer models
    dataset (ArithmeticDataset): Dataset for training
    dataset_val (ArithmeticDataset): Dataset for validation
    vocab_size (int): Size of the vocabulary
    criterion (callable): Loss function
    device (torch.device): Device to train on
    max_steps (int): Maximum number of training steps (total, not additional)
    batch_size (int): Batch size
    learning_rate (float): Learning rate
    writer (SummaryWriter, optional): TensorBoard writer
    config (Config, optional): Configuration object
    args (argparse.Namespace, optional): Command line arguments
    models_params (dict, optional): Dictionary with 'simple' and 'latent' keys containing parameter counts
    start_step (int, optional): Starting step for resumed training
    simple_checkpoint (dict, optional): Checkpoint for SimpleTransformer
    latent_checkpoint (dict, optional): Checkpoint for LatentTransformer
    log_dir (str, optional): Directory for saving logs and checkpoints
    """
    # Create log directories for each model
    log_dir_simple = os.path.join(log_dir, "simple")
    log_dir_latent = os.path.join(log_dir, "latent")
    os.makedirs(log_dir_simple, exist_ok=True)
    os.makedirs(log_dir_latent, exist_ok=True)
    
    # Setup tensorboard writers if not provided
    simple_writer = writer or SummaryWriter(log_dir=log_dir_simple)
    latent_writer = writer or SummaryWriter(log_dir=log_dir_latent)
    
    # Parse model parameters
    simple_model = models["simple"]
    latent_model = models["latent"]
    simple_params = models_params["simple"] if models_params else sum(p.numel() for p in simple_model.parameters())
    latent_params = models_params["latent"] if models_params else sum(p.numel() for p in latent_model.parameters())
    
    # Set default max steps based on command line arguments
    max_steps_original = max_steps
    
    # Check if max_steps is set in either checkpoint
    if simple_checkpoint is not None and 'max_steps' in simple_checkpoint:
        # If max_steps in checkpoint is larger than specified, use the larger value
        checkpoint_max_steps = simple_checkpoint.get('max_steps')
        if checkpoint_max_steps > max_steps:
            logger.info(f"Found larger max_steps in checkpoint: {checkpoint_max_steps} > {max_steps}")
            logger.info(f"Using larger value from checkpoint to ensure training completion")
            max_steps = checkpoint_max_steps
    
    # Calculate the target steps for each model based on start step and max steps
    simple_start_step = 0
    latent_start_step = 0
    simple_global_step = 0
    latent_global_step = 0
    
    # If resuming training, set global step to match the model step
    if simple_checkpoint is not None and "step" in simple_checkpoint:
        simple_global_step = simple_checkpoint["step"]
        simple_start_step = simple_checkpoint["step"]
        logger.info(f"Setting simple_global_step to {simple_global_step} for TensorBoard continuity")
        logger.info(f"SimpleTransformer resuming from step {simple_start_step}")
        logger.info(f"SimpleTransformer target steps: {max_steps}")
    
    if latent_checkpoint is not None and "step" in latent_checkpoint:
        latent_global_step = latent_checkpoint["step"] 
        latent_start_step = latent_checkpoint["step"]
        logger.info(f"Setting latent_global_step to {latent_global_step} for TensorBoard continuity")
        logger.info(f"LatentTransformer resuming from step {latent_start_step}")
        logger.info(f"LatentTransformer target steps: {max_steps}")
    
    # Calculate steps to train in this run
    simple_target = max(0, max_steps - simple_start_step)
    latent_target = max(0, max_steps - latent_start_step)
    target_step = max(simple_target, latent_target)
    
    logger.info(f"SimpleTransformer will train for {simple_target} more steps")
    logger.info(f"LatentTransformer will train for {latent_target} more steps")
    logger.info(f"Target step for training loop: {target_step}")
    
    # Setup dataset loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        dataset_val if dataset_val else dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Default config if not provided
    if not config:
        from src.config import Config
        config = Config()
        config.base_lr = learning_rate or 3e-4
        config.warmup_steps = 100
        config.weight_decay = 0.01
    
    # SimpleTransformer: Standard Adam optimizer
    simple_optimizer = torch.optim.Adam(
        simple_model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
    )
    
    # LatentTransformer: Also Adam but with slightly lower weight decay for better exploration
    latent_optimizer = torch.optim.Adam(
        latent_model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay * 0.9,  # Slightly reduced
    )
    
    # Create learning rate schedulers 
    simple_scheduler = get_cosine_schedule_with_warmup(
        simple_optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_steps + simple_start_step,  # Add the start_step to account for resumed training
    )
    
    latent_scheduler = get_cosine_schedule_with_warmup(
        latent_optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_steps + latent_start_step,  # Add the start_step to account for resumed training
    )
    
    # Setup learning rate trackers to ensure consistency
    prev_lrs = {'simple': None, 'latent': None}
    
    # If resuming from checkpoints, store initial learning rates
    if simple_checkpoint is not None and 'optimizer_state_dict' in simple_checkpoint:
        prev_lrs['simple'] = simple_optimizer.param_groups[0]['lr']
        logger.info(f"Initial learning rate from checkpoint (SimpleTransformer): {prev_lrs['simple']}")
    
    if latent_checkpoint is not None and 'optimizer_state_dict' in latent_checkpoint:
        prev_lrs['latent'] = latent_optimizer.param_groups[0]['lr']
        logger.info(f"Initial learning rate from checkpoint (LatentTransformer): {prev_lrs['latent']}")
    
    # Compile models if available for performance
    if hasattr(torch, "compile") and args is not None and hasattr(args, 'disable_compile') and not args.disable_compile:
        logger.info("Using torch.compile to optimize models")
        simple_model = torch.compile(simple_model, mode="max-autotune")
        latent_model = torch.compile(latent_model, mode="max-autotune")
    
    # Define constants for loss thresholds
    LOSS_MAX_THRESHOLD = {
        'simple': 5.0,  # SimpleTransformer can handle higher loss
        'latent': 5.0,  # LatentTransformer can be more sensitive
    }
    
    # Define learning rate emergency scale factor
    LR_EMERGENCY_FACTOR = {
        'simple': 0.5,  # Halve the learning rate for SimpleTransformer
        'latent': 0.4,  # More aggressive for LatentTransformer
    }
    
    # Default teacher forcing schedule if config doesn't specify
    tf_schedule = getattr(args, 'tf_schedule', "none")
    tf_start_step = getattr(args, 'tf_start_step', 1000)
    
    # Check if teacher forcing parameters exist in checkpoints and use them instead
    if simple_checkpoint is not None and 'tf_schedule' in simple_checkpoint:
        # Only override if we found teacher forcing data in the checkpoint
        tf_schedule = simple_checkpoint.get('tf_schedule', tf_schedule)
        tf_start_step = simple_checkpoint.get('tf_start_step', tf_start_step)
        logger.info(f"Using teacher forcing parameters from checkpoint: schedule={tf_schedule}, start_step={tf_start_step}")
    elif latent_checkpoint is not None and 'tf_schedule' in latent_checkpoint:
        # Use latent checkpoint if simple checkpoint doesn't have the data
        tf_schedule = latent_checkpoint.get('tf_schedule', tf_schedule)
        tf_start_step = latent_checkpoint.get('tf_start_step', tf_start_step)
        logger.info(f"Using teacher forcing parameters from checkpoint: schedule={tf_schedule}, start_step={tf_start_step}")
    else:
        # Log if we're using args-provided values
        logger.info(f"Using teacher forcing parameters from args: schedule={tf_schedule}, start_step={tf_start_step}")
    
    # Setup models dict
    models_dict = {
        "simple": {
            "name": "SimpleTransformer",
            "model": simple_model,
            "optimizer": simple_optimizer,
            "scheduler": simple_scheduler,
            "criterion": criterion,
            "writer": simple_writer,
            "step": simple_start_step,
            "val_loss": float('inf'),
            "val_sequence_accuracy": 0.0,
            "val_digit_accuracy": 0.0,
            "best_val_loss": float('inf'),
            "recent_losses": [],
            "stability_window": 100,
            "last_explosion_step": -1000,
            "scaler": torch.cuda.amp.GradScaler(),
            "config": config,
            "params": simple_params,
        },
        "latent": {
            "name": "LatentTransformer",
            "model": latent_model,
            "optimizer": latent_optimizer,
            "scheduler": latent_scheduler,
            "criterion": criterion,
            "writer": latent_writer,
            "step": latent_start_step,
            "val_loss": float('inf'),
            "val_sequence_accuracy": 0.0,
            "val_digit_accuracy": 0.0,
            "best_val_loss": float('inf'),
            "recent_losses": [],
            "stability_window": 100,
            "last_explosion_step": -1000,
            "scaler": torch.cuda.amp.GradScaler(),
            "config": config,
            "params": latent_params,
        }
    }
    
    # Check if criterion is None and provide a fallback
    if criterion is None:
        logger.warning("Criterion is None! Creating default SequenceAccuracyLoss as fallback")
        fallback_criterion = SequenceAccuracyLoss()
        models_dict["simple"]["criterion"] = fallback_criterion
        models_dict["latent"]["criterion"] = fallback_criterion
    
    # Load from checkpoints if provided
    if simple_checkpoint is not None:
        logger.info("Loading SimpleTransformer from checkpoint")
        try:
            # Load model state dict
            if 'model_state_dict' in simple_checkpoint:
                # Filter state dict to match model architecture
                filtered_state_dict = filter_state_dict_for_model(
                    simple_checkpoint['model_state_dict'], models_dict["simple"]["model"]
                )
                models_dict["simple"]["model"].load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded filtered SimpleTransformer state dict with {len(filtered_state_dict)} parameters")
            else:
                # Filter state dict to match model architecture 
                filtered_state_dict = filter_state_dict_for_model(
                    simple_checkpoint, models_dict["simple"]["model"]
                )
                models_dict["simple"]["model"].load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded filtered SimpleTransformer state dict")
            
            # Load optimizer state - do this BEFORE setting learning rate
            if 'optimizer_state_dict' in simple_checkpoint:
                # Store current learning rate for safety
                current_lr = simple_optimizer.param_groups[0]['lr']
                
                # Load optimizer state
                simple_optimizer.load_state_dict(simple_checkpoint['optimizer_state_dict'])
                logger.info("Loaded SimpleTransformer optimizer state")
                
                # Get step count 
                step_count = simple_checkpoint.get('step', 0)
                
                # Now set the learning rate consistently
                if 'last_lr' in simple_checkpoint:
                    # Get saved learning rate from checkpoint
                    saved_lr = simple_checkpoint['last_lr']
                    logger.info(f"Using saved learning rate from checkpoint: {saved_lr}")
                    
                    # Apply saved LR to all parameter groups
                    for param_group in simple_optimizer.param_groups:
                        param_group['lr'] = saved_lr
                    
                    # Store for consistency check
                    prev_lrs['simple'] = saved_lr
                else:
                    # If no saved LR found, keep the current one
                    logger.info(f"No learning rate found in checkpoint, keeping current: {current_lr}")
                    
                    # Store for consistency check
                    prev_lrs['simple'] = current_lr
            
            # Load scheduler - do this AFTER setting optimizer learning rate
            if 'scheduler_state_dict' in simple_checkpoint:
                try:
                    simple_scheduler.load_state_dict(simple_checkpoint['scheduler_state_dict'])
                    logger.info("Loaded SimpleTransformer scheduler state")
                except Exception as e:
                    logger.warning(f"Error loading scheduler state dict: {e}")
                    logger.warning("Creating new scheduler with current optimizer state")
                    
                    # Recreate scheduler based on current optimizer state
                    simple_scheduler = get_cosine_schedule_with_warmup(
                        simple_optimizer,
                        num_warmup_steps=config.warmup_steps,
                        num_training_steps=max_steps
                    )
                    
                    # Fast-forward scheduler to current step if resuming
                    if step_count > 0:
                        for _ in range(step_count):
                            simple_scheduler.step()
                        logger.info(f"Fast-forwarded new SimpleTransformer scheduler to step {step_count}")
                    
                    models_dict["simple"]["scheduler"] = simple_scheduler
            
            # Ensure TensorBoard continuity with checkpoint
            ensure_checkpoint_tensorboard_consistency(models_dict["simple"], simple_checkpoint)
            
        except Exception as e:
            logger.error(f"Error loading SimpleTransformer checkpoint: {e}")
            logger.error(traceback.format_exc())
            

    if latent_checkpoint is not None:
        logger.info("Loading LatentTransformer from checkpoint")
        try:
            # Load model state dict
            if 'model_state_dict' in latent_checkpoint:
                # Filter state dict to match model architecture
                filtered_state_dict = filter_state_dict_for_model(
                    latent_checkpoint['model_state_dict'], models_dict["latent"]["model"]
                )
                models_dict["latent"]["model"].load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded filtered LatentTransformer state dict with {len(filtered_state_dict)} parameters")
            else:
                # Filter state dict to match model architecture
                filtered_state_dict = filter_state_dict_for_model(
                    latent_checkpoint, models_dict["latent"]["model"]
                )
                models_dict["latent"]["model"].load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded filtered LatentTransformer state dict")
                
            # Load optimizer state - do this BEFORE setting learning rate  
            if 'optimizer_state_dict' in latent_checkpoint:
                # Store current learning rate for safety
                current_lr = latent_optimizer.param_groups[0]['lr']
                
                # Load optimizer state
                latent_optimizer.load_state_dict(latent_checkpoint['optimizer_state_dict'])
                logger.info("Loaded LatentTransformer optimizer state")
                
                # Get step count
                step_count = latent_checkpoint.get('step', 0)
                
                # Now set the learning rate consistently
                if 'last_lr' in latent_checkpoint:
                    # Get saved learning rate from checkpoint
                    saved_lr = latent_checkpoint['last_lr']
                    logger.info(f"Using saved learning rate from checkpoint: {saved_lr}")
                    
                    # Apply saved LR to all parameter groups
                    for param_group in latent_optimizer.param_groups:
                        param_group['lr'] = saved_lr
                    
                    # Store for consistency check
                    prev_lrs['latent'] = saved_lr
                else:
                    # If no saved LR found, keep the current one
                    logger.info(f"No learning rate found in checkpoint, keeping current: {current_lr}")
                    
                    # Store for consistency check
                    prev_lrs['latent'] = current_lr
            
            # Load scheduler - do this AFTER setting optimizer learning rate
            if 'scheduler_state_dict' in latent_checkpoint:
                try:
                    latent_scheduler.load_state_dict(latent_checkpoint['scheduler_state_dict'])
                    logger.info("Loaded LatentTransformer scheduler state")
                except Exception as e:
                    logger.warning(f"Error loading scheduler state dict: {e}")
                    logger.warning("Creating new scheduler with current optimizer state")
                    
                    # Recreate scheduler based on current optimizer state
                    latent_scheduler = get_cosine_schedule_with_warmup(
                        latent_optimizer,
                        num_warmup_steps=config.warmup_steps,
                        num_training_steps=max_steps
                    )
                    
                    # Fast-forward scheduler to current step if resuming
                    if step_count > 0:
                        for _ in range(step_count):
                            latent_scheduler.step()
                        logger.info(f"Fast-forwarded new LatentTransformer scheduler to step {step_count}")
                    
                    models_dict["latent"]["scheduler"] = latent_scheduler
            
            # Ensure TensorBoard continuity with checkpoint
            ensure_checkpoint_tensorboard_consistency(models_dict["latent"], latent_checkpoint)
            
        except Exception as e:
            logger.error(f"Error loading LatentTransformer checkpoint: {e}")
            logger.error(traceback.format_exc())
    
    # After loading checkpoints, reset the seed to ensure reproducibility
    if simple_checkpoint is not None or latent_checkpoint is not None:
        logger.info("Resetting seed after loading checkpoint to ensure reproducibility")
        seed = 42
        # Extract seed from checkpoint config if available
        if simple_checkpoint and 'config' in simple_checkpoint and 'seed' in simple_checkpoint['config']:
            seed = simple_checkpoint['config']['seed']
        set_seed(seed)
    
    # Time tracking
    start_time = time.time()
    
    # Gradient scalers for mixed precision on CUDA
    if device.type == 'cuda':
        for model_info in models_dict.values():
            model_info["scaler"] = torch.amp.GradScaler()
    
    # Create a tqdm wrapper with logging integration
    def tqdm_with_logging(total=None, desc=None):
        """
        Create a tqdm progress bar that integrates with the logging system
        """
        return tqdm(total=total, desc=desc, ncols=100, leave=True)
    
    # Generate evaluation examples with lower range for easier evaluation
    try:
        # Define min_digits based on args parameter
        min_digits = 1
        if args is not None:
            min_digits = args.min_digits
            
        eval_max_val = 999 if min_digits >= 3 else (99 if min_digits == 2 else 9)
        logger.info(f"Precomputing val set with range 10-{min(5, eval_max_val)}")
        eval_examples = generate_evaluation_examples(dataset, device, min_digits)
        logger.info(f"Created {len(eval_examples)} evaluation examples")
        
    except Exception as e:
        logger.error(f"Error generating evaluation examples: {e}")
        logger.info("Adding fallback examples")
        # Add some basic fallback examples
        eval_examples = []
        for a, b in [(2, 3), (4, 5), (3, 7)]:
            input_str = f"{a}*{b}"
            input_tokens = dataset.encode(input_str)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            result = a * b
            result_str = str(result)
            eval_examples.append((input_tensor, result_str, a, b))
    
    # Main training loop
    val_freq = 50  # Validate every 50 steps
    
    try:
        # Reset random states for consistent data ordering
        if start_step > 0:
            logger.info("Setting consistent RNG state for resuming")
            # Get seed from checkpoint or use default
            resume_seed = 42
            if simple_checkpoint and 'seed' in simple_checkpoint:
                resume_seed = simple_checkpoint['seed']
            elif simple_checkpoint and 'config' in simple_checkpoint and 'seed' in simple_checkpoint['config']:
                resume_seed = simple_checkpoint['config']['seed']
            
            # Set all random states
            logger.info(f"Using seed {resume_seed} for consistent data ordering")
            random.seed(resume_seed)
            np.random.seed(resume_seed)
            torch.manual_seed(resume_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(resume_seed)
                torch.cuda.manual_seed_all(resume_seed)
            
            # Make sure the DataLoader worker seeds are set
            def worker_init_fn(worker_id):
                worker_seed = resume_seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)
            
            # Create a new DataLoader with fixed seed
            if hasattr(train_loader, 'dataset'):
                logger.info("Recreating training DataLoader with fixed seed")
                fixed_train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    collate_fn=train_loader.collate_fn,
                    num_workers=train_loader.num_workers,
                    worker_init_fn=worker_init_fn,
                    generator=torch.Generator().manual_seed(resume_seed),
                    pin_memory=True
                )
                train_loader = fixed_train_loader
        
        # Set models to training mode
        for model_info in models_dict.values():
            model_info["model"].train()
        
        # Calculate the maximum steps to be performed across both models
        total_steps_left = max(max_steps - models_dict['simple']['step'], 
                              max_steps - models_dict['latent']['step'])
        
        # If no steps left, we're already done
        if total_steps_left <= 0:
            logger.info("All models have already reached or exceeded max_steps, nothing to do")
            return {
                "simple": {
                    "loss": models_dict["simple"].get("final_loss", 0.0),
                    "sequence_accuracy": models_dict["simple"].get("final_sequence_accuracy", 0.0),
                    "digit_accuracy": models_dict["simple"].get("final_digit_accuracy", 0.0),
                    "params": models_dict["simple"]["params"],
                    "steps": models_dict["simple"]["step"]
                },
                "latent": {
                    "loss": models_dict["latent"].get("final_loss", 0.0),
                    "sequence_accuracy": models_dict["latent"].get("final_sequence_accuracy", 0.0),
                    "digit_accuracy": models_dict["latent"].get("final_digit_accuracy", 0.0),
                    "params": models_dict["latent"]["params"],
                    "steps": models_dict["latent"]["step"]
                },
                "training_time": 0.0
            }
        
        # Create progress bar showing correct total and progress
        pbar = tqdm(total=total_steps_left, desc=f"Training (steps)")
        
        # Training loop
        train_iter = itertools.cycle(train_loader)
        
        while True:
            # Get a batch
            inp, tgt, inp_lens, tgt_lens = next(train_iter)
            inp, tgt = inp.to(device), tgt.to(device)
            
            # Calculate teacher forcing probability based on schedule
            current_step = max(models_dict['simple']['step'], models_dict['latent']['step'])
            if tf_schedule == "none" or current_step < tf_start_step:
                tf_prob = 1.0  # Always use teacher forcing
            elif tf_schedule == "linear":
                progress = (current_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = 1.0 - progress  # Linear decay from 1.0 to 0.0
            elif tf_schedule == "exp":
                progress = (current_step - tf_start_step) / (max_steps - tf_start_step)
                tf_prob = math.exp(-10 * progress)  # Exponential decay
            else:
                tf_prob = 1.0
                
            # Track if any model is still training
            any_model_still_training = False
                
            # Train each model on the same batch
            for model_type, model_info in models_dict.items():
                # Check if model already reached max steps
                current_model_step = model_info["step"]
                if current_model_step >= max_steps:
                    continue
                    
                any_model_still_training = True
                
                model = model_info["model"]
                optimizer = model_info["optimizer"]
                criterion = model_info["criterion"]
                writer = model_info["writer"]
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Decide whether to use teacher forcing for this batch
                use_teacher_forcing = random.random() < tf_prob
                
                if use_teacher_forcing:
                    # Standard teacher forcing - use target as decoder input
                    decoder_input = tgt[:, :-1]
                    decoder_target = tgt[:, 1:]
                    
                    # Skip this batch if decoder input is empty
                    if decoder_input.size(1) == 0:
                        logger.warning(f"Empty decoder input in batch with teacher forcing, skipping for {model_info['name']}")
                        continue
                else:
                    # Generate decoder input on-the-fly
                    batch_size = inp.size(0)
                    decoder_target = tgt[:, 1:]
                    max_len = decoder_target.size(1)
                    
                    # Skip this batch if target is empty
                    if max_len == 0:
                        logger.warning(f"Empty target sequence, skipping batch for {model_info['name']}")
                        continue
                    
                    # Start with the start token
                    decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                    
                    # Only generate additional tokens if target length > 1
                    if max_len > 1:
                        try:
                            # Generate partial sequence for decoder input
                            with torch.no_grad():
                                for i in range(max_len - 1):
                                    # Forward pass to get next token prediction
                                    temp_output = model(inp, decoder_input)
                                    next_token_logits = temp_output[:, -1, :]
                                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                    # Append to decoder input
                                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                            
                            # Use all but the last token as decoder input
                            if decoder_input.size(1) > 1:
                                decoder_input = decoder_input[:, :-1]
                        except Exception as e:
                            logger.error(f"Error during sequence generation for {model_info['name']}: {e}")
                            logger.info(f"Falling back to teacher forcing for this batch")
                            decoder_input = tgt[:, :-1]
                            decoder_target = tgt[:, 1:]
                
                # Process batch with appropriate precision and error handling
                try:
                    # Mixed precision context for CUDA devices
                    amp_context = torch.amp.autocast(device_type=device.type) if device.type == 'cuda' else nullcontext()
                    scaler = model_info["scaler"] if device.type == 'cuda' else None
                    
                    # Forward pass and loss calculation
                    with amp_context:
                        output = model(inp, decoder_input)
                        
                        # Check for output/target shape mismatch
                        if output.size(0) != decoder_target.size(0) or output.size(1) != decoder_target.size(1):
                            # Resize to compatible shapes
                            min_batch = min(output.size(0), decoder_target.size(0))
                            min_seq_len = min(output.size(1), decoder_target.size(1))
                            output = output[:min_batch, :min_seq_len, :]
                            decoder_target = decoder_target[:min_batch, :min_seq_len]
                        
                        # Calculate loss
                        loss, batch_seq_accuracy = criterion(output, decoder_target)
                        
                        # Handle loss explosions
                        if torch.isnan(loss).any() or loss > LOSS_MAX_THRESHOLD[model_type]:
                            # Save emergency checkpoint
                            checkpoint_path = f"checkpoints/{model_info['name'].lower()}/{model_info['name'].lower()}_emergency.pt"
                            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                            
                            # Capture current state
                            current_lr = optimizer.param_groups[0]['lr']
                            emergency_state = {
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': model_info["scheduler"].state_dict(),
                                'step': model_info["step"],
                                'val_loss': model_info["val_loss"],
                                'config': model_info["config"].__dict__,
                                'last_lr': current_lr,
                                'explosion_detected': True,
                                'loss_value': loss.item() if not torch.isnan(loss).any() else float('nan')
                            }
                            torch.save(emergency_state, checkpoint_path)
                            
                            logger.warning(f"⚠️ Detected loss explosion for {model_info['name']} at step {model_info['step']}")
                            logger.warning(f"Loss value: {loss.item() if not torch.isnan(loss).any() else 'NaN'}, threshold: {LOSS_MAX_THRESHOLD[model_type]}")
                            logger.warning(f"Saved emergency checkpoint to {checkpoint_path}")
                            
                            # Check if this is a repeated explosion
                            steps_since_last_explosion = model_info["step"] - model_info["last_explosion_step"]
                            
                            # More aggressive LR reduction for frequent explosions
                            reduction_factor = LR_EMERGENCY_FACTOR[model_type]
                            if steps_since_last_explosion < 100:
                                # If exploding frequently, reduce more aggressively
                                reduction_factor *= 0.5
                                logger.warning(f"Frequent explosions detected! Using more aggressive LR reduction: {reduction_factor}")
                            
                            # Apply the emergency LR reduction
                            new_lr = current_lr * reduction_factor
                            logger.warning(f"🚨 Emergency LR reduction from {current_lr:.6f} to {new_lr:.6f}")
                            
                            # Update optimizer learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            
                            # Update scheduler to ensure consistency on future steps
                            # Create a new scheduler with the reduced learning rate
                            new_scheduler = get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=config.warmup_steps,
                                num_training_steps=max_steps
                            )
                            
                            # Fast-forward the scheduler to the current step
                            for _ in range(model_info["step"]):
                                new_scheduler.step()
                            
                            # Replace the old scheduler with the new one
                            model_info["scheduler"] = new_scheduler
                            
                            # Mark explosion step for cooldown and backtrack
                            model_info["last_explosion_step"] = model_info["step"]
                            
                            # Skip the rest of this batch processing
                            continue
                    
                    # Backward pass with appropriate scaling if needed
                    if device.type == 'cuda':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    
                    # Apply gradient clipping using the consistent approach
                    clip_norm = MAX_GRAD_NORM[model_type]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    
                    # Perform optimizer step
                    if device.type == 'cuda':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Log training metrics
                    with torch.no_grad():
                        # Calculate token and sequence accuracy
                        _, predicted = torch.max(output, dim=-1)
                        mask = (decoder_target != 0).float()
                        token_correct = (predicted == decoder_target).float() * mask
                        token_accuracy = token_correct.sum() / mask.sum().clamp(min=1e-8)
                        
                        # Log metrics using absolute model step
                        writer.add_scalar('train/loss', loss.item(), model_info["step"])
                        writer.add_scalar('train/token_accuracy', token_accuracy.item(), model_info["step"])
                        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], model_info["step"])
                        
                        # Check for learning rate consistency when resuming
                        if model_info["step"] == start_step + 1 and prev_lrs[model_type] is not None:
                            new_lr = optimizer.param_groups[0]['lr']
                            if abs(new_lr - prev_lrs[model_type]) > 1e-6:
                                logger.warning(f"Learning rate jump detected for {model_info['name']}: {prev_lrs[model_type]:.6f} -> {new_lr:.6f}")
                                logger.warning(f"This may cause training instability. Consider adjusting scheduler parameters.")
                    
                    # Update model step counter first, before any validation
                    model_info["step"] += 1
                    
                    # Track losses for stability monitoring
                    model_info["recent_losses"].append(loss.item())
                    if len(model_info["recent_losses"]) > model_info["stability_window"]:
                        model_info["recent_losses"].pop(0)
                    
                    # Step the learning rate scheduler
                    model_info["scheduler"].step()
                    
                    # Validation and checkpoint saving
                    save_checkpoint = False
                    perform_validation = False
                    
                    # Check if custom checkpoint frequency is set
                    if args is not None and hasattr(args, 'save_every') and args.save_every is not None:
                        # Use custom checkpoint frequency
                        if model_info["step"] % args.save_every == 0:
                            save_checkpoint = True
                            # Also perform validation when saving checkpoint
                            perform_validation = True
                    else:
                        # Use default validation frequency
                        if model_info["step"] > 0 and model_info["step"] % val_freq == 0:
                            perform_validation = True
                            save_checkpoint = True
                    
                    if perform_validation:
                        # Determine if we should rotate validation examples
                        rotate_examples = model_info["step"] > 3000 and model_info["step"] % (val_freq * 4) == 0
                        
                        val_loss, val_sequence_accuracy, val_digit_accuracy = evaluate(
                            model=model,
                            data_loader=val_loader,
                            criterion=criterion,
                            dataset=dataset,
                            device=device,
                            vocab_size=vocab_size,
                            desc=f"Val {model_info['name']}",
                            examples=eval_examples,
                            rotate_examples=rotate_examples
                        )
                        
                        # Store validation metrics
                        model_info["val_loss"] = val_loss
                        model_info["val_sequence_accuracy"] = val_sequence_accuracy
                        model_info["val_digit_accuracy"] = val_digit_accuracy
                        
                        # Log validation metrics using the current model step
                        writer.add_scalar('val/loss', val_loss, model_info["step"])
                        writer.add_scalar('val/sequence_accuracy', val_sequence_accuracy, model_info["step"])
                        writer.add_scalar('val/digit_accuracy', val_digit_accuracy, model_info["step"])
                        
                        # Print validation results
                        logger.info(f"\n{model_info['name']} - Step {model_info['step']}/{max_steps} - Val Loss: {val_loss:.6f} - Seq Acc: {val_sequence_accuracy:.2%}")
                        
                        # Check for best model and save if improved
                        if val_loss < model_info["best_val_loss"]:
                            model_info["best_val_loss"] = val_loss
                            # Save best model checkpoint
                            model_checkpoint_dir = f"checkpoints/{model_info['name'].lower()}"
                            os.makedirs(model_checkpoint_dir, exist_ok=True)
                            current_lr = optimizer.param_groups[0]['lr']
                            torch.save({
                                'step': model_info["step"],
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': model_info["scheduler"].state_dict(),
                                'val_loss': val_loss,
                                'val_sequence_accuracy': val_sequence_accuracy,
                                'config': model_info["config"].__dict__,
                                'seed': getattr(model_info["config"], "seed", 42),
                                'd_model': getattr(model_info["config"], "d_model", 64),
                                'last_lr': current_lr,
                            }, f"{model_checkpoint_dir}/{model_info['name'].lower()}_best.pt")
                            logger.info(f"Saved new best model for {model_info['name']} (val_loss: {val_loss:.6f})")
                        
                            # Return to training mode
                            model.train()
                        
                        # Save latest checkpoint if needed
                        if save_checkpoint:
                            model_checkpoint_dir = f"checkpoints/{model_info['name'].lower()}"
                            os.makedirs(model_checkpoint_dir, exist_ok=True)
                            current_lr = optimizer.param_groups[0]['lr']
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': model_info["scheduler"].state_dict(),
                                'step': model_info["step"],
                                'val_loss': model_info.get("val_loss", float('inf')),
                                'val_sequence_accuracy': model_info.get("val_sequence_accuracy", 0.0),
                                'config': model_info["config"].__dict__,
                                'seed': getattr(model_info["config"], "seed", 42),
                                'd_model': getattr(model_info["config"], "d_model", 64),
                                'last_lr': current_lr,
                                # Save teacher forcing state
                                'tf_schedule': tf_schedule,
                                'tf_start_step': tf_start_step,
                                'tf_current_probability': tf_prob,
                                # Save additional training state
                                'stability_window': model_info.get("stability_window", 100),
                                'recent_losses_mean': sum(model_info["recent_losses"])/len(model_info["recent_losses"]) if model_info["recent_losses"] else 0.0,
                                'max_steps': max_steps
                            }, f"{model_checkpoint_dir}/{model_info['name'].lower()}_latest.pt")
                            
                            if not perform_validation:
                                # Only log this if we haven't already logged validation results
                                logger.info(f"Saved checkpoint for {model_info['name']} at step {model_info['step']}")
                except Exception as e:
                    logger.error(f"Error in training step for {model_info['name']}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Update progress bar postfix with current loss values
            simple_loss = "N/A"
            latent_loss = "N/A"
            
            if models_dict['simple']['recent_losses']:
                simple_loss = f"{models_dict['simple']['recent_losses'][-1]:.4f}"
            if models_dict['latent']['recent_losses']:
                latent_loss = f"{models_dict['latent']['recent_losses'][-1]:.4f}"
                
            pbar.set_postfix({
                "simple_loss": simple_loss, 
                "latent_loss": latent_loss
            })
            
            # Update progress bar
            pbar.update(1)
            
            # Final check: If both models have reached the max steps, break
            simple_completed = models_dict['simple']['step'] >= max_steps
            latent_completed = models_dict['latent']['step'] >= max_steps
            
            # Log progress periodically
            current_step = max(models_dict['simple']['step'], models_dict['latent']['step'])
            if current_step % 100 == 0:
                logger.info(f"Training progress: SimpleTransformer {models_dict['simple']['step']}/{max_steps}, LatentTransformer {models_dict['latent']['step']}/{max_steps}")
            
            # Break if no more training needed or if no models were trained this iteration
            if (simple_completed and latent_completed) or not any_model_still_training:
                logger.info(f"Training complete. SimpleTransformer: {models_dict['simple']['step']}/{max_steps}, LatentTransformer: {models_dict['latent']['step']}/{max_steps}")
                break
                
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        
    # Final validation for both models
    for model_type, model_info in models_dict.items():
        final_val_loss, final_sequence_accuracy, final_digit_accuracy = evaluate(
            model=model_info["model"],
            data_loader=val_loader,
            criterion=model_info["criterion"],
            dataset=dataset,
            device=device,
            vocab_size=vocab_size,
            desc=f"Final {model_info['name']} validation",
            examples=eval_examples
        )
        
        model_info["final_loss"] = final_val_loss
        model_info["final_sequence_accuracy"] = final_sequence_accuracy
        model_info["final_digit_accuracy"] = final_digit_accuracy
    
    training_time = time.time() - start_time
    
    # Print summaries
    for model_type, model_info in models_dict.items():
        logger.info(f"\n{model_info['name']} Training Summary:")
        logger.info(f"Steps completed: {model_info['step']}/{max_steps}")
        logger.info(f"Best validation loss: {model_info['best_val_loss']:.6f}")
        logger.info(f"Final validation loss: {model_info['final_loss']:.6f}")
        logger.info(f"Final sequence accuracy: {model_info['final_sequence_accuracy']:.2%}")
        logger.info(f"Final digit accuracy: {model_info['final_digit_accuracy']:.2%}")
    
    # Close writers
    for model_info in models_dict.values():
        model_info["writer"].close()
    
    return {
        "simple": {
            "loss": models_dict["simple"]["final_loss"],
            "sequence_accuracy": models_dict["simple"]["final_sequence_accuracy"],
            "digit_accuracy": models_dict["simple"]["final_digit_accuracy"],
            "params": models_dict["simple"]["params"],
            "steps": models_dict["simple"]["step"]
        },
        "latent": {
            "loss": models_dict["latent"]["final_loss"],
            "sequence_accuracy": models_dict["latent"]["final_sequence_accuracy"],
            "digit_accuracy": models_dict["latent"]["final_digit_accuracy"],
            "params": models_dict["latent"]["params"],
            "steps": models_dict["latent"]["step"]
        },
        "training_time": training_time
    }

def ensure_checkpoint_tensorboard_consistency(model_info, checkpoint=None):
    """
    Ensure proper consistency between model step, checkpoint data, and TensorBoard metrics.
    Creates a smooth transition in graphs when resuming training.
    
    Args:
        model_info (dict): Dictionary containing model, step, and writer information
        checkpoint (dict, optional): Loaded checkpoint data
    """
    if checkpoint is None or "step" not in checkpoint:
        return  # Nothing to do if no checkpoint or step info
    
    # Get the resuming step from checkpoint
    resume_step = checkpoint.get("step", 0)
    
    # Update the model step to match the checkpoint
    model_info["step"] = resume_step
    
    logger.info(f"Ensuring TensorBoard consistency for {model_info['name']} at step {resume_step}")
    
    # Get the writer
    writer = model_info["writer"]
    
    # List of common metrics to preserve continuity for
    common_metrics = [
        'train/loss', 
        'train/token_accuracy', 
        'train/learning_rate', 
        'val/loss', 
        'val/sequence_accuracy', 
        'val/digit_accuracy'
    ]
    
    # Extract values from checkpoint for bridging
    metric_values = {
        'val/loss': checkpoint.get('val_loss', None),
        'val/sequence_accuracy': checkpoint.get('val_sequence_accuracy', None),
        'train/learning_rate': checkpoint.get('last_lr', None)
    }
    
    # Add metrics to TensorBoard to establish visual continuity between runs
    # Place a bridging point at the resuming step to create a continuous line
    for metric in common_metrics:
        # Use actual values from checkpoint when available
        if metric in metric_values and metric_values[metric] is not None:
            value = metric_values[metric]
            writer.add_scalar(metric, value, resume_step - 1)
        else:
            # For metrics without checkpoint values, use a small positive value
            # that won't disrupt the visual continuity (0.0 works well)
            writer.add_scalar(metric, 0.0, resume_step - 1)
    
    # Restore other training state from checkpoint if available
    if 'recent_losses_mean' in checkpoint and checkpoint['recent_losses_mean'] > 0:
        # Add a representative loss value to the recent losses to help with stability calculations
        mean_loss = checkpoint['recent_losses_mean']
        for _ in range(min(5, model_info["stability_window"])):
            model_info["recent_losses"].append(mean_loss)
        logger.info(f"Restored mean loss value ({mean_loss:.4f}) from checkpoint")
    
    # Log teacher forcing state if available
    if 'tf_schedule' in checkpoint and 'tf_start_step' in checkpoint:
        tf_schedule = checkpoint.get('tf_schedule')
        tf_start_step = checkpoint.get('tf_start_step')
        tf_prob = checkpoint.get('tf_current_probability', 1.0)
        logger.info(f"Restored teacher forcing state: schedule={tf_schedule}, start_step={tf_start_step}, current_prob={tf_prob:.4f}")
    
    logger.info(f"TensorBoard continuity established for {model_info['name']}")