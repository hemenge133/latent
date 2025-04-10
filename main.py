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
import optuna  # Import Optuna for hyperparameter optimization
from optuna.exceptions import TrialPruned  # For pruning incompatible trials

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


def main(args):
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Enable MPS fallback
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Enable TF32 on Ampere GPUs for faster training with minimal precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Default to mixed precision for training speed
    torch.set_float32_matmul_precision("high")

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

    # --- Load and prepare datasets ---
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

    # Check if using best parameters from a file
    if args.use_best_params:
        try:
            with open(args.use_best_params, 'r') as f:
                best_params = json.load(f)
                logger.info(f"Loaded best parameters from {args.use_best_params}")
                # Override args with loaded parameters
                for key, value in best_params.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                        logger.info(f"Overrode {key}={value}")
        except Exception as e:
            logger.error(f"Failed to load best parameters from {args.use_best_params}: {e}")
    
    # Check if we should load best parameters directly from a study
    if args.use_best_from_study:
        try:
            logger.info(f"Loading best parameters from study: {args.study_name}")
            db_path = f"{args.study_name}.db"
            storage_path = f"sqlite:///{db_path}"
            
            # Check if the database file exists
            if not os.path.exists(db_path):
                logger.error(f"Study database file '{db_path}' does not exist.")
                logger.info(f"Run optimization first with: python main.py --optimize --study-name {args.study_name}")
                if args.optimize:
                    logger.info("Will create a new study since --optimize is specified.")
                else:
                    return
            
            try:
                study = optuna.load_study(
                    study_name=args.study_name,
                    storage=storage_path
                )
                
                # Check if study has any completed trials
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if not completed_trials:
                    logger.error(f"Study '{args.study_name}' exists but has no completed trials.")
                    if args.optimize:
                        logger.info("Continuing with optimization since --optimize is specified.")
                    else:
                        return
                else:
                    logger.info(f"Found {len(completed_trials)} completed trials in study.")
                    logger.info(f"Best trial: #{study.best_trial.number} with value: {study.best_trial.value}")
                    best_params = study.best_trial.params
                    
                    # Override args with loaded parameters
                    for key, value in best_params.items():
                        if hasattr(args, key):
                            setattr(args, key, value)
                            logger.info(f"Overrode {key}={value}")
                    
                    # Also save to a file for reference
                    best_params_file = f"best_params_{args.study_name}_used.json"
                    with open(best_params_file, 'w') as f:
                        json.dump(best_params, f, indent=4)
                    logger.info(f"Saved used parameters to {best_params_file}")
            except ValueError as ve:
                if "Record does not exist" in str(ve):
                    logger.error(f"Study '{args.study_name}' exists but has no best trial data.")
                    if args.optimize:
                        logger.info("Continuing with optimization since --optimize is specified.")
                    else:
                        return
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to load best parameters from study {args.study_name}: {e}")
            logger.error(traceback.format_exc())
            if not args.optimize:
                logger.info("If you meant to start a new optimization run, add the --optimize flag.")

    # Define objective function for Optuna
    def objective(trial):
        try:
            # Generate trial-specific seed
            trial_seed = seed + trial.number
            set_seed(trial_seed)
            MultiplicationDataset.set_fixed_seed(trial_seed)
            
            # Suggest hyperparameters
            # Model architecture parameters
            current_d_model = trial.suggest_categorical("d_model", [64, 128, 256, 384, 512])
            current_nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
            
            # Check compatibility between d_model and nhead
            if current_d_model % current_nhead != 0:
                logger.warning(f"Incompatible d_model={current_d_model} and nhead={current_nhead}. Pruning trial.")
                raise TrialPruned()
                
            current_num_layers = trial.suggest_int("num_layers", 1, 6)
            current_dropout = trial.suggest_float("dropout", 0.0, 0.5)
            current_num_latent = trial.suggest_int("num_latent", 1, 32)
            current_bottleneck_factor = trial.suggest_float("bottleneck_factor", 0.1, 1.0)
            
            # Training parameters
            current_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            current_warmup_steps = trial.suggest_int("warmup_steps", 50, 500)
            
            # Optimizer parameters
            simple_lr = trial.suggest_float("simple_lr", 1e-5, 1e-3, log=True)
            simple_wd = trial.suggest_float("simple_wd", 1e-5, 0.1, log=True)
            latent_lr = trial.suggest_float("latent_lr", 1e-5, 1e-3, log=True)
            latent_wd = trial.suggest_float("latent_wd", 1e-5, 0.1, log=True)
            
            # Create models for this trial
            logger.info(f"Trial {trial.number}: Creating models with d_model={current_d_model}, nhead={current_nhead}, "
                        f"layers={current_num_layers}, dropout={current_dropout}, latent={current_num_latent}, "
                        f"bottleneck={current_bottleneck_factor}")
            
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
                bottleneck_factor=current_bottleneck_factor
            ).to(device)

            simple_params_count = sum(p.numel() for p in simple_transformer.parameters())
            latent_params_count = sum(p.numel() for p in latent_transformer.parameters())
            logger.info(f"Simple Params: {simple_params_count:,}, Latent Params: {latent_params_count:,}")

            # Define unique log directory for this trial, including key hyperparameters
            param_str = (
                f"d{current_d_model}_nl{current_num_layers}_nlt{current_num_latent}_bs{current_batch_size}_"
                f"slr{simple_lr:.0e}_swd{simple_wd:.0e}_llr{latent_lr:.0e}_lwd{latent_wd:.0e}"
            )
            log_dir = f"runs/optuna_study_{args.study_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_trial_{trial.number}_{param_str}"
            logger.info(f"Logging to: {log_dir}")

            # Train with current hyperparameters
            results = train_models_parallel(
                models={"simple": simple_transformer, "latent": latent_transformer},
                dataset=train_dataset,
                dataset_val=val_dataset,
                vocab_size=vocab_size,
                device=device,
                max_steps=args.max_steps,
                batch_size=current_batch_size,
                config=config,
                args=args,
                models_params={"simple": simple_params_count, "latent": latent_params_count},
                start_step=0,
                simple_checkpoint=None,
                latent_checkpoint=None,
                log_dir=log_dir,
                # Pass the hyperparameters from the trial
                simple_lr=simple_lr,
                simple_wd=simple_wd,
                latent_lr=latent_lr,
                latent_wd=latent_wd,
                warmup_steps=current_warmup_steps,
                bottleneck_factor=current_bottleneck_factor,
            )
            
            # Extract metrics for Optuna to optimize
            # Note: "sequence_accuracy" here is the inference-based sequence accuracy,
            # which measures the model's ability to solve problems from scratch without teacher forcing
            latent_final_val_acc = results.get("latent", {}).get("sequence_accuracy", 0.0)
            latent_final_val_loss = results.get("latent", {}).get("loss", float('inf'))
            
            # Get the simple model results for comparison
            simple_final_val_acc = results.get("simple", {}).get("sequence_accuracy", 0.0)
            
            # Create a more detailed trial summary
            logger.info(f"\n{'='*80}")
            logger.info(f"TRIAL {trial.number} COMPLETED")
            logger.info(f"{'='*80}")
            logger.info(f"Latent model inference accuracy: {latent_final_val_acc:.4f}")
            logger.info(f"Latent model validation loss: {latent_final_val_loss:.6f}")
            logger.info(f"Simple model inference accuracy: {simple_final_val_acc:.4f}")
            logger.info(f"Parameters:")
            for param_name, param_value in trial.params.items():
                logger.info(f"  {param_name}: {param_value}")
            logger.info(f"{'='*80}")
            
            # Check if this is the best accuracy so far for this trial
            if hasattr(trial, "study") and trial.study.best_value and latent_final_val_acc >= trial.study.best_value:
                logger.info(f"NEW BEST TRIAL! Accuracy: {latent_final_val_acc:.4f}")
            
            logger.info(f"--- Objective function for Trial {trial.number} returning value: {latent_final_val_acc:.4f} ---")
            # Return metric to maximize (or negative loss to minimize)
            return latent_final_val_acc
            
        except Exception as e:
            logger.error(f"Error during trial {trial.number}: {e}")
            logger.error(traceback.format_exc())
            # Return a default value on error
            return 0.0

    # --- Checkpoint Loading for Single Run ---
    simple_checkpoint = None
    latent_checkpoint = None
    simple_checkpoint_path = "checkpoints/simpletransformer/simpletransformer_latest.pt"
    latent_checkpoint_path = "checkpoints/latenttransformer/latenttransformer_latest.pt"
    start_step = 0
    
    # Handle standalone study check or diagnosis first
    if args.check_study or args.diagnose_study:
        db_path = f"{args.study_name}.db"
        storage_path = f"sqlite:///{db_path}"
        if not os.path.exists(db_path):
            logger.error(f"Database file {db_path} does not exist. Cannot check/diagnose.")
            return
        
        study = optuna.load_study(study_name=args.study_name, storage=storage_path)
        
        if args.check_study:
            logger.info(f"Checking progress of study: {args.study_name}")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            logger.info(f"Total trials: {len(study.trials)}")
            logger.info(f"Number of completed trials: {len(completed_trials)}")
            
            # Check if a best trial exists before accessing it
            best_trial = None
            try:
                best_trial = study.best_trial
            except ValueError: # Handles case where no completed trials exist
                logger.info("No best trial found yet (likely no completed trials with a value).")
            
            if best_trial:
                logger.info(f"Best trial number: {best_trial.number}")
                logger.info(f"Best value: {best_trial.value}")
                logger.info("Best hyperparameters:")
                for key, value in best_trial.params.items():
                    logger.info(f"    {key}: {value}")
                # Save best parameters
                best_params_file = f"best_params_{args.study_name}.json"
                with open(best_params_file, 'w') as f:
                    json.dump(best_trial.params, f, indent=4)
                logger.info(f"Best parameters saved to {best_params_file}")
            
            # Export trial history to CSV regardless of best trial
            if study.trials:
                try:
                    import pandas as pd
                    trials_df = study.trials_dataframe()
                    csv_file = f"{args.study_name}_trials.csv"
                    trials_df.to_csv(csv_file)
                    logger.info(f"Trial history exported to {csv_file}")
                except Exception as e:
                    logger.error(f"Failed to export trial history to CSV: {e}")
            else:
                logger.info("No trials found in the study to export.")
                
            return # Exit after checking study

        if args.diagnose_study:
            logger.info(f"Diagnosing study database: {args.study_name}.db")
            db_path = f"{args.study_name}.db"
            
            if not os.path.exists(db_path):
                logger.error(f"Database file {db_path} does not exist.")
                return
                
            # Log file info
            file_size = os.path.getsize(db_path)
            logger.info(f"Database file size: {file_size} bytes")
            
            try:
                # Try to directly access the SQLite database to list studies
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Try to get study information
                cursor.execute("SELECT study_id, study_name FROM studies")
                studies = cursor.fetchall()
                
                if not studies:
                    logger.info("No studies found in the database.")
                else:
                    logger.info(f"Found {len(studies)} studies in the database:")
                    for study_id, study_name in studies:
                        logger.info(f"  Study ID: {study_id}, Name: {study_name}")
                        
                        # Get trial information (ID and state) from the trials table
                        cursor.execute(f"SELECT trial_id, state FROM trials WHERE study_id = {study_id}")
                        trials_info = cursor.fetchall()
                        
                        if not trials_info:
                            logger.info(f"    No trials found for study {study_name}")
                        else:
                            states = {}
                            trial_values = {}
                            for trial_id, state in trials_info:
                                states[state] = states.get(state, 0) + 1
                                # Query trial_values table separately for the objective value
                                cursor.execute(f"SELECT value FROM trial_values WHERE trial_id = {trial_id}")
                                value_result = cursor.fetchone()
                                if value_result:
                                    trial_values[trial_id] = value_result[0]
                                
                            logger.info(f"    Found {len(trials_info)} trials:")
                            logger.info(f"    Trial states (0=RUNNING, 1=COMPLETE, 2=PRUNED, 3=FAIL, 4=WAITING): {states}")
                            
                            # Look for completed trials with values
                            completed_ids_with_value = [tid for tid, state in trials_info if state == 1 and tid in trial_values]
                            logger.info(f"    Completed trials with values: {len(completed_ids_with_value)}")
                            
                            if completed_ids_with_value:
                                # Find best trial ID based on stored values
                                best_trial_id = max(completed_ids_with_value, key=lambda tid: trial_values.get(tid, float('-inf')))
                                best_value = trial_values.get(best_trial_id)
                                logger.info(f"    Best trial: ID={best_trial_id}, Value={best_value}")
                                
                                # Get parameters for best trial
                                cursor.execute(f"SELECT param_name, param_value FROM trial_params WHERE trial_id = {best_trial_id}")
                                params = cursor.fetchall()
                                if params:
                                    logger.info(f"    Best trial parameters:")
                                    for param_name, param_value in params:
                                        # Attempt to convert JSON string params back
                                        try:
                                            param_value_parsed = json.loads(param_value)
                                        except (json.JSONDecodeError, TypeError):
                                            param_value_parsed = param_value # Keep as is if not JSON
                                        logger.info(f"      {param_name}: {param_value_parsed}")
                
                conn.close()
                
                # If studies were found, suggest next steps
                if studies:
                    study_names = [name for _, name in studies]
                    if args.study_name not in study_names:
                        logger.warning(f"Study name '{args.study_name}' not found in database! Available studies: {study_names}")
                        logger.info(f"Use one of these study names instead: {', '.join(study_names)}")
                        
            except Exception as e:
                logger.error(f"Error diagnosing database: {e}")
                logger.error(traceback.format_exc())
                
            return # Exit after diagnosing study

    # --- Optuna Optimization or Standard Training Run ---
    if args.optimize:
        logger.info(f"Starting Optuna optimization with study name: {args.study_name}")
        
        # Create or load study
        storage_path = f"sqlite:///{args.study_name}.db"
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="maximize"  # Maximize accuracy
        )
        
        # Warm start with provided parameters if requested
        if args.warm_start_params:
            try:
                with open(args.warm_start_params, 'r') as f:
                    warm_start_params = json.load(f)
                    logger.info(f"Enqueuing warm start parameters: {warm_start_params}")
                    study.enqueue_trial(warm_start_params)
            except Exception as e:
                logger.error(f"Failed to load warm start parameters: {e}")
        
        # Run optimization
        if args.indefinite:
            logger.info("Starting indefinite optimization - will run until manually stopped")
            logger.info(f"Study results are being saved to: {args.study_name}.db")
            logger.info("You can check the current best parameters at any time in another terminal using:")
            logger.info(f"  python -c \"import optuna; study = optuna.load_study(study_name='{args.study_name}', storage='sqlite:///{args.study_name}.db'); print('Best value:', study.best_value); print('Best params:', study.best_trial.params)\"")
            study.optimize(objective, n_trials=None, timeout=args.timeout)
        else:
            logger.info(f"Starting optimization with {args.num_trials} trials")
            study.optimize(objective, n_trials=args.num_trials, timeout=args.timeout)
        
        # Output best trial information
        logger.info("--- Optimization Complete ---")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_trial.value}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")
        
        # Save best parameters to file
        best_params_file = f"best_params_{args.study_name}.json"
        with open(best_params_file, 'w') as f:
            json.dump(study.best_trial.params, f, indent=4)
        logger.info(f"Best parameters saved to {best_params_file}")
        
        # Export all trial results to CSV for analysis without TensorBoard
        try:
            import pandas as pd
            trials_df = study.trials_dataframe()
            csv_file = f"{args.study_name}_trials.csv"
            trials_df.to_csv(csv_file)
            logger.info(f"Trial history exported to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to export trial history to CSV: {e}")
            
    else:
        # --- Standard Single Training Run ---
        logger.info("Starting standard single training run...")

        # Handle checkpoint loading for single run
        if args.resume:
             logger.info("Attempting to resume training...")
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
                         args.d_model = simple_dims['d_model']
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
                         args.num_layers = simple_dims['num_layers']
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
                         args.num_latent = latent_dims['num_latent']
                     # else: keep final_num_latent as args.num_latent
                 else:
                     logger.warning("Could not extract num_latent from latent checkpoint, using CLI value.")
                     # Keep final_num_latent as args.num_latent
                 
                 # Check for d_model consistency (Latent vs Final determined above)
                 if latent_dims['d_model'] is not None and latent_dims['d_model'] != args.d_model:
                     logger.warning(f"d_model mismatch! Latent checkpoint suggests {latent_dims['d_model']}, but using {args.d_model} (determined by simple checkpoint/CLI).")
                 # Check for num_layers consistency
                 if latent_dims['num_layers'] is not None and latent_dims['num_layers'] != args.num_layers:
                     logger.warning(f"num_layers mismatch! Latent checkpoint suggests {latent_dims['num_layers']}, but using {args.num_layers} (determined by simple checkpoint/CLI).")
             # else: Keep final_num_latent as args.num_latent if no latent checkpoint
             
             logger.info(f"Resuming with parameters: d_model={args.d_model}, layers={args.num_layers}, latent={args.num_latent}, start_step={start_step}")

        else: # if not resume_run
             logger.info("Starting training from scratch (no resume).")
             start_step = 0
             # Use CLI args directly

        # Log the final parameters that will be used for model creation
        logger.info(f"Final model creation parameters (standard run): d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}, dropout={args.dropout}")

        # Create the models with the determined parameters
        logger.info(f"Creating SimpleTransformer with d_model={args.d_model}, nhead={args.nhead}, num_layers={args.num_layers}, dropout={args.dropout}")
        simple_transformer = StableSimpleTransformer(
             vocab_size=vocab_size,
             d_model=args.d_model,
             nhead=args.nhead, # Use nhead from args
             num_layers=args.num_layers,
             dropout=args.dropout, # Use dropout from args
        ).to(device)

        logger.info(f"Creating LatentTransformer with d_model={args.d_model}, nhead={args.nhead}, num_layers={args.num_layers}, "
                   f"dropout={args.dropout}, num_latent={args.num_latent}, bottleneck_factor={args.bottleneck_factor}")
        latent_transformer = StableLatentTransformer(
             vocab_size=vocab_size,
             d_model=args.d_model,
             nhead=args.nhead, # Use nhead from args
             num_layers=args.num_layers,
             num_latent=args.num_latent,
             dropout=args.dropout, # Use dropout from args
             bottleneck_factor=args.bottleneck_factor  # Use from args
        ).to(device)

        simple_params_count = sum(p.numel() for p in simple_transformer.parameters())
        latent_params_count = sum(p.numel() for p in latent_transformer.parameters())
        logger.info(f"SimpleTransformer has {simple_params_count:,} parameters")
        logger.info(f"LatentTransformer has {latent_params_count:,} parameters")

        # Define log directory for TensorBoard (standard run)
        # Include key hyperparameters in the log directory name
        _simple_lr = args.simple_lr if hasattr(args, 'simple_lr') and args.simple_lr is not None else config.base_lr
        _simple_wd = args.simple_wd if hasattr(args, 'simple_wd') and args.simple_wd is not None else config.weight_decay
        _latent_lr = args.latent_lr if hasattr(args, 'latent_lr') and args.latent_lr is not None else config.base_lr
        _latent_wd = args.latent_wd if hasattr(args, 'latent_wd') and args.latent_wd is not None else config.weight_decay
        
        param_str = (
            f"d{args.d_model}_nl{args.num_layers}_nlt{args.num_latent}_bs{args.batch_size}_"
            f"slr{_simple_lr:.0e}_swd{_simple_wd:.0e}_llr{_latent_lr:.0e}_lwd{_latent_wd:.0e}"
        )
        log_dir_base = "runs/standard_run"
        log_dir = f"{log_dir_base}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{param_str}"
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
            simple_checkpoint=simple_checkpoint if args.resume else None,
            latent_checkpoint=latent_checkpoint if args.resume else None,
            log_dir=log_dir,
            # Pass hyperparameters (use args for simple_lr, etc. if provided)
            simple_lr=args.simple_lr if hasattr(args, 'simple_lr') and args.simple_lr is not None else config.base_lr,
            simple_wd=args.simple_wd if hasattr(args, 'simple_wd') and args.simple_wd is not None else config.weight_decay,
            latent_lr=args.latent_lr if hasattr(args, 'latent_lr') and args.latent_lr is not None else config.base_lr,
            latent_wd=args.latent_wd if hasattr(args, 'latent_wd') and args.latent_wd is not None else config.weight_decay,
            warmup_steps=args.warmup_steps,
            bottleneck_factor=args.bottleneck_factor,
        )

        # Flush logs after training
        logger.info("Flushing logs after training loop completion...")
        sys.stdout.flush()
        sys.stderr.flush()

    # Finalize logging
    logger.info("Finalizing logging before exit...")
    logger.complete()


if __name__ == "__main__":
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

    # Optimization related arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use Optuna for hyperparameter optimization",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="latent_transformer_study",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for Optuna study",
    )
    parser.add_argument(
        "--warm-start-params",
        type=str,
        help="JSON file with parameters to warm start Optuna",
    )
    parser.add_argument(
        "--use-best-params",
        type=str,
        help="JSON file with best parameters to use for single run",
    )
    parser.add_argument(
        "--use-best-from-study",
        action="store_true",
        help="Automatically use the best parameters from the study specified by --study-name",
    )
    parser.add_argument(
        "--check-study",
        action="store_true",
        help="Check progress of an existing study and export results without running new trials",
    )
    parser.add_argument(
        "--diagnose-study",
        action="store_true",
        help="Show detailed diagnostic information about studies and trials in the database",
    )

    # Individual hyperparameters for non-optuna runs
    parser.add_argument(
        "--simple-lr", type=float, help="Learning rate for SimpleTransformer"
    )
    parser.add_argument(
        "--simple-wd", type=float, help="Weight decay for SimpleTransformer"
    )
    parser.add_argument(
        "--latent-lr", type=float, help="Learning rate for LatentTransformer"
    )
    parser.add_argument(
        "--latent-wd", type=float, help="Weight decay for LatentTransformer"
    )

    # New argument for indefinite optimization
    parser.add_argument(
        "--indefinite",
        action="store_true",
        help="Run optimization indefinitely until manually stopped",
    )

    # Clean option to remove all training and optimization artifacts
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all training runs, checkpoints, logs, and optimization artifacts",
    )

    args = parser.parse_args()
    
    # Handle clean option
    if args.clean:
        import subprocess
        import os
        
        print("Cleaning all training runs, checkpoints, logs, and optimization artifacts...")
        
        try:
            # Run the clean_runs.sh script
            clean_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_runs.sh")
            subprocess.run(["bash", clean_script_path], check=True)
            print("Clean completed successfully.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"Error cleaning artifacts: {e}")
            sys.exit(1)
    
    main(args)
