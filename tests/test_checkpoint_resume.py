#!/usr/bin/env python
"""
Test script to verify that optimizer and scheduler states are properly persisted during checkpoint resumption.
"""

import pytest
import torch
import os
import shutil
from loguru import logger
import sys
import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("test_checkpoint_resume.log", rotation="100 MB")  # Add file handler with rotation

from src.Models import StableSimpleTransformer
from src.Dataset import MultiplicationDataset
from src.Losses import SequenceAccuracyLoss
from src.Config import TrainingConfig
from src.TrainingLoop import train_models_parallel

def setup_test_environment():
    """Set up a clean test environment for checkpoint testing"""
    # Create test log directory
    log_dir = os.path.join(os.getcwd(), "runs/test_checkpoint")
    os.makedirs(log_dir, exist_ok=True)
    
    # Clean previous test runs if any
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    # Recreate directories
    os.makedirs(os.path.join(log_dir, "simple"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "latent"), exist_ok=True)
    
    return log_dir

def cleanup_test_environment():
    """Clean up test environment"""
    # Don't delete checkpoints as they may be needed for other tests
    pass

def extract_optimizer_info(optimizer_state_dict):
    """Extract key information from optimizer state dict for comparison"""
    info = {}
    
    # Extract learning rate from param_groups
    if 'param_groups' in optimizer_state_dict:
        info['lr'] = optimizer_state_dict['param_groups'][0]['lr']
        info['weight_decay'] = optimizer_state_dict['param_groups'][0]['weight_decay']
    
    # Extract state size information
    if 'state' in optimizer_state_dict:
        info['state_keys'] = len(optimizer_state_dict['state'])
        # Get sample of momentum/velocity values
        if optimizer_state_dict['state']:
            sample_key = list(optimizer_state_dict['state'].keys())[0]
            sample_state = optimizer_state_dict['state'][sample_key]
            
            # For Adam optimizer
            if 'exp_avg' in sample_state:
                info['exp_avg_mean'] = float(sample_state['exp_avg'].mean().item())
                info['exp_avg_sq_mean'] = float(sample_state['exp_avg_sq'].mean().item())
                
            # Step count
            if 'step' in sample_state:
                info['step'] = int(sample_state['step'].item())
    
    return info

def extract_scheduler_info(scheduler_state_dict):
    """Extract key information from scheduler state dict for comparison"""
    info = {}
    
    # Extract common fields
    if 'last_epoch' in scheduler_state_dict:
        info['last_epoch'] = scheduler_state_dict['last_epoch']
    
    # For cosine annealing
    if '_step_count' in scheduler_state_dict:
        info['step_count'] = scheduler_state_dict['_step_count']
    
    # Add other scheduler-specific fields as needed
    for key in ['base_lrs', 'finished', 'verbose']:
        if key in scheduler_state_dict:
            info[key] = scheduler_state_dict[key]
    
    return info

def test_checkpoint_optimizer_persistence():
    """Test that optimizer and scheduler states are properly persisted during checkpoint resumption"""
    try:
        # Setup test environment
        log_dir = setup_test_environment()
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config with test-specific settings
        config = TrainingConfig()
        config.base_lr = 3e-4
        config.warmup_steps = 10  # Small warmup for testing
        config.weight_decay = 0.01
        config.batch_size = 4  # Small batch size for testing
        config.validate_every_n_steps = 5  # Validate more frequently for testing
        config.test_every_n_steps = 10  # Test more frequently for testing
        config.save_every = 5  # Save checkpoints more frequently for testing
        config.min_value = 10  # Use smaller numbers for testing
        config.max_value = 99
        
        # Create a small dataset for testing
        dataset = MultiplicationDataset(
            num_samples=100,
            split='train',
            min_value=config.min_value,
            max_value=config.max_value,
            seed=42
        )
        
        # Create validation dataset
        val_dataset = MultiplicationDataset(
            num_samples=20,
            split='val',
            min_value=config.min_value,
            max_value=config.max_value,
            seed=43
        )
        
        # Create models
        vocab_size = 12  # Small vocab size for testing
        d_model = 32     # Small model size for testing
        
        simple_model = StableSimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # Create a dummy latent model (we'll use SimpleTransformer for simplicity)
        latent_model = StableSimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # Create criterion
        criterion = SequenceAccuracyLoss()
        
        # Create models dict
        models = {
            "simple": simple_model,
            "latent": latent_model
        }
        
        # Create models params dict
        models_params = {
            "simple": sum(p.numel() for p in simple_model.parameters()),
            "latent": sum(p.numel() for p in latent_model.parameters())
        }
        
        # Run training for a small number of steps
        logger.info("Starting first training run (15 steps)")
        results = train_models_parallel(
            models=models,
            dataset=dataset,
            dataset_val=val_dataset,
            vocab_size=vocab_size,
            criterion=criterion,
            device=device,
            max_steps=15,  # Run for 15 steps
            batch_size=4,
            learning_rate=3e-4,
            config=config,
            models_params=models_params,
            start_step=0,
            log_dir=log_dir
        )
        
        # Check that checkpoint files were created
        simple_checkpoint_path = os.path.join(os.getcwd(), "checkpoints/simpletransformer/simpletransformer_latest.pt")
        latent_checkpoint_path = os.path.join(os.getcwd(), "checkpoints/latenttransformer/latenttransformer_latest.pt")
        
        logger.info(f"Looking for checkpoint at: {simple_checkpoint_path}")
        assert os.path.exists(simple_checkpoint_path), "Simple checkpoint not created"
        assert os.path.exists(latent_checkpoint_path), "Latent checkpoint not created"
        
        # Load checkpoints
        simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
        latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)
        
        # Extract optimizer and scheduler info from first run
        simple_optimizer_info_first = extract_optimizer_info(simple_checkpoint['optimizer_state_dict'])
        simple_scheduler_info_first = extract_scheduler_info(simple_checkpoint['scheduler_state_dict'])
        
        latent_optimizer_info_first = extract_optimizer_info(latent_checkpoint['optimizer_state_dict'])
        latent_scheduler_info_first = extract_scheduler_info(latent_checkpoint['scheduler_state_dict'])
        
        # Record the last step from checkpoints
        simple_last_step = simple_checkpoint.get("step", 0)
        latent_last_step = latent_checkpoint.get("step", 0)
        
        logger.info(f"Last step from checkpoint - Simple: {simple_last_step}")
        logger.info(f"Last step from checkpoint - Latent: {latent_last_step}")
        logger.info(f"Simple optimizer info: {simple_optimizer_info_first}")
        logger.info(f"Simple scheduler info: {simple_scheduler_info_first}")
        
        # Record the last learning rate
        last_lr_simple = simple_checkpoint.get("last_lr", 0)
        last_lr_latent = latent_checkpoint.get("last_lr", 0)
        
        logger.info(f"Last LR from checkpoint - Simple: {last_lr_simple}")
        logger.info(f"Last LR from checkpoint - Latent: {last_lr_latent}")
        
        # Run a second training session, resuming from checkpoints
        logger.info("Starting second training run (resumed, +15 steps)")
        # Reset models to ensure we're genuinely loading from checkpoints
        simple_model = StableSimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        latent_model = StableSimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # Create models dict
        models = {
            "simple": simple_model,
            "latent": latent_model
        }
        
        # Run training for another 15 steps
        results_resumed = train_models_parallel(
            models=models,
            dataset=dataset,
            dataset_val=val_dataset,
            vocab_size=vocab_size,
            criterion=criterion,
            device=device,
            max_steps=30,  # Total 30 steps (15 + 15)
            batch_size=4,
            learning_rate=3e-4,
            config=config,
            models_params=models_params,
            start_step=simple_last_step,  # Resume from last step
            simple_checkpoint=simple_checkpoint,
            latent_checkpoint=latent_checkpoint,
            log_dir=log_dir
        )
        
        # Load the new checkpoints
        simple_checkpoint_path_new = os.path.join(os.getcwd(), "checkpoints/simpletransformer/simpletransformer_latest.pt")
        latent_checkpoint_path_new = os.path.join(os.getcwd(), "checkpoints/latenttransformer/latenttransformer_latest.pt")
        
        simple_checkpoint_new = torch.load(simple_checkpoint_path_new, map_location=device)
        latent_checkpoint_new = torch.load(latent_checkpoint_path_new, map_location=device)
        
        # Extract optimizer and scheduler info from second run
        simple_optimizer_info_second = extract_optimizer_info(simple_checkpoint_new['optimizer_state_dict'])
        simple_scheduler_info_second = extract_scheduler_info(simple_checkpoint_new['scheduler_state_dict'])
        
        latent_optimizer_info_second = extract_optimizer_info(latent_checkpoint_new['optimizer_state_dict'])
        latent_scheduler_info_second = extract_scheduler_info(latent_checkpoint_new['scheduler_state_dict'])
        
        # Record the step from checkpoint
        new_step_simple = simple_checkpoint_new.get("step", 0)
        new_step_latent = latent_checkpoint_new.get("step", 0)
        
        logger.info(f"New step from checkpoint - Simple: {new_step_simple}")
        logger.info(f"New step from checkpoint - Latent: {new_step_latent}")
        logger.info(f"Simple optimizer info (second run): {simple_optimizer_info_second}")
        logger.info(f"Simple scheduler info (second run): {simple_scheduler_info_second}")
        
        # Record the last learning rate after resumed training
        new_lr_simple = simple_checkpoint_new.get("last_lr", 0)
        new_lr_latent = latent_checkpoint_new.get("last_lr", 0)
        
        logger.info(f"New LR from checkpoint - Simple: {new_lr_simple}")
        logger.info(f"New LR from checkpoint - Latent: {new_lr_latent}")
        
        # Verify that the optimizer state contains momentum information from the previous run
        assert 'exp_avg_mean' in simple_optimizer_info_first, "Optimizer state missing momentum info in first run"
        assert 'exp_avg_mean' in simple_optimizer_info_second, "Optimizer state missing momentum info in second run"
        
        # Verify that states are properly continued
        assert new_step_simple == 30, f"Expected step 30, got {new_step_simple}"
        assert new_step_latent == 30, f"Expected step 30, got {new_step_latent}"
        
        # Verify that scheduler state is properly preserved
        assert simple_scheduler_info_second['last_epoch'] > simple_scheduler_info_first['last_epoch'], "Scheduler epoch not advanced"
        
        # Verify that learning rate follows expected scheduler pattern (should change according to the schedule)
        # This is a basic check, the exact change would depend on the scheduler
        logger.info("Learning rate comparison:")
        logger.info(f"Previous LR: {last_lr_simple}, New LR: {new_lr_simple}")
        
        # Verify that the steps have advanced correctly (checkpoint steps + new steps)
        # The actual step count may vary depending on how the counting is implemented
        assert results_resumed["simple"]["steps"] == 30, f"Expected 30 total steps, got {results_resumed['simple']['steps']}"
        
        logger.info("Checkpoint optimizer/scheduler persistence test passed successfully!")
        
    finally:
        # Clean up
        cleanup_test_environment()

if __name__ == "__main__":
    test_checkpoint_optimizer_persistence() 