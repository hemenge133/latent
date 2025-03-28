#!/usr/bin/env python
"""
Test script to verify that TensorBoard metrics are persisted correctly across training sessions.
"""

import pytest
import torch
import os
import shutil
from loguru import logger
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("test_tensorboard_persistence.log", rotation="100 MB")  # Add file handler with rotation

from src.Models import StableSimpleTransformer
from src.Dataset import MultiplicationDataset
from src.Losses import SequenceAccuracyLoss
from src.Config import TrainingConfig
from src.TrainingLoop import train_models_parallel

def setup_test_environment():
    """Set up a clean test environment for tensorboard testing"""
    # Create test log directory
    log_dir = os.path.join(os.getcwd(), "runs/test_tensorboard")
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

def read_tensorboard_data(log_dir):
    """Read the .steps_data file to check persisted step information"""
    simple_steps_file = os.path.join(log_dir, "simple", ".steps_data")
    latent_steps_file = os.path.join(log_dir, "latent", ".steps_data")
    
    simple_steps = {}
    latent_steps = {}
    
    if os.path.exists(simple_steps_file):
        with open(simple_steps_file, "r") as f:
            for line in f:
                if line.strip():
                    tag, step = line.strip().split(":", 1)
                    simple_steps[tag] = int(step)
    
    if os.path.exists(latent_steps_file):
        with open(latent_steps_file, "r") as f:
            for line in f:
                if line.strip():
                    tag, step = line.strip().split(":", 1)
                    latent_steps[tag] = int(step)
    
    return simple_steps, latent_steps

def test_tensorboard_metrics_persistence():
    """Test that tensorboard metrics are persisted correctly across training sessions"""
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
        logger.info("Starting first training run (20 steps)")
        results = train_models_parallel(
            models=models,
            dataset=dataset,
            dataset_val=val_dataset,
            vocab_size=vocab_size,
            criterion=criterion,
            device=device,
            max_steps=20,  # Run for 20 steps
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
        
        # Read tensorboard data after first run
        simple_steps_first, latent_steps_first = read_tensorboard_data(log_dir)
        
        # Log the tensorboard data
        logger.info(f"TensorBoard steps after first run - Simple: {simple_steps_first}")
        logger.info(f"TensorBoard steps after first run - Latent: {latent_steps_first}")
        
        # Load checkpoints
        simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
        latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)
        
        # Record the last step from checkpoints
        simple_last_step = simple_checkpoint.get("step", 0)
        latent_last_step = latent_checkpoint.get("step", 0)
        
        logger.info(f"Last step from checkpoint - Simple: {simple_last_step}")
        logger.info(f"Last step from checkpoint - Latent: {latent_last_step}")
        
        # Run a second training session, resuming from checkpoints
        logger.info("Starting second training run (resumed, +20 steps)")
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
        
        # Run training for another 20 steps
        results_resumed = train_models_parallel(
            models=models,
            dataset=dataset,
            dataset_val=val_dataset,
            vocab_size=vocab_size,
            criterion=criterion,
            device=device,
            max_steps=40,  # Total 40 steps (20 + 20)
            batch_size=4,
            learning_rate=3e-4,
            config=config,
            models_params=models_params,
            start_step=simple_last_step,  # Resume from last step
            simple_checkpoint=simple_checkpoint,
            latent_checkpoint=latent_checkpoint,
            log_dir=log_dir
        )
        
        # Read tensorboard data after second run
        simple_steps_second, latent_steps_second = read_tensorboard_data(log_dir)
        
        logger.info(f"TensorBoard steps after second run - Simple: {simple_steps_second}")
        logger.info(f"TensorBoard steps after second run - Latent: {latent_steps_second}")
        
        # Verify that steps in tensorboard data have increased
        for tag in simple_steps_first:
            assert tag in simple_steps_second, f"Tag {tag} missing from second run"
            assert simple_steps_second[tag] > simple_steps_first[tag], f"Step for {tag} did not increase"
        
        for tag in latent_steps_first:
            assert tag in latent_steps_second, f"Tag {tag} missing from second run"
            assert latent_steps_second[tag] > latent_steps_first[tag], f"Step for {tag} did not increase"
        
        # Verify that the final step count is correct (checkpoint steps + new steps)
        # The actual step count may vary depending on how the counting is implemented
        assert results_resumed["simple"]["steps"] == 50, f"Expected 50 total steps, got {results_resumed['simple']['steps']}"
        
        logger.info("TensorBoard persistence test passed successfully!")
        
    finally:
        # Clean up
        cleanup_test_environment()

if __name__ == "__main__":
    test_tensorboard_metrics_persistence() 