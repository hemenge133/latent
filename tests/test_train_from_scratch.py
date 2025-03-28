#!/usr/bin/env python
"""
Test script to run training from scratch without resuming from a checkpoint.
"""

import torch
import logging
import argparse
import sys
import os
import pytest
import subprocess
from loguru import logger

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("test_training.log", rotation="100 MB")  # Add file handler with rotation

def test_training_from_scratch():
    """Test training a small model from scratch for a few steps"""
    # Run for minimal steps to test functionality
    cmd = ["python", "main.py", 
           "--d-model", "64", 
           "--num-layers", "2", 
           "--num-latent", "4", 
           "--max-steps", "5",
           "--min-digits", "1",
           "--max-digits", "1",
           "--batch-size", "16"]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check that the process completed successfully
    assert result.returncode == 0, f"Training failed with error: {result.stderr}"
    logger.info("Training from scratch completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Test training without resuming from a checkpoint')
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num-latent', type=int, default=4, help='Number of latent tokens')
    parser.add_argument('--max-steps', type=int, default=10, help='Maximum training steps')
    args = parser.parse_args()
    
    # Run the training without resume flag
    cmd = f"python main.py --d-model {args.d_model} --num-layers {args.num_layers} --num-latent {args.num_latent} --max-steps {args.max_steps}"
    logger.info(f"Running command: {cmd}")
    os.system(cmd)
    logger.info("Training from scratch completed")

if __name__ == "__main__":
    main() 