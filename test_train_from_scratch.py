#!/usr/bin/env python
"""
Test script to run training from scratch without resuming from a checkpoint.
"""

import torch
import logging
import argparse
import sys
from loguru import logger
import os

# Setup logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("test_training.log", rotation="100 MB")  # Add file handler with rotation

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