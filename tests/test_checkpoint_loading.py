#!/usr/bin/env python
"""
Test script to verify checkpoint loading with layer mismatches.
"""

import torch
import logging
import argparse
import sys
import os
import pytest
from loguru import logger

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Models import StableSimpleTransformer, StableLatentTransformer

# Initialize logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler

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

@pytest.mark.skipif(not os.path.exists('checkpoints/SimpleTransformer/latest.pt'), 
                    reason="Requires a checkpoint file to test")
def test_checkpoint_loading_with_layer_mismatch():
    """Test loading a checkpoint with layer mismatch using filtering"""
    # Get the latest checkpoint file
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints/SimpleTransformer/')
    if not os.path.exists(checkpoints_dir):
        pytest.skip("No checkpoints directory found")
    
    checkpoint_file = os.path.join(checkpoints_dir, 'latest.pt')
    if not os.path.exists(checkpoint_file):
        pytest.skip("No latest.pt checkpoint found")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Print the number of layers in the checkpoint if we can detect it
    layer_indices = []
    for key in checkpoint.get("model_state_dict", checkpoint).keys():
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
        checkpoint_num_layers = max_layer_idx + 1  # +1 because indexing starts at 0
        logger.info(f"Detected {checkpoint_num_layers} layers in the checkpoint")
        
        # Create model with fewer layers
        test_layers = max(1, checkpoint_num_layers - 1)
    else:
        # Default to 2 layers if we can't detect
        test_layers = 2
    
    # Get vocab size from checkpoint if possible
    vocab_size = 11  # Default
    if 'config' in checkpoint and 'vocab_size' in checkpoint['config']:
        vocab_size = checkpoint['config']['vocab_size']
    elif 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    
    d_model = 64  # Default small value
    if 'config' in checkpoint and 'd_model' in checkpoint['config']:
        d_model = checkpoint['config']['d_model']
    
    # Create a model with fewer layers than the checkpoint
    logger.info(f"Creating SimpleTransformer with {test_layers} layers (d_model={d_model})")
    model = StableSimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=test_layers,
        dropout=0.1,
    )
    
    # Now try with our filter_state_dict_for_model function and strict=False
    try:
        logger.info("Trying with filter_state_dict_for_model and strict=False")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        filtered_state_dict = filter_state_dict_for_model(state_dict, model)
        logger.info(f"Filtered state dict has {len(filtered_state_dict)} keys")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loading with filtered state dict and strict=False succeeded")
        
        # Assertion to verify the test passes
        assert True, "Checkpoint loading with filtered state dict succeeded"
    except Exception as e:
        logger.error(f"Filtered loading failed: {str(e)}")
        assert False, f"Checkpoint loading failed: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Test checkpoint loading with layer mismatch')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--d-model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers (less than in checkpoint)')
    parser.add_argument('--vocab-size', type=int, default=11, help='Vocabulary size')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Print the number of layers in the checkpoint if we can detect it
    layer_indices = []
    for key in checkpoint.get("model_state_dict", checkpoint).keys():
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
        checkpoint_num_layers = max_layer_idx + 1  # +1 because indexing starts at 0
        logger.info(f"Detected {checkpoint_num_layers} layers in the checkpoint")
    
    # Create a model with fewer layers than the checkpoint
    logger.info(f"Creating SimpleTransformer with {args.num_layers} layers (d_model={args.d_model})")
    model = StableSimpleTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    
    # Attempt to load with strict=True (this should fail with layer mismatch)
    try:
        logger.info("Attempting to load with strict=True (should fail if layers mismatch)")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Strict loading succeeded (no layer mismatch or keys already match)")
    except Exception as e:
        logger.error(f"Strict loading failed as expected: {str(e)}")
    
    # Now try with our filter_state_dict_for_model function and strict=False
    try:
        logger.info("Trying with filter_state_dict_for_model and strict=False")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        filtered_state_dict = filter_state_dict_for_model(state_dict, model)
        logger.info(f"Filtered state dict has {len(filtered_state_dict)} keys")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loading with filtered state dict and strict=False succeeded")
    except Exception as e:
        logger.error(f"Filtered loading failed: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 