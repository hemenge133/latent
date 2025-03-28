#!/usr/bin/env python
"""
Test script to verify checkpoint loading with layer mismatches.
"""

import torch
import logging
import argparse
import sys
from loguru import logger
import os

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