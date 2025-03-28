#!/usr/bin/env python
"""
Unit test for the extract_model_dimensions function in main.py.
"""

import os
import sys
import torch
import pytest
from pathlib import Path
from loguru import logger

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import extract_model_dimensions

# Initialize logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler

def create_mock_checkpoint(d_model=64, num_layers=2, num_latent=4, vocab_size=12, max_len=20):
    """Create a mock checkpoint for testing with specified dimensions."""
    # Create state dict with embedding weight tensor of appropriate shape
    state_dict = {
        'embed.weight': torch.zeros(vocab_size, d_model),
        'pos_encoder': torch.zeros(max_len, d_model),
    }
    
    # Add layer-specific keys
    for i in range(num_layers):
        state_dict[f'encoder.layers.{i}.self_attn.in_proj_weight'] = torch.zeros(3 * d_model, d_model)
    
    # For latent model
    if num_latent > 0:
        state_dict['latent_tokens'] = torch.zeros(num_latent, d_model)
    
    # Wrap in a proper checkpoint structure
    checkpoint = {
        'model_state_dict': state_dict,
        'config': {
            'd_model': d_model,
            'num_layers': num_layers,
            'vocab_size': vocab_size,
            'num_latent': num_latent,
            'max_len': max_len
        },
        'step': 100
    }
    
    return checkpoint

def test_extract_dimensions_from_config():
    """Test that dimensions are correctly extracted from the config."""
    # Create a mock checkpoint with config
    checkpoint = create_mock_checkpoint(d_model=128, num_layers=4, num_latent=8, vocab_size=16, max_len=30)
    
    # Extract dimensions
    dimensions = extract_model_dimensions(checkpoint)
    
    # Verify dimensions match config
    assert dimensions['d_model'] == 128, f"d_model mismatch: {dimensions['d_model']} != 128"
    assert dimensions['num_layers'] == 4, f"num_layers mismatch: {dimensions['num_layers']} != 4"
    assert dimensions['num_latent'] == 8, f"num_latent mismatch: {dimensions['num_latent']} != 8"
    assert dimensions['vocab_size'] == 16, f"vocab_size mismatch: {dimensions['vocab_size']} != 16"
    assert dimensions['max_len'] == 30, f"max_len mismatch: {dimensions['max_len']} != 30"
    
    logger.info("Dimensions extracted correctly from config")

def test_extract_dimensions_from_state_dict():
    """Test that dimensions are correctly extracted from the state dict when config is not available."""
    # Create a mock checkpoint with no config
    checkpoint = create_mock_checkpoint(d_model=96, num_layers=3, num_latent=6, vocab_size=20, max_len=25)
    state_dict = checkpoint['model_state_dict']
    
    # Remove config to simulate older checkpoint format
    checkpoint_no_config = {'model_state_dict': state_dict}
    
    # Extract dimensions
    dimensions = extract_model_dimensions(checkpoint_no_config)
    
    # Verify dimensions match those used to create the state dict
    assert dimensions['d_model'] == 96, f"d_model mismatch: {dimensions['d_model']} != 96"
    assert dimensions['num_layers'] == 3, f"num_layers mismatch: {dimensions['num_layers']} != 3"
    assert dimensions['vocab_size'] == 20, f"vocab_size mismatch: {dimensions['vocab_size']} != 20"
    assert dimensions['max_len'] == 25, f"max_len mismatch: {dimensions['max_len']} != 25"
    
    logger.info("Dimensions extracted correctly from state dict")

def test_extract_dimensions_with_direct_state_dict():
    """Test that dimensions are correctly extracted from a direct state dict (not wrapped in 'model_state_dict')."""
    # Create a mock checkpoint
    checkpoint = create_mock_checkpoint(d_model=64, num_layers=2, num_latent=4, vocab_size=12, max_len=20)
    
    # Extract the state dict directly
    direct_state_dict = checkpoint['model_state_dict']
    
    # Extract dimensions
    dimensions = extract_model_dimensions(direct_state_dict)
    
    # Verify dimensions match
    assert dimensions['d_model'] == 64, f"d_model mismatch: {dimensions['d_model']} != 64"
    assert dimensions['num_layers'] == 2, f"num_layers mismatch: {dimensions['num_layers']} != 2"
    assert dimensions['vocab_size'] == 12, f"vocab_size mismatch: {dimensions['vocab_size']} != 12"
    assert dimensions['max_len'] == 20, f"max_len mismatch: {dimensions['max_len']} != 20"
    
    logger.info("Dimensions extracted correctly from direct state dict")

def test_extract_dimensions_with_real_checkpoint():
    """Test extracting dimensions from a real checkpoint if available."""
    # Check if a real checkpoint exists
    simple_checkpoint_path = Path("checkpoints/simpletransformer/simpletransformer_latest.pt")
    if not simple_checkpoint_path.exists():
        logger.warning(f"No real checkpoint found at {simple_checkpoint_path}. Skipping test.")
        pytest.skip(f"No real checkpoint found at {simple_checkpoint_path}")
        return
    
    # Load the real checkpoint
    device = torch.device("cpu")
    checkpoint = torch.load(simple_checkpoint_path, map_location=device)
    
    # Extract dimensions
    dimensions = extract_model_dimensions(checkpoint)
    
    # Log the extracted dimensions
    logger.info(f"Dimensions extracted from real checkpoint: {dimensions}")
    
    # Verify that essential dimensions are not None
    assert dimensions['d_model'] is not None, "d_model is None"
    assert dimensions['num_layers'] is not None, "num_layers is None"
    assert dimensions['vocab_size'] is not None, "vocab_size is None"
    assert dimensions['max_len'] is not None, "max_len is None"
    
    logger.info("Dimensions extracted correctly from real checkpoint")

if __name__ == "__main__":
    test_extract_dimensions_from_config()
    test_extract_dimensions_from_state_dict()
    test_extract_dimensions_with_direct_state_dict()
    test_extract_dimensions_with_real_checkpoint() 