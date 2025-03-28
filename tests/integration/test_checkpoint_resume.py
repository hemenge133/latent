#!/usr/bin/env python
"""
Integration test for checkpoint loading and resuming, skipping actual training for faster tests.

This test verifies that:
1. Dimensions are correctly extracted from checkpoints
2. Models are recreated with correct dimensions
3. Checkpoint loading works with the new dimensions
"""

import os
import sys
import torch
import pytest
from pathlib import Path
from loguru import logger

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import extract_model_dimensions
from src.Models import StableSimpleTransformer, StableLatentTransformer

# Initialize logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("integration_test.log", rotation="100 MB")  # Add file handler with rotation

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
        'step': 100,
        'epoch': 1,
        'optimizer_state_dict': {'param_groups': [{'lr': 0.001}]},
        'scheduler_state_dict': {'last_epoch': 100}
    }
    
    return checkpoint

@pytest.mark.integration
def test_checkpoint_resume_integration():
    """Integration test for checkpoint loading and resuming"""
    try:
        # Create a mock checkpoint for SimpleTransformer
        logger.info("Creating mock checkpoint for SimpleTransformer")
        simple_checkpoint = create_mock_checkpoint(
            d_model=64, 
            num_layers=2, 
            num_latent=0, 
            vocab_size=12, 
            max_len=20
        )
        
        # Create a mock checkpoint for LatentTransformer
        logger.info("Creating mock checkpoint for LatentTransformer")
        latent_checkpoint = create_mock_checkpoint(
            d_model=64, 
            num_layers=2, 
            num_latent=4, 
            vocab_size=12, 
            max_len=20
        )
        
        # Extract dimensions from checkpoints
        simple_dimensions = extract_model_dimensions(simple_checkpoint)
        latent_dimensions = extract_model_dimensions(latent_checkpoint)
        
        logger.info(f"SimpleTransformer dimensions: {simple_dimensions}")
        logger.info(f"LatentTransformer dimensions: {latent_dimensions}")
        
        # Create models with wrong dimensions intentionally
        logger.info("Creating models with wrong dimensions")
        wrong_d_model = 32  # Different from checkpoint
        wrong_num_layers = 1  # Different from checkpoint
        wrong_num_latent = 2  # Different from checkpoint
        
        # SimpleTransformer with wrong dimensions
        simple_model = StableSimpleTransformer(
            vocab_size=12,
            d_model=wrong_d_model,
            nhead=8,
            num_layers=wrong_num_layers,
            dropout=0.1
        )
        
        # LatentTransformer with wrong dimensions
        latent_model = StableLatentTransformer(
            vocab_size=12,
            d_model=wrong_d_model,
            nhead=8,
            num_layers=wrong_num_layers,
            num_latent=wrong_num_latent,
            dropout=0.1
        )
        
        # Verify dimensions don't match
        assert simple_model.d_model != simple_dimensions['d_model'], "SimpleTransformer d_model should not match yet"
        assert len(simple_model.encoder.layers) != simple_dimensions['num_layers'], "SimpleTransformer num_layers should not match yet"
        assert latent_model.num_latent != latent_dimensions['num_latent'], "LatentTransformer num_latent should not match yet"
        
        # Now recreate models with correct dimensions from checkpoints
        logger.info("Recreating models with correct dimensions")
        simple_model = StableSimpleTransformer(
            vocab_size=simple_dimensions['vocab_size'],
            d_model=simple_dimensions['d_model'],
            nhead=8,
            num_layers=simple_dimensions['num_layers'],
            dropout=0.1
        )
        
        latent_model = StableLatentTransformer(
            vocab_size=latent_dimensions['vocab_size'],
            d_model=latent_dimensions['d_model'],
            nhead=8,
            num_layers=latent_dimensions['num_layers'],
            num_latent=latent_dimensions['num_latent'],
            dropout=0.1
        )
        
        # Verify dimensions now match
        assert simple_model.d_model == simple_dimensions['d_model'], f"SimpleTransformer d_model should match: {simple_model.d_model} vs {simple_dimensions['d_model']}"
        assert len(simple_model.encoder.layers) == simple_dimensions['num_layers'], f"SimpleTransformer num_layers should match: {len(simple_model.encoder.layers)} vs {simple_dimensions['num_layers']}"
        assert latent_model.num_latent == latent_dimensions['num_latent'], f"LatentTransformer num_latent should match: {latent_model.num_latent} vs {latent_dimensions['num_latent']}"
        
        # Test loading state_dict for SimpleTransformer
        logger.info("Testing state_dict loading for SimpleTransformer")
        simple_model.load_state_dict(simple_checkpoint['model_state_dict'], strict=False)
        
        # Test loading state_dict for LatentTransformer
        logger.info("Testing state_dict loading for LatentTransformer")
        latent_model.load_state_dict(latent_checkpoint['model_state_dict'], strict=False)
        
        # Test positional encoder dimension handling
        logger.info("Testing positional encoder dimension handling")
        # Create a checkpoint with a different max_len
        custom_max_len = 30  # Different from default 20
        custom_checkpoint = create_mock_checkpoint(
            d_model=64,
            num_layers=2,
            num_latent=0,
            vocab_size=12,
            max_len=custom_max_len
        )
        
        # Create a model with default max_len
        simple_model = StableSimpleTransformer(
            vocab_size=12,
            d_model=64,
            nhead=8,
            num_layers=2,
            dropout=0.1
        )
        
        # Verify max_len is different
        assert simple_model.max_len != custom_max_len, f"Model max_len should be different from checkpoint: {simple_model.max_len} vs {custom_max_len}"
        
        # Fix the pos_encoder to match the checkpoint
        pos_encoder_tensor = custom_checkpoint['model_state_dict']['pos_encoder']
        simple_model.max_len = pos_encoder_tensor.shape[0]
        simple_model.pos_encoder = torch.nn.Parameter(torch.zeros_like(pos_encoder_tensor))
        
        # Verify max_len now matches
        assert simple_model.max_len == custom_max_len, f"Model max_len should now match checkpoint: {simple_model.max_len} vs {custom_max_len}"
        
        # Now loading should work
        simple_model.load_state_dict(custom_checkpoint['model_state_dict'], strict=False)
        
        logger.info("Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Test failed with exception: {str(e)}")
        assert False, f"Test failed with exception: {str(e)}"

if __name__ == "__main__":
    test_checkpoint_resume_integration() 