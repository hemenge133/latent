"""
Unit tests for the utility functions in src/Utils.py
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
import sys
import numpy as np
from typing import List, Tuple, Dict, Any

# Add the project root to the system path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Utils import (
    save_checkpoint,
    load_checkpoint, 
    calculate_model_size,
    format_time,
    collate_fn,
    setup_training_dir
)

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


@pytest.fixture
def model() -> SimpleModel:
    """Create a simple model for testing"""
    return SimpleModel()


@pytest.fixture
def optimizer(model: SimpleModel) -> torch.optim.Optimizer:
    """Create an optimizer for the model"""
    return Adam(model.parameters(), lr=0.001)


@pytest.fixture
def temp_checkpoint_path(tmp_path) -> str:
    """Create a temporary path for checkpoint saving/loading"""
    checkpoint_dir = tmp_path / "test_checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir / "test_checkpoint.pt")


@pytest.mark.parametrize("seconds,expected", [
    (30, "30.0s"),
    (90, "1m 30s"),
    (3700, "1h 1m 40s"),
])
def test_format_time(seconds: float, expected: str) -> None:
    """Test the format_time function with various time values"""
    result = format_time(seconds)
    assert result == expected


def test_calculate_model_size(model: SimpleModel) -> None:
    """Test the calculate_model_size function for parameter counting"""
    # Calculate the expected number of parameters
    expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Use the utility function
    params = calculate_model_size(model)
    
    assert params == expected_params


def test_checkpoint_save_load(
    model: SimpleModel, optimizer: torch.optim.Optimizer, temp_checkpoint_path: str
) -> None:
    """Test saving and loading a checkpoint"""
    # Change model parameters to ensure they're different after loading
    for param in model.parameters():
        nn.init.normal_(param)

    # Save original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Save the checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        train_loss=0.5,
        val_loss=0.4,
        save_path=temp_checkpoint_path,
        is_best=False
    )
    
    # Change model parameters again
    for param in model.parameters():
        nn.init.uniform_(param)
    
    # Verify parameters have changed
    for name, param in model.named_parameters():
        assert not torch.allclose(param, original_params[name])
    
    # Load the checkpoint
    checkpoint = load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=temp_checkpoint_path,
        device=torch.device("cpu")
    )
    
    # Verify parameters have been restored
    for name, param in model.named_parameters():
        assert torch.allclose(param, original_params[name])
    
    # Check that checkpoint metadata is correct
    assert checkpoint["epoch"] == 10
    assert checkpoint["train_loss"] == 0.5
    assert checkpoint["val_loss"] == 0.4


def test_collate_fn() -> None:
    """Test the custom collate function for padding sequences"""
    # Create sample batch with varying sequence lengths
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])),
        (torch.tensor([8, 9]), torch.tensor([10, 11, 12])),
        (torch.tensor([13, 14, 15, 16]), torch.tensor([17, 18])),
    ]
    
    # Apply collate function
    inputs_padded, targets_padded, input_lens, target_lens = collate_fn(batch)
    
    # Check shapes
    assert inputs_padded.shape == (3, 4)
    assert targets_padded.shape == (3, 4)
    
    # Check padded values (should be 0s)
    assert inputs_padded[1, 2] == 0  # The third element of the second sequence should be padded
    assert inputs_padded[1, 3] == 0  # The fourth element of the second sequence should be padded
    
    # Check original lengths were preserved correctly
    assert input_lens == [3, 2, 4]
    assert target_lens == [4, 3, 2]
    
    # Check that original values are preserved
    assert torch.all(inputs_padded[0, :3] == torch.tensor([1, 2, 3]))
    assert torch.all(inputs_padded[1, :2] == torch.tensor([8, 9]))
    assert torch.all(inputs_padded[2, :4] == torch.tensor([13, 14, 15, 16]))


def test_setup_training_dir() -> None:
    """Test the setup_training_dir function for directory creation"""
    model_name = "test_model"
    expected_path = os.path.join("checkpoints", model_name)
    
    # Run the function
    result_path = setup_training_dir(model_name)
    
    # Check directory was created
    assert os.path.exists(expected_path)
    assert os.path.isdir(expected_path)
    
    # Check correct path was returned
    assert result_path == expected_path
    
    # Clean up
    try:
        os.rmdir(expected_path)
    except:
        pass  # Directory might contain files from other tests 