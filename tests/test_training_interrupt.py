import pytest
import torch
import os
from src.TrainingLoop import train_models_parallel
from src.Models import StableSimpleTransformer
from src.Dataset import MultiplicationDataset
from src.Losses import SequenceAccuracyLoss
from src.Config import TrainingConfig

def test_training_basic():
    """Test that training starts correctly and runs without errors"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create config with test-specific settings
    config = TrainingConfig()
    config.base_lr = 3e-4
    config.warmup_steps = 10  # Small warmup for testing
    config.weight_decay = 0.01
    config.batch_size = 4  # Small batch size for testing
    config.validate_every_n_steps = 10  # Validate more frequently for testing
    config.test_every_n_steps = 20  # Test more frequently for testing
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
    
    # Create a dummy latent model (we won't use it)
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
    results = train_models_parallel(
        models=models,
        dataset=dataset,
        dataset_val=val_dataset,
        vocab_size=vocab_size,
        criterion=criterion,
        device=device,
        max_steps=50,  # Just run for 50 steps
        batch_size=4,
        learning_rate=3e-4,
        config=config,
        models_params=models_params,
        start_step=0
    )
    
    # Verify that training completed and returned results
    assert results is not None, "Training failed to return results"
    assert "simple" in results, "Results missing simple model data"
    assert "latent" in results, "Results missing latent model data"
    assert "training_time" in results, "Results missing training time"
    
    # Verify that checkpoints were created
    simple_checkpoint_path = "checkpoints/simpletransformer/simpletransformer_latest.pt"
    assert os.path.exists(simple_checkpoint_path), "Checkpoint was not created"
    
    # Load the checkpoint and verify it contains the expected keys
    checkpoint = torch.load(simple_checkpoint_path)
    expected_keys = [
        'model_state_dict',
        'optimizer_state_dict',
        'scheduler_state_dict',
        'step',
        'val_loss',
        'config',
        'seed',
        'd_model',
        'last_lr'
    ]
    
    for key in expected_keys:
        assert key in checkpoint, f"Checkpoint missing expected key: {key}"
    
    # Verify the model state dict matches our model
    model_state = checkpoint['model_state_dict']
    for name, param in simple_model.named_parameters():
        assert name in model_state, f"Model state missing parameter: {name}"
        assert model_state[name].shape == param.shape, f"Shape mismatch for {name}"
    
    print("Test passed successfully!") 