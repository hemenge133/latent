#!/usr/bin/env python3
"""
A simple script to manually test the step counter logic and resumed training
"""
import os
import sys
import torch
import time
import random
import numpy as np
import tempfile
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StepTest")


# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# Create a simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def train_model(model, optimizer, steps=5, start_step=0, checkpoint_path=None):
    """Train a model for a specified number of steps"""
    logger.info(f"Training model for {steps} steps starting from step {start_step}")

    # Define a loss function
    criterion = torch.nn.MSELoss()

    # Create dummy data
    input_size = 10
    batch_size = 32

    # Training loop
    for step in range(start_step, start_step + steps):
        # Generate random data
        inputs = torch.randn(batch_size, input_size)
        targets = torch.randn(batch_size, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"Step {step}: loss = {loss.item():.4f}")

        # Save checkpoint
        if checkpoint_path and (step + 1) % 2 == 0:
            checkpoint = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at step {step + 1}")

    return step + 1


def main():
    # Model parameters
    input_size = 10
    hidden_size = 20
    output_size = 1

    # Create model and optimizer
    model = SimpleModel(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create a temporary checkpoint file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp:
        checkpoint_path = temp.name
        logger.info(f"Created temporary checkpoint file: {checkpoint_path}")

    # First training phase
    logger.info("=== Phase 1: Initial Training ===")
    final_step = train_model(
        model, optimizer, steps=5, start_step=0, checkpoint_path=checkpoint_path
    )

    # Save final state for comparison
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    # Load the checkpoint and resume training
    logger.info("\n=== Phase 2: Resumed Training ===")
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Check the step
    if "step" in checkpoint:
        start_step = checkpoint["step"]
        logger.info(f"Checkpoint contains step: {start_step}")
    else:
        logger.warning("Checkpoint doesn't contain step information")
        start_step = 0

    # Create a new model and optimizer
    new_model = SimpleModel(input_size, hidden_size, output_size)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)

    # Load the state dictionaries
    new_model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Continue training
    final_step = train_model(
        new_model,
        new_optimizer,
        steps=5,
        start_step=start_step,
        checkpoint_path=checkpoint_path,
    )

    # Cleanup
    os.remove(checkpoint_path)
    logger.info(f"Removed temporary checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()
