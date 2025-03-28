# Latent Transformer for Multiplication

This repository contains the implementation of SimpleTransformer and LatentTransformer models for learning multiplication. The LatentTransformer uses latent tokens to represent intermediate computation steps.

## Quickstart

```bash
# Set up the environment
./scripts/setup_env.sh

# Train a small model to quickly test functionality
./scripts/test_resume.sh

# Or start a full training run
./run_training.sh

# Monitor training progress
tensorboard --logdir=runs/parallel_comparison
```

## Technology Stack

- **Framework**: PyTorch for deep learning models
- **Data Processing**: NumPy, pandas for data manipulation
- **Parallel Processing**: Dask, joblib for efficient data processing
- **Logging**: Loguru for structured, configurable logging
- **Testing**: pytest for unit testing and validation
- **Progress Monitoring**: tqdm for progress tracking
- **Experiment Tracking**: TensorBoard for visualization

## Setup

To set up the environment:

```bash
./scripts/setup_env.sh
```

This will create a virtual environment, install dependencies, and set up the necessary directory structure.

## Training

To start a new training run:

```bash
./run_training.sh
```

Or with custom parameters:

```bash
python main.py --d-model 768 --num-layers 8 --num-latent 16 --min-digits 3 --max-steps 20000 --batch-size 512
```

### Training Arguments

- `--d-model`: Model dimension (default: 384)
- `--num-layers`: Number of transformer layers (default: 4)
- `--num-latent`: Number of latent tokens (default: 8)
- `--min-digits`: Minimum number of digits (default: 1)
- `--max-digits`: Maximum number of digits (default: 2)
- `--batch-size`: Batch size (default: 256)
- `--max-steps`: Maximum training steps (default: 10000)
- `--accuracy-weight`: Weight for accuracy in combined loss (default: 0.5)
- `--tf-schedule`: Teacher forcing schedule (linear, cosine, step) (default: linear)
- `--tf-start-step`: Teacher forcing reduction start step (default: 5000)
- `--use-checkpointing`: Use gradient checkpointing to save memory
- `--save-every`: Save checkpoint every N steps (default: every 50 steps)
- `--device`: Device to use for training (auto, cuda, mps, cpu) (default: auto)

## Development Practices

### Reproducibility

All training runs are fully reproducible through:

- Explicit random seed setting across all libraries (Python, NumPy, PyTorch)
- Fixed CUDA/cuDNN configurations for deterministic GPU operations
- Full state preservation in checkpoints (model, optimizer, RNG states)
- Configuration preservation across training runs

```python
# To ensure reproducibility in your experiments
from src.Training import set_seed

set_seed(42)  # Or any other seed value
```

### Performance Optimization

The codebase implements several optimizations:

- Mixed precision training (automatic FP16/BF16 where supported)
- Gradient checkpointing to reduce memory usage for large models
- Efficient data loading with pin_memory and num_workers optimization
- Parallel data processing using Dask and joblib
- GPU acceleration detection and utilization

### Code Quality

- Type hints throughout the codebase for better IDE support and error catching
- Comprehensive docstrings following PEP 8 guidelines
- Unit tests for critical components
- Modular architecture with clear separation of concerns

## Training Technologies

The implementation uses several advanced training techniques to ensure stable and efficient training:

### Optimizer and Learning Rate Management

- **AdamW**: Optimizer with weight decay that correctly handles L2 regularization
- **Cosine Annealing Schedule**: Gradually reduces learning rate to improve convergence
- **Warmup**: Linear learning rate warmup period to stabilize early training
- **Dynamic Gradient Clipping**: Adjusts clip norms based on current learning rate and training step
- **Gradient Checkpointing**: Optional memory-saving technique for training larger models

### Training Stability Features

- **Mixed Precision Training**: Automatic use of FP16 on CUDA devices for faster training
- **Loss Explosion Detection**: Automatic detection and response to unstable training
- **Emergency LR Reduction**: Automatic learning rate reduction if training becomes unstable
- **Emergency Checkpoints**: Saves model state before potential crashes due to instability
- **Teacher Forcing Schedule**: Gradually reduces reliance on ground truth during sequence generation

### Checkpoint and Resume System

- **Complete State Preservation**: Saves model, optimizer, and scheduler states
- **Run ID Management**: Identifies runs via unique IDs with embedded configuration
- **Configuration Preservation**: Maintains model configuration across training runs
- **Seamless Resuming**: Training can be paused and resumed with consistent learning rates

## Testing Checkpoint Resuming

To quickly test the resume functionality with a small model:

```bash
# Run the automated test script
./scripts/test_resume.sh
```

This script:
1. Trains a small model for 20 steps, saving checkpoints every 5 steps
2. Retrieves the latest run ID
3. Resumes training from step 20 to step 40 using the same model configuration

The script verifies that training can be paused and continued seamlessly, maintaining model state and continuing from the exact step where training was interrupted.

## Resuming Training

The system supports resuming training from checkpoints with complete state preservation:

```bash
# Resume the most recent training run
./scripts/resume_training.sh

# Resume a specific run by ID
./scripts/resume_training.sh <run_id>

# Resume a specific run but override its configuration with command-line parameters
./scripts/resume_training.sh <run_id> force
```

### What Gets Resumed

When resuming training, the following states are restored:

1. **Model Weights**: The exact state of the model weights
2. **Optimizer State**: All optimizer momentum and statistics
3. **Learning Rate Scheduler**: Learning rate state continues without disruption
4. **Training Step**: Continues from the saved step
5. **Model Configuration**: Uses the original configuration unless overridden

## Managing Runs

### Listing Runs

To list available runs:

```bash
./scripts/list_runs.py
```

Optional arguments:
- `--rescan`: Rescan runs directory
- `--id <run_id>`: Show details for specific run ID
- `--sort <method>`: Sort method: date, step, or loss
- `--limit <n>`: Limit number of runs to show

### Checkpoint Management

For synchronizing and fixing checkpoints:

```bash
# Check checkpoint status
python scripts/checkpoint_utils/check_sync_checkpoints.py

# Synchronize checkpoint steps 
python scripts/checkpoint_utils/check_sync_checkpoints.py --sync

# Update checkpoint to specific step
python scripts/checkpoint_utils/update_checkpoint_step.py <step>
```

## Model Efficiency Analysis

To analyze and compare the efficiency of SimpleTransformer and LatentTransformer models:

```bash
# Compare efficiency metrics between models using their checkpoints
python calculate_efficiency.py
```

This script:
1. Loads the best/latest checkpoints for both models
2. Compares parameter counts, validation losses, and accuracy metrics
3. Calculates efficiency metrics like loss-to-parameter ratios and accuracy-per-parameter
4. Provides a comparison showing which model is more parameter-efficient

## TensorBoard

To view training progress:

```bash
tensorboard --logdir=runs/parallel_comparison
```

## Project Structure

```
latent/
├── main.py                   # Main entry point for training
├── src/                      # Source code directory
│   ├── Config.py             # Configuration dataclasses
│   ├── Dataset.py            # Dataset implementation for multiplication
│   ├── Losses.py             # Custom loss functions
│   ├── Metrics.py            # Evaluation metrics
│   ├── Models.py             # Model definitions (SimpleTransformer, LatentTransformer)
│   ├── RunManagement.py      # Run tracking and management
│   ├── SummaryWriter.py      # TensorBoard logging utilities
│   ├── Training.py           # Training utilities
│   ├── TrainingLoop.py       # Main training loop implementation
│   └── Utils.py              # Utility functions
├── scripts/                  # Utility scripts
│   ├── checkpoint_utils/     # Checkpoint management utilities
│   ├── list_runs.py          # List available runs
│   ├── resume_training.sh    # Resume training
│   ├── setup_env.sh          # Environment setup
│   └── test_resume.sh        # Test resume functionality
├── checkpoints/              # Model checkpoint storage
├── runs/                     # TensorBoard logs and run information
└── requirements.txt          # Python dependencies
```

## Key Modules

- **Models.py**: Implements the SimpleTransformer and LatentTransformer architectures
- **TrainingLoop.py**: Handles the main training loop with evaluation and checkpointing
- **Dataset.py**: Provides data generation and processing for multiplication tasks
- **Metrics.py**: Implements evaluation metrics for model performance assessment
- **RunManagement.py**: Manages experiment runs with unique IDs and configuration tracking
- **Config.py**: Defines configuration parameters for models and training