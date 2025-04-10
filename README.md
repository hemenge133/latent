# Latent Transformer for Multiplication

This repository contains the implementation of SimpleTransformer and LatentTransformer models for learning multiplication, along with tools for training, evaluation, and hyperparameter optimization.

## Quickstart

```bash
# Set up the environment (creates venv, installs deps)
./scripts/setup_env.sh

# Activate the environment (using virtualenvwrapper)
workon latent

# Train a small model to quickly test functionality
python main.py --d-model 64 --num-layers 2 --num-latent 4 --max-steps 100 --batch-size 32 --save-every 10

# Run hyperparameter optimization using Optuna
python optimize.py --n-trials 10 --storage sqlite:///optuna_studies/multiplication_study.db --study-name "initial_optimization"

# Monitor training/optimization progress
tensorboard --logdir=runs
```

## Technology Stack

- **Framework**: PyTorch
- **Optimization**: Optuna
- **Data Processing**: NumPy, pandas
- **Parallel Processing**: Dask, joblib
- **Logging**: Loguru
- **Testing**: pytest
- **Monitoring**: TensorBoard, tqdm

## Setup

Ensure you have Python 3.x installed. Set up the environment using the provided script:

```bash
./scripts/setup_env.sh
```

This script creates a Python virtual environment named `latent` (compatible with `virtualenvwrapper`), installs dependencies from `requirements.txt`, and sets up necessary directories. Activate the environment using `workon latent`.

## Training

Start a new training run using `main.py`. Key configuration parameters can be passed as command-line arguments.

```bash
# Example: Train a medium-sized model
python main.py --d-model 384 --num-layers 4 --num-latent 8 --max-steps 10000 --batch-size 256

# For a full list of arguments and their defaults
python main.py --help
```

### Resuming Training

Training can be resumed from the latest checkpoint automatically or from a specific run ID.

```bash
# Resume the most recently modified run
./scripts/resume_training.sh

# Resume a specific run by ID
./scripts/resume_training.sh <run_id>

# Resume and force override config with CLI args
./scripts/resume_training.sh <run_id> force
```

The resume system restores model weights, optimizer state, scheduler state, and the exact training step.

## Hyperparameter Optimization (Optuna)

Use `optimize.py` to perform hyperparameter optimization using Optuna. Studies are stored in an SQLite database.

```bash
# Start an optimization study
python optimize.py --n-trials 50 --storage sqlite:///optuna_studies/multiplication_study.db --study-name "latent_transformer_opt"

# Key optimization arguments:
# --n-trials: Number of optimization trials to run.
# --storage: Optuna storage URL (e.g., database file).
# --study-name: Name for the Optuna study.
# --pruner: Optuna pruner to use (e.g., 'median', 'hyperband').
# --sampler: Optuna sampler to use (e.g., 'tpe', 'random').
# --metric: Metric to optimize ('val_loss', 'val_accuracy', 'val_combined_loss').
# --direction: Optimization direction ('minimize', 'maximize').

# For a full list of arguments
python optimize.py --help
```

Results and logs for each trial are stored within the `runs/` directory, similar to regular training runs.

## Monitoring

Training progress, including losses, metrics, and hyperparameters, can be monitored using TensorBoard.

```bash
tensorboard --logdir=runs
```

Optuna study progress can also be monitored via tools compatible with its storage backend (e.g., Optuna Dashboard for SQLite).

## Key Features

- **Models**: `SimpleTransformer` and `LatentTransformer` implementations.
- **Reproducibility**: Ensures runs are reproducible via random seed management and full state checkpointing.
- **Performance**: Includes mixed precision training, gradient checkpointing, and efficient data loading.
- **Stability**: Implements dynamic gradient clipping, loss explosion detection, and teacher forcing schedules.
- **Run Management**: Tracks runs with unique IDs and configuration preservation. Use `./scripts/list_runs.py` to view run details.
- **Checkpointing**: Saves complete training state (model, optimizer, scheduler, RNGs) for seamless resumption.

## Project Structure

```
latent/
├── main.py                 # Main training script
├── optimize.py             # Optuna hyperparameter optimization script
├── src/                    # Source code
│   ├── Config.py           # Configuration dataclasses
│   ├── Dataset.py          # Data generation and loading
│   ├── Losses.py           # Loss functions
│   ├── Metrics.py          # Evaluation metrics
│   ├── Models.py           # Transformer model definitions
│   ├── RunManagement.py    # Run tracking and management
│   ├── SummaryWriter.py    # TensorBoard logging utilities
│   ├── Training.py         # Training helpers (seed, device, etc.)
│   ├── TrainingLoop.py     # Core training loop logic
│   ├── OptimizationLoop.py # Optuna objective function logic
│   └── Utils.py            # General utilities
├── scripts/                # Utility scripts (setup, resume, list runs)
│   ├── checkpoint_utils/   # Checkpoint management tools
│   ├── list_runs.py
│   ├── resume_training.sh
│   ├── setup_env.sh
│   └── test_resume.sh
├── checkpoints/            # Saved model checkpoints (organized by run_id)
├── runs/                   # TensorBoard logs and run metadata (organized by run_id)
├── optuna_studies/         # Optuna study databases
└── requirements.txt        # Python dependencies
```

## Key Modules

- **Models.py**: Implements the SimpleTransformer and LatentTransformer architectures
- **TrainingLoop.py**: Handles the main training loop with evaluation and checkpointing
- **Dataset.py**: Provides data generation and processing for multiplication tasks
- **Metrics.py**: Implements evaluation metrics for model performance assessment
- **RunManagement.py**: Manages experiment runs with unique IDs and configuration tracking
- **Config.py**: Defines configuration parameters for models and training