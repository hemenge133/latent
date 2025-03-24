# Latent Transformer vs. Simple Transformer

This project compares two transformer architectures for sequence-to-sequence tasks, with a focus on the multiplication task:

1. **SimpleTransformer**: A standard encoder-decoder transformer architecture.
2. **LatentTransformer**: A modified architecture with a latent bottleneck between encoder and decoder.

## Architecture Comparison

The key concept being tested is whether a latent bottleneck can provide advantages in terms of:
- Parameter efficiency
- Generalization ability
- Computational performance

### SimpleTransformer
- Standard encoder-decoder transformer architecture
- Direct attention from decoder to encoder memory
- Parameter count scales with input sequence length

### LatentTransformer
- Uses a fixed set of latent tokens as an information bottleneck
- The latent tokens function as an "abstract thought" layer
- Forces the model to compress information before decoding
- Parameter count grows less with input sequence length
- Potentially better generalization from the compression effect

## Project Structure

The codebase has been streamlined to focus on the core comparison:

- `parallel_comparison.py`: Main training script to compare both models simultaneously
- `calculate_efficiency.py`: Script to calculate efficiency metrics between models
- `stable_comparison_with_accuracy.py`: Contains stable implementations of both architectures
- `Dataset.py`: Implementation of the multiplication dataset
- `utils.py`: Utility functions including collation and evaluation
- `config.py`: Configuration parameters for training

## Running the Comparison

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU recommended

### Installation

```bash
pip install -r requirements.txt
```

### Running Training

For 3-digit multiplication with 512-dimensional models and 32 latent tokens:

```bash
python parallel_comparison.py --d-model 512 --num-layers 6 --num-latent 32 --max-steps 100000 --min-digits 3 --device cuda --use-checkpointing
```

For 1-digit multiplication (faster for testing):

```bash
python parallel_comparison.py --d-model 128 --num-layers 4 --num-latent 8 --max-steps 10000 --min-digits 1 --device cuda
```

### Calculating Efficiency

After training or when interrupted with Ctrl+C:

```bash
python calculate_efficiency.py
```

## Features

- **Parallel Training**: Both models trained simultaneously for fair comparison
- **TensorBoard Integration**: Comprehensive metrics tracking including:
  - Loss curves
  - Learning rates
  - Percentage of completely correct predictions
  - Generalization accuracy
- **Ctrl+C Handling**: Gracefully calculates efficiency metrics when interrupted
- **Stable Implementation**: Numerical stability enhancements for reliable training
- **Checkpoint Management**: Automatically saves best model checkpoints

## Experiment Results

The effectiveness of the latent bottleneck varies by task complexity:

- For simple tasks (1-digit multiplication), SimpleTransformer often converges faster
- For complex tasks (3-digit multiplication), LatentTransformer can show parameter efficiency advantages
- The number of latent tokens significantly impacts model performance on complex tasks

## Viewing Results

```bash
tensorboard --logdir=runs
```

This will show training curves and metrics for both models, allowing side-by-side comparison. 