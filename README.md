# Latent Transformer for Multiplication

This project implements and compares two transformer architectures for learning the multiplication task:
1. **SimpleTransformer**: A standard encoder-decoder transformer
2. **LatentTransformer**: A transformer that uses latent tokens as an intermediate representation

## Project Structure

```
├── Collate.py              # Collate function for batching sequences
├── compare_models.py       # Script to compare model performance
├── config.py               # Configuration parameters
├── Dataset.py              # Dataset implementation for multiplication task
├── dataset_size_estimate.py # Memory usage estimation script
├── train.py                # Training script for SimpleTransformer
├── train2.py               # Training script for LatentTransformer
├── Transformer.py          # Model implementations
├── utils.py                # Utility functions for training and evaluation
└── requirements.txt        # Package dependencies
```

## Requirements

- PyTorch
- tensorboard
- numpy
- tqdm
- psutil (for memory monitoring)

To install all required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of multiplication problems between two-digit numbers (e.g., "23*45") and their results (e.g., "1035"). The dataset implementation:

- Supports train/validation/test splits
- Generates samples on-the-fly for training to save memory
- Pre-computes validation and test samples for consistency
- Can handle large dataset sizes efficiently

To check if your system can handle the dataset size:
```bash
python dataset_size_estimate.py
```

## Training the Models

### SimpleTransformer

```bash
python train.py
```

### LatentTransformer

```bash
python train2.py
```

Both training scripts support the same configuration options defined in `config.py`. The training process includes:
- Learning rate warmup
- Gradient clipping
- Early stopping
- Model checkpointing
- Validation and test set evaluation

## Monitoring Training Progress

Training metrics are logged to TensorBoard:
```bash
tensorboard --logdir runs
```

This will display:
- Training loss curves
- Validation loss curves
- Test loss (evaluated periodically)
- Learning rate schedule

## Comparing Models

After training both models, you can compare their performance with:

```bash
python compare_models.py
```

This script will provide:
- A summary of the best validation loss for each model
- The latest training and validation metrics
- Direct TensorBoard command for detailed visualization

## Architecture Details

### SimpleTransformer

A standard encoder-decoder transformer architecture with:
- Encoder: Processes the input sequence (e.g., "23*45")
- Decoder: Generates the output sequence (e.g., "1035")

### LatentTransformer

A modified transformer architecture that introduces latent tokens:
- Encoder: Processes the input sequence
- Latent Tokens: A fixed set of learnable tokens that attend to the encoder output
- Decoder: Attends to the latent tokens (not directly to the encoder output) to generate the output

The key innovation is that the latent tokens serve as an information bottleneck, forcing the model to compress the relevant information about the task.

## Checkpoints

Model checkpoints are saved in the `checkpoints/` directory with the following structure:
- `checkpoints/simple_transformer/simple_transformer_best.pt`: Best SimpleTransformer model
- `checkpoints/latent_transformer/latent_transformer_best.pt`: Best LatentTransformer model

Regular checkpoints are also saved every N epochs for each model. 