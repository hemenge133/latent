# Transformer Architecture Comparison

This project compares two transformer architectures for sequence-to-sequence tasks, with a focus on the multiplication task:

1. **SimpleTransformer**: A standard encoder-decoder transformer architecture.
2. **LatentTransformer**: A modified architecture with a latent bottleneck between encoder and decoder.

## Project Structure

- `Transformer.py`: Contains implementations of both transformer architectures
- `Dataset.py`: Implementation of the multiplication dataset
- `Collate.py`: Collation function for batch processing
- `config.py`: Configuration parameters for training

### Comparison Scripts

- `fair_comparison.py`: Script for comparing both architectures with fair parameter settings
- `stable_comparison.py`: More numerically stable version of the comparison script 
- `inference_comparison.py`: Script for comparing inference results without training
- `transformer_comparison_results.md`: Summary of findings and recommendations

## Main Architectural Differences

1. **SimpleTransformer**:
   - Standard encoder-decoder transformer architecture
   - Direct attention from decoder to encoder memory
   - Parameter count scales linearly with input sequence length

2. **LatentTransformer**:
   - Uses a fixed set of latent tokens as an information bottleneck
   - Decoder only attends to latent tokens, not full encoder memory
   - Parameter count is less dependent on input sequence length

## Running the Comparison

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- NumPy
- tqdm

### Basic Usage

For a basic comparison:

```bash
python fair_comparison.py --d-model 128 --num-layers 3 --num-latent 8
```

For improved stability:

```bash
python stable_comparison.py --d-model 32 --num-layers 2 --num-latent 4 --max-steps 500
```

To compare inference without training:

```bash
python inference_comparison.py --use-dummy-weights
```

Using pre-trained models (if available):

```bash
python inference_comparison.py --simple-checkpoint=checkpoints/simple/best.pt --latent-checkpoint=checkpoints/latent/best.pt
```

## Key Findings

The theoretical advantages of the LatentTransformer include:

- Fixed computational cost for decoder regardless of input length
- Information distillation through latent tokens
- Reduced memory requirements for long sequences

However, practical challenges were encountered:

- Both models showed numerical stability issues
- MPS backend limitations affected transformer operations on Apple Silicon
- The additional complexity of LatentTransformer may not be justified for simpler tasks

For detailed findings, see the [comparison results](transformer_comparison_results.md).

## Future Work

- Evaluate on more stable hardware (CUDA GPU)
- Conduct proper hyperparameter tuning once stability issues are resolved
- Experiment with longer sequence tasks where LatentTransformer might show stronger advantages 