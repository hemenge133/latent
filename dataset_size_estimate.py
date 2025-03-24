"""
Script to estimate the memory usage of the multiplication dataset
"""
import os
import psutil
import torch
import torch.utils.data
from Dataset import MultiplicationDataset
from Collate import collate_fn
from config import TrainingConfig

def format_size(size_bytes):
    """Format byte size to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def estimate_dataset_memory():
    """Estimate memory usage for the dataset"""
    config = TrainingConfig()
    print(f"Estimating memory usage for dataset with {config.total_samples} samples...")
    
    # Create sample datasets for each split
    train_dataset = MultiplicationDataset(config.total_samples, split='train', split_ratio=config.split_ratio)
    val_dataset = MultiplicationDataset(config.total_samples, split='val', split_ratio=config.split_ratio)
    test_dataset = MultiplicationDataset(config.total_samples, split='test', split_ratio=config.split_ratio)
    
    print(f"Dataset split sizes:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    # Memory tracking
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Measure memory usage of a small batch
    sample_size = min(100, len(train_dataset))
    sample_batch = [train_dataset[i] for i in range(sample_size)]
    inputs, targets = zip(*sample_batch)
    
    # Calculate average tensor size
    total_elements = sum(inp.nelement() + target.nelement() for inp, target in zip(inputs, targets))
    avg_elements_per_sample = total_elements / sample_size
    
    # Element size for LongTensor (4 bytes typically)
    element_size = inputs[0].element_size()
    
    # Estimate size for the entire dataset (only for the tensors)
    estimated_tensor_bytes = avg_elements_per_sample * element_size * config.total_samples
    
    # Measure DataLoader overhead with a small loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 to measure in the main process
    )
    
    # Load a batch to measure memory
    batch = next(iter(train_loader))
    
    # Calculate memory overhead
    current_memory = process.memory_info().rss
    memory_overhead = current_memory - initial_memory
    
    # Estimate memory with DataLoader
    batch_memory_usage = memory_overhead
    estimated_total_memory = batch_memory_usage * (len(train_dataset) / config.batch_size)
    
    # Output results
    print("\nMemory Usage Estimation:")
    print("-" * 50)
    print(f"Average tensor elements per sample: {avg_elements_per_sample:.2f}")
    print(f"Element size: {element_size} bytes")
    print(f"Estimated tensor memory for all samples: {format_size(estimated_tensor_bytes)}")
    
    print("\nDataLoader Memory Overhead:")
    print("-" * 50)
    print(f"Memory used for processing one batch: {format_size(batch_memory_usage)}")
    print(f"Estimated memory with DataLoader: {format_size(estimated_total_memory)}")
    
    # DataLoader with workers
    workers_estimate = estimated_total_memory * 1.5  # Approximate overhead for worker processes
    print(f"Estimated memory with DataLoader (with workers): {format_size(workers_estimate)}")
    
    # Check if within safe limits
    system_memory = psutil.virtual_memory().total
    print(f"\nSystem total memory: {format_size(system_memory)}")
    
    memory_usage_percent = (workers_estimate / system_memory) * 100
    print(f"Estimated peak memory usage: {memory_usage_percent:.1f}% of system memory")
    
    if memory_usage_percent < 50:
        print("VERDICT: Memory usage is within safe limits.")
    elif memory_usage_percent < 80:
        print("VERDICT: Memory usage is moderate. Monitor system during training.")
    else:
        print("VERDICT: Memory usage may be too high. Consider reducing batch size or dataset size.")
        max_safe_samples = int(config.total_samples * 50 / memory_usage_percent)
        print(f"Recommended maximum samples for safe operation: {max_safe_samples:,}")

def main():
    try:
        estimate_dataset_memory()
    except ImportError as e:
        print(f"Error: {e}")
        print("To run this script, install psutil: pip install psutil")
    except Exception as e:
        print(f"Error estimating dataset size: {e}")

if __name__ == "__main__":
    main() 