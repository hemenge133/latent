"""
Script to generate simple model comparison summary using TensorBoard data
"""
import os
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from config import TrainingConfig

def get_latest_tensorboard_data(log_dir, tag, max_entries=5):
    """
    Get the latest TensorBoard entries for a specific tag
    
    Args:
        log_dir: Directory containing TensorBoard logs
        tag: The specific tag to extract (e.g., 'Loss/val')
        max_entries: Maximum number of latest entries to retrieve
        
    Returns:
        List of tuples (step, value)
    """
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        if tag not in event_acc.Tags().get('scalars', []):
            print(f"Tag {tag} not found in TensorBoard logs at {log_dir}")
            return []
        
        events = event_acc.Scalars(tag)
        # Get the last max_entries entries
        latest_events = events[-max_entries:] if len(events) > max_entries else events
        return [(event.step, event.value) for event in latest_events]
    except Exception as e:
        print(f"Error reading TensorBoard data from {log_dir}: {e}")
        return []

def get_model_performance_summary(config):
    """
    Get a summary of the best performance for both models
    """
    results = {}
    
    # SimpleTransformer results
    simple_checkpoint_path = os.path.join(
        config.checkpoints_dir, 
        config.simple_transformer_name,
        f"{config.simple_transformer_name}_best.pt"
    )
    
    if os.path.exists(simple_checkpoint_path):
        try:
            checkpoint = torch.load(simple_checkpoint_path, map_location='cpu')
            results["SimpleTransformer"] = {
                "train_loss": checkpoint.get("train_loss", float('nan')),
                "val_loss": checkpoint.get("val_loss", float('nan')),
                "epoch": checkpoint.get("epoch", -1),
            }
        except Exception as e:
            results["SimpleTransformer"] = {"status": f"Error loading checkpoint: {e}"}
    else:
        results["SimpleTransformer"] = {"status": "No checkpoint found"}
    
    # LatentTransformer results
    latent_checkpoint_path = os.path.join(
        config.checkpoints_dir, 
        config.latent_transformer_name,
        f"{config.latent_transformer_name}_best.pt"
    )
    
    if os.path.exists(latent_checkpoint_path):
        try:
            checkpoint = torch.load(latent_checkpoint_path, map_location='cpu')
            results["LatentTransformer"] = {
                "train_loss": checkpoint.get("train_loss", float('nan')),
                "val_loss": checkpoint.get("val_loss", float('nan')),
                "epoch": checkpoint.get("epoch", -1),
            }
        except Exception as e:
            results["LatentTransformer"] = {"status": f"Error loading checkpoint: {e}"}
    else:
        results["LatentTransformer"] = {"status": "No checkpoint found"}
    
    return results

def print_latest_tensorboard_metrics(config):
    """
    Print the latest metrics from TensorBoard for both models
    """
    print("\nLatest TensorBoard Metrics:")
    print("=" * 50)
    
    # SimpleTransformer
    print("\nSimpleTransformer latest metrics:")
    simple_train = get_latest_tensorboard_data(config.simple_transformer_logdir, 'Loss/train_epoch')
    simple_val = get_latest_tensorboard_data(config.simple_transformer_logdir, 'Loss/val')
    
    if simple_train:
        print("  Train Loss (last 5 epochs):")
        for step, value in simple_train:
            print(f"    Epoch {step}: {value:.6f}")
    else:
        print("  No training loss data found")
    
    if simple_val:
        print("  Validation Loss (last 5 epochs):")
        for step, value in simple_val:
            print(f"    Epoch {step}: {value:.6f}")
    else:
        print("  No validation loss data found")
    
    # LatentTransformer
    print("\nLatentTransformer latest metrics:")
    latent_train = get_latest_tensorboard_data(config.latent_transformer_logdir, 'Loss/train_epoch')
    latent_val = get_latest_tensorboard_data(config.latent_transformer_logdir, 'Loss/val')
    
    if latent_train:
        print("  Train Loss (last 5 epochs):")
        for step, value in latent_train:
            print(f"    Epoch {step}: {value:.6f}")
    else:
        print("  No training loss data found")
    
    if latent_val:
        print("  Validation Loss (last 5 epochs):")
        for step, value in latent_val:
            print(f"    Epoch {step}: {value:.6f}")
    else:
        print("  No validation loss data found")

def main():
    config = TrainingConfig()
    
    # Print TensorBoard command for convenience
    print("\nTo view detailed training progress with TensorBoard, run:")
    print("=" * 50)
    print(f"tensorboard --logdir {os.path.dirname(config.simple_transformer_logdir)}")
    print("=" * 50)
    
    # Compare model performance from checkpoints
    results = get_model_performance_summary(config)
    print("\nModel Performance Summary (from checkpoints):")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if "status" in metrics:
            print(f"  {metrics['status']}")
        else:
            print(f"  Best epoch: {metrics['epoch'] + 1}")
            print(f"  Train loss: {metrics['train_loss']:.6f}")
            print(f"  Val loss: {metrics['val_loss']:.6f}")
    
    # Print latest metrics from TensorBoard
    print_latest_tensorboard_metrics(config)

if __name__ == "__main__":
    main() 