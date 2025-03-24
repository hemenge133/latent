"""
Script to generate model comparison summary with fair parameter settings
"""
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import EventAccumulator
import argparse
import json
from pathlib import Path

from Dataset import MultiplicationDataset
from Collate import collate_fn
from Transformer import SimpleTransformer, LatentTransformer
from utils import evaluate
from config import TrainingConfig

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_best_hyperparams():
    """Get the best hyperparameters from grid search results"""
    results_file = Path("grid_search_results/all_results.json")
    
    if not results_file.exists():
        # Default parameters if grid search hasn't been run
        return {
            'simple': {
                'd_model': 320,
                'num_layers': 5,
                'learning_rate': 3e-4,
                'batch_size': 128
            },
            'latent': {
                'd_model': 320,
                'num_layers': 4,
                'num_latent': 10,
                'bottleneck_factor': 1.0,
                'learning_rate': 3e-4,
                'batch_size': 128
            }
        }
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Find best parameters for each model type
    simple_results = [r for r in all_results if r['model_type'] == 'simple']
    latent_results = [r for r in all_results if r['model_type'] == 'latent']
    
    # Sort by validation loss
    simple_results.sort(key=lambda x: x['best_val_loss'])
    latent_results.sort(key=lambda x: x['best_val_loss'])
    
    # Get best for each type
    best_simple = simple_results[0] if simple_results else None
    best_latent = latent_results[0] if latent_results else None
    
    if not best_simple or not best_latent:
        raise ValueError("Couldn't find best parameters for both model types")
    
    return {
        'simple': {
            'd_model': best_simple['d_model'],
            'num_layers': best_simple['num_layers'],
            'learning_rate': best_simple['learning_rate'],
            'batch_size': best_simple['batch_size']
        },
        'latent': {
            'd_model': best_latent['d_model'],
            'num_layers': best_latent['num_layers'],
            'num_latent': best_latent['num_latent'],
            'bottleneck_factor': 1.0,  # Default to full bottleneck
            'learning_rate': best_latent['learning_rate'],
            'batch_size': best_latent['batch_size']
        }
    }

def evaluate_models(args):
    """Evaluate both models using the best hyperparameters"""
    config = TrainingConfig()
    
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Get best hyperparameters
    best_params = get_best_hyperparams()
    
    # Create test dataset
    test_dataset = MultiplicationDataset(
        config.total_samples, 
        split='test', 
        split_ratio=config.split_ratio
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Set up evaluation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    results = {}
    
    # Evaluate SimpleTransformer
    if args.simple_checkpoint:
        print("\nEvaluating SimpleTransformer...")
        simple_path = args.simple_checkpoint
        
        # Build model with the best hyperparameters
        simple_model = SimpleTransformer(
            vocab_size=test_dataset.vocab_size,
            d_model=best_params['simple']['d_model'],
            nhead=config.nhead,
            num_layers=best_params['simple']['num_layers']
        ).to(device)
        
        try:
            checkpoint = torch.load(simple_path, map_location=device)
            simple_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded SimpleTransformer checkpoint from {simple_path}")
            
            # Evaluate model
            simple_test_loss = evaluate(
                model=simple_model,
                data_loader=test_loader,
                criterion=criterion,
                device=device,
                vocab_size=test_dataset.vocab_size,
                desc="SimpleTransformer Test"
            )
            
            num_params = count_parameters(simple_model)
            results['SimpleTransformer'] = {
                'test_loss': simple_test_loss,
                'parameters': num_params,
                'hyperparameters': best_params['simple']
            }
            print(f"SimpleTransformer - Test Loss: {simple_test_loss:.6f}, Parameters: {num_params:,}")
            
        except Exception as e:
            print(f"Error evaluating SimpleTransformer: {e}")
    
    # Evaluate LatentTransformer
    if args.latent_checkpoint:
        print("\nEvaluating LatentTransformer...")
        latent_path = args.latent_checkpoint
        
        # Build model with the best hyperparameters
        latent_model = LatentTransformer(
            vocab_size=test_dataset.vocab_size,
            d_model=best_params['latent']['d_model'],
            num_latent=best_params['latent']['num_latent'],
            nhead=config.nhead,
            num_layers=best_params['latent']['num_layers'],
            bottleneck_factor=best_params['latent']['bottleneck_factor']
        ).to(device)
        
        try:
            checkpoint = torch.load(latent_path, map_location=device)
            latent_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded LatentTransformer checkpoint from {latent_path}")
            
            # Evaluate model
            latent_test_loss = evaluate(
                model=latent_model,
                data_loader=test_loader,
                criterion=criterion,
                device=device,
                vocab_size=test_dataset.vocab_size,
                desc="LatentTransformer Test"
            )
            
            num_params = count_parameters(latent_model)
            results['LatentTransformer'] = {
                'test_loss': latent_test_loss,
                'parameters': num_params,
                'hyperparameters': best_params['latent']
            }
            print(f"LatentTransformer - Test Loss: {latent_test_loss:.6f}, Parameters: {num_params:,}")
            
        except Exception as e:
            print(f"Error evaluating LatentTransformer: {e}")
    
    # Compare the models
    if 'SimpleTransformer' in results and 'LatentTransformer' in results:
        print("\nModel Comparison:")
        print("=" * 50)
        print(f"SimpleTransformer: {results['SimpleTransformer']['test_loss']:.6f} loss ({results['SimpleTransformer']['parameters']:,} parameters)")
        print(f"LatentTransformer: {results['LatentTransformer']['test_loss']:.6f} loss ({results['LatentTransformer']['parameters']:,} parameters)")
        
        # Calculate parameter efficiency
        simple_efficiency = results['SimpleTransformer']['test_loss'] * results['SimpleTransformer']['parameters']
        latent_efficiency = results['LatentTransformer']['test_loss'] * results['LatentTransformer']['parameters']
        
        # Lower is better for efficiency (loss * params)
        if latent_efficiency < simple_efficiency:
            efficiency_ratio = simple_efficiency / latent_efficiency
            print(f"LatentTransformer is {efficiency_ratio:.2f}x more parameter-efficient")
        else:
            efficiency_ratio = latent_efficiency / simple_efficiency
            print(f"SimpleTransformer is {efficiency_ratio:.2f}x more parameter-efficient")
    
    # Save results
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare transformer models")
    parser.add_argument("--simple-checkpoint", type=str, 
                        default="checkpoints/simple_transformer_3digit_fast/simple_transformer_3digit_fast_best.pt",
                        help="Path to the SimpleTransformer checkpoint")
    parser.add_argument("--latent-checkpoint", type=str, 
                        default="checkpoints/latent_transformer_3digit_fast/latent_transformer_3digit_fast_best.pt",
                        help="Path to the LatentTransformer checkpoint")
    
    args = parser.parse_args()
    evaluate_models(args)

if __name__ == "__main__":
    main() 