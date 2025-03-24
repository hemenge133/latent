import torch
from stable_comparison_with_accuracy import StableSimpleTransformer, StableLatentTransformer
from Dataset import MultiplicationDataset
import torch.utils.data as data
import os
import sys

def calculate_efficiency(simple_results, latent_results):
    print("\nEfficiency Metrics:")
    
    # Parameter efficiency (lower is better: loss * num_params)
    param_efficiency_simple = simple_results['loss'] * simple_results['params']
    param_efficiency_latent = latent_results['loss'] * latent_results['params']
    
    # Accuracy efficiency (higher is better: accuracy / num_params)
    acc_param_efficiency_simple = simple_results['sequence_accuracy'] / simple_results['params'] if simple_results['params'] > 0 and simple_results['sequence_accuracy'] > 0 else 0
    acc_param_efficiency_latent = latent_results['sequence_accuracy'] / latent_results['params'] if latent_results['params'] > 0 and latent_results['sequence_accuracy'] > 0 else 0
    
    if param_efficiency_latent < param_efficiency_simple:
        ratio = param_efficiency_simple / param_efficiency_latent
        print(f"LatentTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
    else:
        ratio = param_efficiency_latent / param_efficiency_simple
        print(f"SimpleTransformer is {ratio:.2f}x more parameter-efficient (loss*params)")
    
    # Only show accuracy efficiency if both models have non-zero accuracy
    if acc_param_efficiency_latent > 0 and acc_param_efficiency_simple > 0:
        if acc_param_efficiency_latent > acc_param_efficiency_simple:
            ratio = acc_param_efficiency_latent / acc_param_efficiency_simple
            print(f"LatentTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
        else:
            ratio = acc_param_efficiency_simple / acc_param_efficiency_latent
            print(f"SimpleTransformer is {ratio:.2f}x more accuracy-per-parameter efficient")
    else:
        print("Cannot calculate accuracy efficiency metrics: at least one model has 0% accuracy")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detect the latent token size from the saved checkpoint
    try:
        # First check if the checkpoint exists
        latent_paths = [
            'checkpoints/latenttransformer/latenttransformer_best.pt',
            'checkpoints/latent_transformer/latent_transformer_best.pt'
        ]
        
        # Find existing latent transformer checkpoint
        latent_checkpoint_path = None
        for path in latent_paths:
            if os.path.exists(path):
                latent_checkpoint_path = path
                break
        
        if latent_checkpoint_path:
            # Load checkpoint but don't apply to model yet
            print(f"Detecting latent token size from: {latent_checkpoint_path}")
            checkpoint = torch.load(latent_checkpoint_path)
            # Extract latent_tokens shape from the checkpoint
            if 'model_state_dict' in checkpoint and 'latent_tokens' in checkpoint['model_state_dict']:
                latent_shape = checkpoint['model_state_dict']['latent_tokens'].shape
                num_latent = latent_shape[0]
                print(f"Detected {num_latent} latent tokens in checkpoint")
            else:
                print("Could not find latent_tokens in checkpoint, defaulting to 32")
                num_latent = 32
        else:
            print("No checkpoint found, defaulting to 32 latent tokens")
            num_latent = 32
    except Exception as e:
        print(f"Error detecting latent token size: {e}")
        print("Defaulting to 32 latent tokens")
        num_latent = 32
    
    # Load models with architecture matching the saved checkpoints
    simple_model = StableSimpleTransformer(
        d_model=512,
        num_layers=6,  # Changed from 4 to 6 to match saved model
        vocab_size=11  # Changed from 12 to 11 to match saved model
    ).to(device)
    
    latent_model = StableLatentTransformer(
        d_model=512,
        num_layers=6,  # Changed from 4 to 6 to match saved model
        num_latent=num_latent,  # Use detected or default size
        vocab_size=11,  # Changed from 12 to 11 to match saved model
        bottleneck_factor=0.5  # Added to match training configuration
    ).to(device)
    
    # Try to load the latest checkpoints
    try:
        # Check both possible checkpoint locations
        simple_paths = [
            'checkpoints/simpletransformer/simpletransformer_best.pt',
            'checkpoints/simple_transformer/simple_transformer_best.pt'
        ]
        latent_paths = [
            'checkpoints/latenttransformer/latenttransformer_best.pt',
            'checkpoints/latent_transformer/latent_transformer_best.pt'
        ]
        
        # Find existing simple transformer checkpoint
        simple_checkpoint_path = None
        for path in simple_paths:
            if os.path.exists(path):
                simple_checkpoint_path = path
                break
        
        # Find existing latent transformer checkpoint
        latent_checkpoint_path = None
        for path in latent_paths:
            if os.path.exists(path):
                latent_checkpoint_path = path
                break
        
        if not simple_checkpoint_path or not latent_checkpoint_path:
            raise FileNotFoundError("Could not find checkpoint files")
            
        print(f"Loading simple transformer checkpoint from: {simple_checkpoint_path}")
        print(f"Loading latent transformer checkpoint from: {latent_checkpoint_path}")
        
        simple_checkpoint = torch.load(simple_checkpoint_path)
        latent_checkpoint = torch.load(latent_checkpoint_path)
        
        simple_model.load_state_dict(simple_checkpoint['model_state_dict'])
        latent_model.load_state_dict(latent_checkpoint['model_state_dict'])
        
        simple_results = {
            'loss': simple_checkpoint['val_loss'],
            'sequence_accuracy': simple_checkpoint['val_sequence_accuracy'],
            'params': sum(p.numel() for p in simple_model.parameters())
        }
        
        latent_results = {
            'loss': latent_checkpoint['val_loss'],
            'sequence_accuracy': latent_checkpoint['val_sequence_accuracy'],
            'params': sum(p.numel() for p in latent_model.parameters())
        }
        
        print("\nModel Parameters:")
        print(f"SimpleTransformer: {simple_results['params']:,}")
        print(f"LatentTransformer: {latent_results['params']:,}")
        print(f"Parameter ratio: {latent_results['params']/simple_results['params']:.2f}x")
        
        print("\nValidation Results:")
        print(f"SimpleTransformer - Loss: {simple_results['loss']:.6f}, Accuracy: {simple_results['sequence_accuracy']:.2%}")
        print(f"LatentTransformer - Loss: {latent_results['loss']:.6f}, Accuracy: {latent_results['sequence_accuracy']:.2%}")
        
        calculate_efficiency(simple_results, latent_results)
        
    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        print("Make sure you have trained models and saved checkpoints.")

if __name__ == '__main__':
    main() 