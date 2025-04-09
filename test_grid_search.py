"""
Test script to verify the grid search logic correctly identifies the best parameters.
This script mocks the train_models_parallel function to return predetermined validation losses.
"""

import os
import sys
import json
import itertools
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO")  # Add stdout handler

# Mock dictionary to store expected results
mock_validation_results = {}

def mock_train_models_parallel(models, dataset, dataset_val, vocab_size, device,
                              max_steps, batch_size, config, args, models_params, 
                              start_step=0, simple_checkpoint=None, latent_checkpoint=None,
                              log_dir="runs/parallel_comparison", simple_lr=None, 
                              simple_wd=None, latent_lr=None, latent_wd=None):
    """Mock implementation of train_models_parallel that returns predefined validation losses."""
    
    # Extract the current parameter combination 
    params = {
        "simple_lr": simple_lr,
        "simple_wd": simple_wd,
        "latent_lr": latent_lr,
        "latent_wd": latent_wd,
    }
    
    # If models contains the architecture parameters, extract them
    if isinstance(models, dict) and "simple" in models and "latent" in models:
        # These would normally come from the model instances, but here we'll use dummy values
        # In real grid search, these would be actual model objects with these attributes
        # Here we're just pretending we got these values from the models
        params["d_model"] = 128
        params["nhead"] = 4
        params["num_layers"] = 2
        params["dropout"] = 0.1
        params["num_latent"] = 2
    
    # Create a unique key for this parameter combination
    param_key = "_".join([f"{k}{v}" for k, v in params.items()])
    
    # Get the predefined result for this parameter combination
    mock_result = mock_validation_results.get(param_key, None)
    
    if mock_result is None:
        # If we don't have a predefined result, generate random-ish values based on parameters
        # We use a simple formula so we can predict which should be best
        simple_loss = 1.0 + 0.1 * simple_lr * 1000 + 0.2 * simple_wd * 10
        simple_acc = max(0.0, 1.0 - (simple_loss - 1.0) / 2.0)
        
        latent_loss = 1.0 + 0.1 * latent_lr * 1000 + 0.2 * latent_wd * 10
        latent_acc = max(0.0, 1.0 - (latent_loss - 1.0) / 2.0)
        
        logger.info(f"No predefined result for {param_key}, using generated values: ")
        logger.info(f"Simple loss: {simple_loss:.4f}, acc: {simple_acc:.4f}")
        logger.info(f"Latent loss: {latent_loss:.4f}, acc: {latent_acc:.4f}")
    else:
        # Use the predefined result
        simple_loss = mock_result["simple_loss"]
        simple_acc = mock_result["simple_acc"]
        latent_loss = mock_result["latent_loss"]
        latent_acc = mock_result["latent_acc"]
        
        logger.info(f"Using predefined result for {param_key}: ")
        logger.info(f"Simple loss: {simple_loss:.4f}, acc: {simple_acc:.4f}")
        logger.info(f"Latent loss: {latent_loss:.4f}, acc: {latent_acc:.4f}")
    
    # Return a dictionary that matches the expected structure from train_models_parallel
    return {
        "simple": {
            "loss": simple_loss,
            "sequence_accuracy": simple_acc,
            "digit_accuracy": 0.9,  # Dummy value
            "params": 10000,        # Dummy value
            "steps": 100            # Dummy value  
        },
        "latent": {
            "loss": latent_loss,
            "sequence_accuracy": latent_acc,
            "digit_accuracy": 0.9,  # Dummy value
            "params": 12000,        # Dummy value
            "steps": 100            # Dummy value
        },
        "training_time": 10.5       # Dummy value
    }

def run_mock_grid_search():
    """Run a mock grid search to test our result tracking logic."""
    
    # Define a small grid for testing
    param_grid = {
        "simple_lr": [2e-4, 3e-4],
        "simple_wd": [0.02, 0.03],
        "latent_lr": [1e-4, 2e-4],
        "latent_wd": [0.01, 0.02],
    }
    
    # Define expected best parameters
    # We'll make one specific combination the clear winner for each model
    best_simple_params = {
        "simple_lr": 2e-4,
        "simple_wd": 0.02,
        "latent_lr": 1e-4,  # These don't matter for simple model
        "latent_wd": 0.01
    }
    
    best_latent_params = {
        "simple_lr": 3e-4,  # These don't matter for latent model
        "simple_wd": 0.03,
        "latent_lr": 1e-4,
        "latent_wd": 0.01
    }
    
    # Set up predetermined results with the best combination having the lowest loss
    best_simple_key = "_".join([f"{k}{v}" for k, v in best_simple_params.items()])
    best_latent_key = "_".join([f"{k}{v}" for k, v in best_latent_params.items()])
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Set up mock results for each combination
    for params in all_combinations:
        param_key = "_".join([f"{k}{v}" for k, v in params.items()])
        
        # Default values - mediocre performance
        simple_loss = 1.5
        simple_acc = 0.7
        latent_loss = 1.3
        latent_acc = 0.8
        
        # If this matches our best simple params, give it the best performance
        if param_key == best_simple_key:
            simple_loss = 0.8  # Best loss for simple
            simple_acc = 0.95
        
        # If this matches our best latent params, give it the best performance
        if param_key == best_latent_key:
            latent_loss = 0.7  # Best loss for latent
            latent_acc = 0.98
            
        # Store the mock results
        mock_validation_results[param_key] = {
            "simple_loss": simple_loss,
            "simple_acc": simple_acc,
            "latent_loss": latent_loss,
            "latent_acc": latent_acc
        }
    
    # Now run the grid search
    logger.info("Starting mock grid search...")
    
    # Initialize trackers for best params (just like in main.py)
    best_simple_metric = float("inf") 
    best_simple_params_found = None
    best_latent_metric = float("inf")
    best_latent_params_found = None
    all_results = []
    
    # Loop through each parameter combination (simplified version of main.py grid search)
    for i, params in enumerate(all_combinations):
        logger.info(f"--- Mock Grid Search Run {i+1}/{len(all_combinations)} ---")
        logger.info(f"Parameters: {params}")
        
        # Call our mock training function
        results = mock_train_models_parallel(
            models={},  # Empty, not used in our mock
            dataset=None,
            dataset_val=None,
            vocab_size=10,
            device=None,
            max_steps=10,
            batch_size=2,
            config=None,
            args=None,
            models_params=None,
            simple_lr=params["simple_lr"],
            simple_wd=params["simple_wd"],
            latent_lr=params["latent_lr"],
            latent_wd=params["latent_wd"]
        )
        
        # Extract metrics - this is the exact code from main.py that we're testing
        simple_final_val_loss = results.get("simple", {}).get("loss", float('inf'))
        simple_final_val_acc = results.get("simple", {}).get("sequence_accuracy", 0.0)
        latent_final_val_loss = results.get("latent", {}).get("loss", float('inf'))
        latent_final_val_acc = results.get("latent", {}).get("sequence_accuracy", 0.0)
        
        # Store result
        current_run_result = {
            "params": params,
            "simple_final_val_loss": simple_final_val_loss,
            "simple_final_val_accuracy": simple_final_val_acc,
            "latent_final_val_loss": latent_final_val_loss,
            "latent_final_val_accuracy": latent_final_val_acc,
        }
        all_results.append(current_run_result)
        
        # Update best parameters independently - copied from main.py
        if simple_final_val_loss < best_simple_metric:
            best_simple_metric = simple_final_val_loss
            best_simple_params_found = params
            logger.info(f"*** New best SimpleTransformer validation loss found: {best_simple_metric:.6f} with params: {best_simple_params_found} ***")

        if latent_final_val_loss < best_latent_metric:
            best_latent_metric = latent_final_val_loss
            best_latent_params_found = params
            logger.info(f"*** New best LatentTransformer validation loss found: {best_latent_metric:.6f} with params: {best_latent_params_found} ***")
    
    # Final results
    logger.info("--- Mock Grid Search Complete ---")
    logger.info(f"Best SimpleTransformer validation loss found: {best_simple_metric:.6f}")
    logger.info(f"Best SimpleTransformer hyperparameters: {best_simple_params_found}")
    logger.info(f"Best LatentTransformer validation loss found: {best_latent_metric:.6f}")
    logger.info(f"Best LatentTransformer hyperparameters: {best_latent_params_found}")
    
    # Save results to file
    results_file = "mock_grid_search_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Mock grid search results saved to {results_file}")
    
    # Verify the results - this is what we're really testing
    simple_correct = (best_simple_params_found == best_simple_params)
    latent_correct = (best_latent_params_found == best_latent_params)
    
    logger.info("--- Verification Results ---")
    logger.info(f"SimpleTransformer best params correctly identified: {simple_correct}")
    if not simple_correct:
        logger.error(f"Expected: {best_simple_params}")
        logger.error(f"Found: {best_simple_params_found}")
    
    logger.info(f"LatentTransformer best params correctly identified: {latent_correct}")
    if not latent_correct:
        logger.error(f"Expected: {best_latent_params}")
        logger.error(f"Found: {best_latent_params_found}")
    
    return simple_correct and latent_correct

if __name__ == "__main__":
    success = run_mock_grid_search()
    sys.exit(0 if success else 1) 