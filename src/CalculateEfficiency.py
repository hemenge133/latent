#!/usr/bin/env python
import os
import sys
import torch
import traceback
from src.Models import StableSimpleTransformer, StableLatentTransformer
from src.Dataset import MultiplicationDataset
import torch.utils.data as data


def calculate_efficiency(simple_results, latent_results):
    print("\nEfficiency Metrics:")

    # Parameter efficiency (lower is better: loss * num_params)
    param_efficiency_simple = simple_results["loss"] * simple_results["params"]
    param_efficiency_latent = latent_results["loss"] * latent_results["params"]

    # Accuracy efficiency (higher is better: accuracy / num_params)
    acc_param_efficiency_simple = (
        simple_results["sequence_accuracy"] / simple_results["params"]
        if simple_results["params"] > 0 and simple_results["sequence_accuracy"] > 0
        else 0
    )
    acc_param_efficiency_latent = (
        latent_results["sequence_accuracy"] / latent_results["params"]
        if latent_results["params"] > 0 and latent_results["sequence_accuracy"] > 0
        else 0
    )

    if param_efficiency_latent < param_efficiency_simple:
        ratio = param_efficiency_simple / param_efficiency_latent
        print(
            f"LatentTransformer is {ratio:.2f}x more parameter-efficient (loss*params)"
        )
    else:
        ratio = param_efficiency_latent / param_efficiency_simple
        print(
            f"SimpleTransformer is {ratio:.2f}x more parameter-efficient (loss*params)"
        )

    # Only show accuracy efficiency if both models have non-zero accuracy
    if acc_param_efficiency_latent > 0 and acc_param_efficiency_simple > 0:
        if acc_param_efficiency_latent > acc_param_efficiency_simple:
            ratio = acc_param_efficiency_latent / acc_param_efficiency_simple
            print(
                f"LatentTransformer is {ratio:.2f}x more accuracy-per-parameter efficient"
            )
        else:
            ratio = acc_param_efficiency_simple / acc_param_efficiency_latent
            print(
                f"SimpleTransformer is {ratio:.2f}x more accuracy-per-parameter efficient"
            )
    else:
        print(
            "Cannot calculate accuracy efficiency metrics: at least one model has 0% accuracy"
        )


def clean_compiled_state_dict(state_dict):
    """Remove _orig_mod. prefix from compiled model state dicts"""
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict

    print("Detected compiled model state dict, cleaning up...")
    return {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
        if k.startswith("_orig_mod.")
    }


def get_best_checkpoint(priority_paths):
    """Find first existing checkpoint from priority list"""
    for path in priority_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths to check, preferring directory checkpoints over the latest ones
    priority_simple_paths = [
        "checkpoints/simpletransformer/simpletransformer_best.pt",
        "checkpoints/simpletransformer_latest.pt",
    ]

    priority_latent_paths = [
        "checkpoints/latenttransformer/latenttransformer_best.pt",
        "checkpoints/latenttransformer_latest.pt",
    ]

    # Find first existing checkpoint from priority list
    simple_checkpoint_path = get_best_checkpoint(priority_simple_paths)
    latent_checkpoint_path = get_best_checkpoint(priority_latent_paths)

    # Check if we found both checkpoints
    if not simple_checkpoint_path:
        print("Error: Could not find SimpleTransformer checkpoint.")
        print("Checked paths:", priority_simple_paths)
        return

    if not latent_checkpoint_path:
        print("Error: Could not find LatentTransformer checkpoint.")
        print("Checked paths:", priority_latent_paths)
        return

    print(f"Using checkpoints:")
    print(f"- SimpleTransformer: {simple_checkpoint_path}")
    print(f"- LatentTransformer: {latent_checkpoint_path}")

    # Detect latent token size from checkpoint
    num_latent = 32  # Default
    vocab_size = 11  # Default based on Dataset.py

    try:
        print(f"Loading from: {latent_checkpoint_path}")
        latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

        # Get model state dict
        latent_state = latent_checkpoint.get("model_state_dict", latent_checkpoint)

        # Clean up compiled model state dict if needed
        latent_state = clean_compiled_state_dict(latent_state)

        if "latent_tokens" in latent_state:
            latent_shape = latent_state["latent_tokens"].shape
            num_latent = latent_shape[0]
            print(f"Detected {num_latent} latent tokens in checkpoint")

        # Check vocab size
        if "embed.weight" in latent_state:
            embed_shape = latent_state["embed.weight"].shape
            vocab_size = embed_shape[0]
            print(f"Detected vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Warning: Error detecting model parameters: {e}")
        print(f"Using defaults: {num_latent} latent tokens, {vocab_size} vocab size")

    # Load models with architecture matching the saved checkpoints
    print(
        f"\nInitializing models with: d_model=512, num_layers=6, vocab_size={vocab_size}"
    )
    simple_model = StableSimpleTransformer(
        d_model=512, num_layers=6, vocab_size=vocab_size
    ).to(device)

    latent_model = StableLatentTransformer(
        d_model=512,
        num_layers=6,
        num_latent=num_latent,
        vocab_size=vocab_size,
        bottleneck_factor=1.0,  # Full bottleneck matching training
    ).to(device)

    # Load the checkpoints
    try:
        print(f"\nLoading SimpleTransformer from: {simple_checkpoint_path}")
        simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)

        print(f"Loading LatentTransformer from: {latent_checkpoint_path}")
        if "latent_checkpoint" not in locals():
            latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

        # Get model state dicts
        simple_state_dict = simple_checkpoint.get("model_state_dict", simple_checkpoint)
        latent_state_dict = latent_checkpoint.get("model_state_dict", latent_checkpoint)

        # Clean up compiled model state dicts if needed
        simple_state_dict = clean_compiled_state_dict(simple_state_dict)
        latent_state_dict = clean_compiled_state_dict(latent_state_dict)

        # Load state dicts
        simple_model.load_state_dict(simple_state_dict)
        latent_model.load_state_dict(latent_state_dict)

        # Get validation metrics
        if "val_loss" in simple_checkpoint:
            simple_val_loss = simple_checkpoint["val_loss"]
            simple_val_accuracy = simple_checkpoint.get("val_sequence_accuracy", 0.0)
        else:
            # If not available, use placeholder values
            print(
                "Warning: No validation metrics in SimpleTransformer checkpoint. Using placeholders."
            )
            simple_val_loss = 1.0
            simple_val_accuracy = 0.0

        if "val_loss" in latent_checkpoint:
            latent_val_loss = latent_checkpoint["val_loss"]
            latent_val_accuracy = latent_checkpoint.get("val_sequence_accuracy", 0.0)
        else:
            # If not available, use placeholder values
            print(
                "Warning: No validation metrics in LatentTransformer checkpoint. Using placeholders."
            )
            latent_val_loss = 1.0
            latent_val_accuracy = 0.0

        # Count parameters
        simple_params = sum(p.numel() for p in simple_model.parameters())
        latent_params = sum(p.numel() for p in latent_model.parameters())

        # Prepare results
        simple_results = {
            "loss": simple_val_loss,
            "sequence_accuracy": simple_val_accuracy,
            "params": simple_params,
        }

        latent_results = {
            "loss": latent_val_loss,
            "sequence_accuracy": latent_val_accuracy,
            "params": latent_params,
        }

        print("\nModel Parameters:")
        print(f"SimpleTransformer: {simple_params:,}")
        print(f"LatentTransformer: {latent_params:,}")
        print(f"Parameter ratio: {latent_params/simple_params:.2f}x")

        print("\nValidation Results:")
        print(
            f"SimpleTransformer - Loss: {simple_val_loss:.6f}, Accuracy: {simple_val_accuracy:.2%}"
        )
        print(
            f"LatentTransformer - Loss: {latent_val_loss:.6f}, Accuracy: {latent_val_accuracy:.2%}"
        )

        calculate_efficiency(simple_results, latent_results)

    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        traceback.print_exc()
        print(
            "\nMake sure you have trained models and saved checkpoints with the correct architecture."
        )


if __name__ == "__main__":
    main()
