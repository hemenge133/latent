#!/usr/bin/env python3
"""
Script to check the dimensions of model checkpoints
"""
import torch
import argparse
import os
import sys


def check_checkpoint(checkpoint_path):
    """Check the dimensions of a checkpoint"""
    print(f"Checking checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"  Keys: {list(checkpoint.keys())}")

        # Try to find d_model
        d_model = None
        if "d_model" in checkpoint:
            d_model = checkpoint["d_model"]
        elif "config" in checkpoint and "d_model" in checkpoint["config"]:
            d_model = checkpoint["config"]["d_model"]
        elif "embed.weight" in checkpoint:
            d_model = checkpoint["embed.weight"].size(1)
        elif "model_state_dict" in checkpoint:
            # Search in model_state_dict
            for key, value in checkpoint["model_state_dict"].items():
                if key.endswith(".weight") and len(value.shape) >= 2:
                    print(f"  Found weight: {key} with shape {value.shape}")
                    if "embed" in key:
                        d_model = value.size(1)
                        break

        if d_model:
            print(f"  d_model: {d_model}")
        else:
            print("  d_model: Not found")

        # Check steps
        if "step" in checkpoint:
            print(f"  step: {checkpoint['step']}")
        else:
            print("  step: Not found")

        return d_model
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None


def main():
    # Check all checkpoints
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return

    # Check SimpleTransformer checkpoints
    simple_dir = os.path.join(checkpoints_dir, "simpletransformer")
    if os.path.exists(simple_dir):
        print("\nSimpleTransformer checkpoints:")
        for filename in os.listdir(simple_dir):
            if filename.endswith(".pt"):
                check_checkpoint(os.path.join(simple_dir, filename))

    # Check LatentTransformer checkpoints
    latent_dir = os.path.join(checkpoints_dir, "latenttransformer")
    if os.path.exists(latent_dir):
        print("\nLatentTransformer checkpoints:")
        for filename in os.listdir(latent_dir):
            if filename.endswith(".pt"):
                check_checkpoint(os.path.join(latent_dir, filename))


if __name__ == "__main__":
    main()
