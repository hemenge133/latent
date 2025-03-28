#!/usr/bin/env python
"""
Helper script to resume training from the latest checkpoints.
This script sets the --resume flag and passes all other arguments to main.py.
"""
import sys
import os
import subprocess
import argparse
import torch


def main():
    # Create parser for any additional args
    parser = argparse.ArgumentParser(
        description="Resume training from latest checkpoints"
    )
    parser.add_argument(
        "--d-model", type=int, default=768, help="Model dimension (default: 768)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of transformer layers (default: 8)",
    )
    parser.add_argument(
        "--num-latent",
        type=int,
        default=16,
        help="Number of latent tokens (default: 16)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum training steps (default: 10000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=3,
        help="Minimum number of digits (default: 3 for triple-digit)",
    )
    parser.add_argument(
        "--use-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )

    # Parse args
    args = parser.parse_args()

    # Specify the exact checkpoint file to use
    simple_checkpoint_path = "checkpoints/simpletransformer/simpletransformer_best.pt"

    # Check if it exists
    if not os.path.exists(simple_checkpoint_path):
        print(f"Error: Specified checkpoint not found: {simple_checkpoint_path}")
        sys.exit(1)

    # Load checkpoint to check step
    checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", "unknown")
    print(f"Resuming from {simple_checkpoint_path}")
    print(f"Step: {step}, Val loss: {val_loss}")

    # Copy the checkpoint to make sure it's used
    latent_checkpoint_path = "checkpoints/latenttransformer/latenttransformer_best.pt"

    # Create directories if needed
    os.makedirs(os.path.dirname(simple_checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(latent_checkpoint_path), exist_ok=True)

    # Copy checkpoint to latest.pt so it will be used by the resume code
    import shutil

    shutil.copy(
        simple_checkpoint_path,
        "checkpoints/simpletransformer/simpletransformer_latest.pt",
    )
    if os.path.exists(latent_checkpoint_path):
        shutil.copy(
            latent_checkpoint_path,
            "checkpoints/latenttransformer/latenttransformer_latest.pt",
        )

    # Build command arguments
    cmd_args = ["python", "main.py", "--resume"]

    # Add other arguments
    if args.d_model != 768:
        cmd_args.extend(["--d-model", str(args.d_model)])
    if args.num_layers != 8:
        cmd_args.extend(["--num-layers", str(args.num_layers)])
    if args.num_latent != 16:
        cmd_args.extend(["--num-latent", str(args.num_latent)])
    if args.max_steps != 10000:
        cmd_args.extend(["--max-steps", str(args.max_steps)])
    if args.batch_size != 512:
        cmd_args.extend(["--batch-size", str(args.batch_size)])
    if args.min_digits != 3:
        cmd_args.extend(["--min-digits", str(args.min_digits)])
    if args.use_checkpointing:
        cmd_args.append("--use-checkpointing")

    print("Resuming training with command:")
    print(" ".join(cmd_args))
    print("\nThis will continue training from the specified checkpoint.")

    # Execute the command
    subprocess.run(cmd_args)


if __name__ == "__main__":
    main()
