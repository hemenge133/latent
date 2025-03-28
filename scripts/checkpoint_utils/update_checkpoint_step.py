#!/usr/bin/env python
"""
Utility to update checkpoint step numbers.
This is useful when you want to resume training from a specific step.
"""
import torch
import os
import shutil
import glob
import time
import sys
import argparse


def find_latest_run_dir():
    """Find the most recent run directory"""
    run_dirs = glob.glob("runs/parallel_comparison/*_d768_l8_n16")
    if not run_dirs:
        print("Error: Could not find any matching run directories")
        return None

    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    latest_dir = run_dirs[0]
    print(f"Latest run directory: {latest_dir}")
    return latest_dir


def update_checkpoints(target_step):
    """Update both model checkpoints to the specified step"""
    # Update SimpleTransformer checkpoint
    simple_path = "checkpoints/simpletransformer/simpletransformer_best.pt"
    if os.path.exists(simple_path):
        print(f"Loading: {simple_path}")
        checkpoint = torch.load(simple_path, map_location="cpu")
        # Check current step
        old_step = checkpoint.get("step", "Not found")
        print(f"  Current step: {old_step}")
        checkpoint["step"] = target_step
        torch.save(
            checkpoint, "checkpoints/simpletransformer/simpletransformer_latest.pt"
        )
        print(
            f"  Updated and saved to simpletransformer_latest.pt with step {target_step}"
        )
    else:
        print(f"Error: {simple_path} not found")

    # Update LatentTransformer checkpoint
    latent_path = "checkpoints/latenttransformer/latenttransformer_best.pt"
    if os.path.exists(latent_path):
        print(f"Loading: {latent_path}")
        checkpoint = torch.load(latent_path, map_location="cpu")
        # Check current step
        old_step = checkpoint.get("step", "Not found")
        print(f"  Current step: {old_step}")
        checkpoint["step"] = target_step
        torch.save(
            checkpoint, "checkpoints/latenttransformer/latenttransformer_latest.pt"
        )
        print(
            f"  Updated and saved to latenttransformer_latest.pt with step {target_step}"
        )
    else:
        print(f"Error: {latent_path} not found")


def create_new_runs_dir(target_step):
    """Create a new runs directory with timestamp and step info"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_dir = (
        f"runs/parallel_comparison/{timestamp}_d768_l8_n16_resumed_from_{target_step}"
    )
    os.makedirs(new_dir, exist_ok=True)

    # Create subdirectories for both models
    os.makedirs(f"{new_dir}/simple", exist_ok=True)
    os.makedirs(f"{new_dir}/latent", exist_ok=True)

    print(f"Created new run directory: {new_dir}")
    return new_dir


def update_resume_script(target_step):
    """Update the resume script to use the correct step"""
    script_path = "scripts/resume_training.sh"

    # Create script if it doesn't exist
    if not os.path.exists(script_path):
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated resume script\n\n")
            f.write("cd /home/h/Devel/latent\n\n")
            f.write("# Activate virtual environment\n")
            f.write("source ~/.virtualenvs/latent/bin/activate\n\n")
            f.write("# Resume training\n")
            f.write(
                f"python main.py --resume --max-steps 20000 --use-checkpointing --d-model 768 --num-layers 8 --num-latent 16 --min-digits 3 --batch-size 512\n"
            )
        os.chmod(script_path, 0o755)
        print(f"Created new resume script: {script_path}")
        return

    # Update existing script
    with open(script_path, "r") as f:
        lines = f.readlines()

    # Update lines with step numbers if found
    updated = False
    for i, line in enumerate(lines):
        if f"step =" in line or f"step=" in line:
            lines[i] = line.replace(
                line.split("=")[1].strip().split()[0], str(target_step)
            )
            updated = True
        if f"step " in line and not "=" in line:
            for word_i, word in enumerate(line.split()):
                if word.isdigit():
                    parts = line.split()
                    parts[word_i] = str(target_step)
                    lines[i] = " ".join(parts) + "\n"
                    updated = True
                    break

    # If no step references were found, we don't need to update
    if not updated:
        print(f"No step references found in {script_path}, no update needed")
        return

    with open(script_path, "w") as f:
        f.writelines(lines)

    print(f"Updated {script_path} to use step {target_step}")


def main():
    parser = argparse.ArgumentParser(description="Update checkpoint step numbers")
    parser.add_argument(
        "step", type=int, help="Target step number to set in checkpoints"
    )
    parser.add_argument(
        "--no-runs-dir", action="store_true", help="Skip creating a new runs directory"
    )
    args = parser.parse_args()

    target_step = args.step

    print(f"Updating checkpoint and run information to start at step {target_step}...")
    update_checkpoints(target_step)

    if not args.no_runs_dir:
        new_dir = create_new_runs_dir(target_step)

    update_resume_script(target_step)

    print("\nAll updates complete!")
    print("\nTo continue training, run:")
    print("scripts/resume_training.sh")


if __name__ == "__main__":
    main()
