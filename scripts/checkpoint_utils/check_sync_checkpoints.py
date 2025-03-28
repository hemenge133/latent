#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to check and synchronize checkpoint steps between models.
This ensures both models will resume from the same step.
"""
import torch
import os
import shutil
import argparse
import sys
import time


def check_checkpoints():
    """Check both model checkpoints and report their status"""
    simple_latest = "checkpoints/simpletransformer/simpletransformer_latest.pt"
    simple_best = "checkpoints/simpletransformer/simpletransformer_best.pt"
    latent_latest = "checkpoints/latenttransformer/latenttransformer_latest.pt"
    latent_best = "checkpoints/latenttransformer/latenttransformer_best.pt"

    checkpoints = {
        "simple_latest": simple_latest,
        "simple_best": simple_best,
        "latent_latest": latent_latest,
        "latent_best": latent_best,
    }

    checkpoint_info = {}

    # Check each checkpoint file
    for name, path in checkpoints.items():
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location="cpu")
                step = checkpoint.get("step", "Not found")
                val_loss = checkpoint.get("val_loss", "Not found")
                has_optimizer = "optimizer_state_dict" in checkpoint

                checkpoint_info[name] = {
                    "exists": True,
                    "step": step,
                    "val_loss": val_loss,
                    "has_optimizer": has_optimizer,
                    "size_mb": os.path.getsize(path) / (1024 * 1024),
                }
                print(
                    f"[OK] {name}: Step {step}, Val Loss {val_loss}, Size {checkpoint_info[name]['size_mb']:.2f}MB, Optimizer: {'Yes' if has_optimizer else 'No'}"
                )
            except Exception as e:
                checkpoint_info[name] = {
                    "exists": True,
                    "error": str(e),
                    "size_mb": os.path.getsize(path) / (1024 * 1024),
                }
                print(f"[ERROR] {name}: Error loading checkpoint: {e}")
        else:
            checkpoint_info[name] = {"exists": False}
            print(f"[ERROR] {name}: File not found")

    # Return collected information
    return checkpoint_info


def sync_checkpoints(target_step=None, target_type=None, backup=True):
    """
    Synchronize checkpoint steps between models.

    Args:
        target_step: The target step to set. If None, use the lowest step.
        target_type: Which checkpoint type to sync ('latest', 'best', or None for both)
        backup: Whether to backup existing checkpoints before modifying
    """
    checkpoint_info = check_checkpoints()

    # Determine which checkpoints exist
    simple_latest_exists = checkpoint_info.get("simple_latest", {}).get("exists", False)
    simple_best_exists = checkpoint_info.get("simple_best", {}).get("exists", False)
    latent_latest_exists = checkpoint_info.get("latent_latest", {}).get("exists", False)
    latent_best_exists = checkpoint_info.get("latent_best", {}).get("exists", False)

    # Get steps for existing checkpoints
    simple_latest_step = checkpoint_info.get("simple_latest", {}).get("step", None)
    simple_best_step = checkpoint_info.get("simple_best", {}).get("step", None)
    latent_latest_step = checkpoint_info.get("latent_latest", {}).get("step", None)
    latent_best_step = checkpoint_info.get("latent_best", {}).get("step", None)

    # Check if synchronization is needed
    if not (
        simple_latest_exists
        and simple_best_exists
        and latent_latest_exists
        and latent_best_exists
    ):
        print("\n[ERROR] Error: Some checkpoint files are missing")
        return False

    # Check if all steps are valid integers
    if not all(
        isinstance(step, int)
        for step in [
            simple_latest_step,
            simple_best_step,
            latent_latest_step,
            latent_best_step,
        ]
        if step is not None
    ):
        print("\n[ERROR] Error: Some checkpoints have invalid step values")
        return False

    # Determine target step if not specified
    if target_step is None:
        steps = [
            step
            for step in [
                simple_latest_step,
                simple_best_step,
                latent_latest_step,
                latent_best_step,
            ]
            if step is not None and isinstance(step, int)
        ]
        if not steps:
            print("\n[ERROR] Error: No valid steps found in checkpoints")
            return False
        target_step = min(steps)
        print(f"\nUsing lowest step value as target: {target_step}")

    # Create backup directory
    if backup:
        backup_dir = f"checkpoints/backup_{int(time.time())}"
        os.makedirs(backup_dir, exist_ok=True)

        # Backup files
        for name, path in {
            "simple_latest": "checkpoints/simpletransformer/simpletransformer_latest.pt",
            "simple_best": "checkpoints/simpletransformer/simpletransformer_best.pt",
            "latent_latest": "checkpoints/latenttransformer/latenttransformer_latest.pt",
            "latent_best": "checkpoints/latenttransformer/latenttransformer_best.pt",
        }.items():
            if os.path.exists(path):
                backup_path = os.path.join(backup_dir, os.path.basename(path))
                shutil.copy2(path, backup_path)
                print(f"[BACKUP] Backed up {name} to {backup_path}")

    # Update checkpoints
    paths_to_update = []
    if target_type is None or target_type == "latest":
        paths_to_update.extend(
            [
                "checkpoints/simpletransformer/simpletransformer_latest.pt",
                "checkpoints/latenttransformer/latenttransformer_latest.pt",
            ]
        )

    if target_type is None or target_type == "best":
        paths_to_update.extend(
            [
                "checkpoints/simpletransformer/simpletransformer_best.pt",
                "checkpoints/latenttransformer/latenttransformer_best.pt",
            ]
        )

    # Update step in each checkpoint
    for path in paths_to_update:
        checkpoint = torch.load(path, map_location="cpu")
        old_step = checkpoint.get("step", None)
        checkpoint["step"] = target_step
        torch.save(checkpoint, path)
        print(
            f"[UPDATE] Updated {os.path.basename(path)}: Step {old_step} -> {target_step}"
        )

    print("\n[SUCCESS] Checkpoint synchronization complete!")
    print(f"All checkpoints now have step = {target_step}")
    return True


def create_missing_checkpoints():
    """Create any missing checkpoint files by copying from existing ones"""
    checkpoint_info = check_checkpoints()

    # Determine which checkpoints exist
    simple_latest_exists = checkpoint_info.get("simple_latest", {}).get("exists", False)
    simple_best_exists = checkpoint_info.get("simple_best", {}).get("exists", False)
    latent_latest_exists = checkpoint_info.get("latent_latest", {}).get("exists", False)
    latent_best_exists = checkpoint_info.get("latent_best", {}).get("exists", False)

    # Create directories if they don't exist
    os.makedirs("checkpoints/simpletransformer", exist_ok=True)
    os.makedirs("checkpoints/latenttransformer", exist_ok=True)

    # Copy checkpoints to create missing ones
    changes_made = False

    # Handle SimpleTransformer checkpoints
    if simple_latest_exists and not simple_best_exists:
        shutil.copy2(
            "checkpoints/simpletransformer/simpletransformer_latest.pt",
            "checkpoints/simpletransformer/simpletransformer_best.pt",
        )
        print("[COPY] Created simple_best from simple_latest")
        changes_made = True
    elif simple_best_exists and not simple_latest_exists:
        shutil.copy2(
            "checkpoints/simpletransformer/simpletransformer_best.pt",
            "checkpoints/simpletransformer/simpletransformer_latest.pt",
        )
        print("[COPY] Created simple_latest from simple_best")
        changes_made = True

    # Handle LatentTransformer checkpoints
    if latent_latest_exists and not latent_best_exists:
        shutil.copy2(
            "checkpoints/latenttransformer/latenttransformer_latest.pt",
            "checkpoints/latenttransformer/latenttransformer_best.pt",
        )
        print("[COPY] Created latent_best from latent_latest")
        changes_made = True
    elif latent_best_exists and not latent_latest_exists:
        shutil.copy2(
            "checkpoints/latenttransformer/latenttransformer_best.pt",
            "checkpoints/latenttransformer/latenttransformer_latest.pt",
        )
        print("[COPY] Created latent_latest from latent_best")
        changes_made = True

    # Cross-copy between models if needed
    if not (simple_latest_exists or simple_best_exists) and (
        latent_latest_exists or latent_best_exists
    ):
        # Get the latent checkpoint that exists
        latent_src = (
            "checkpoints/latenttransformer/latenttransformer_latest.pt"
            if latent_latest_exists
            else "checkpoints/latenttransformer/latenttransformer_best.pt"
        )

        # Load the latent checkpoint
        latent_checkpoint = torch.load(latent_src, map_location="cpu")

        # We can't directly use the state dict because model architectures differ
        # But we can create a placeholder with the same step
        placeholder_checkpoint = {
            "step": latent_checkpoint.get("step", 0),
            "val_loss": latent_checkpoint.get("val_loss", float("inf")),
            "val_sequence_accuracy": latent_checkpoint.get(
                "val_sequence_accuracy", 0.0
            ),
            "model_state_dict": {},  # Empty state dict - will need to be fixed manually
            "optimizer_state_dict": {}
            if "optimizer_state_dict" in latent_checkpoint
            else None,
        }

        # Save placeholder checkpoints
        torch.save(
            placeholder_checkpoint,
            "checkpoints/simpletransformer/simpletransformer_latest.pt",
        )
        torch.save(
            placeholder_checkpoint,
            "checkpoints/simpletransformer/simpletransformer_best.pt",
        )
        print(
            "[WARNING] Created placeholder SimpleTransformer checkpoints with empty state_dict"
        )
        print(
            "[WARNING] You must manually initialize SimpleTransformer before training"
        )
        changes_made = True

    elif not (latent_latest_exists or latent_best_exists) and (
        simple_latest_exists or simple_best_exists
    ):
        # Get the simple checkpoint that exists
        simple_src = (
            "checkpoints/simpletransformer/simpletransformer_latest.pt"
            if simple_latest_exists
            else "checkpoints/simpletransformer/simpletransformer_best.pt"
        )

        # Load the simple checkpoint
        simple_checkpoint = torch.load(simple_src, map_location="cpu")

        # We can't directly use the state dict because model architectures differ
        # But we can create a placeholder with the same step
        placeholder_checkpoint = {
            "step": simple_checkpoint.get("step", 0),
            "val_loss": simple_checkpoint.get("val_loss", float("inf")),
            "val_sequence_accuracy": simple_checkpoint.get(
                "val_sequence_accuracy", 0.0
            ),
            "model_state_dict": {},  # Empty state dict - will need to be fixed manually
            "optimizer_state_dict": {}
            if "optimizer_state_dict" in simple_checkpoint
            else None,
        }

        # Save placeholder checkpoints
        torch.save(
            placeholder_checkpoint,
            "checkpoints/latenttransformer/latenttransformer_latest.pt",
        )
        torch.save(
            placeholder_checkpoint,
            "checkpoints/latenttransformer/latenttransformer_best.pt",
        )
        print(
            "[WARNING] Created placeholder LatentTransformer checkpoints with empty state_dict"
        )
        print(
            "[WARNING] You must manually initialize LatentTransformer before training"
        )
        changes_made = True

    if not changes_made:
        print("[OK] No missing checkpoints to create")

    return changes_made


def verify_checkpoint_content():
    """Perform deeper verification of checkpoint content"""
    checkpoint_info = check_checkpoints()

    # Ensure all checkpoints have model_state_dict
    for name, info in checkpoint_info.items():
        if not info.get("exists", False):
            continue

        path = {
            "simple_latest": "checkpoints/simpletransformer/simpletransformer_latest.pt",
            "simple_best": "checkpoints/simpletransformer/simpletransformer_best.pt",
            "latent_latest": "checkpoints/latenttransformer/latenttransformer_latest.pt",
            "latent_best": "checkpoints/latenttransformer/latenttransformer_best.pt",
        }[name]

        checkpoint = torch.load(path, map_location="cpu")
        problems = []

        # Check for empty model_state_dict
        if "model_state_dict" not in checkpoint:
            problems.append("Missing model_state_dict")
        elif not checkpoint["model_state_dict"]:
            problems.append("Empty model_state_dict")

        # Check for reasonable step value
        if "step" not in checkpoint:
            problems.append("Missing step value")
        elif (
            not isinstance(checkpoint.get("step"), int) or checkpoint.get("step", 0) < 0
        ):
            problems.append("Invalid step value")

        if problems:
            print(f"[WARNING] {name} has issues: {', '.join(problems)}")
        else:
            print(f"[OK] {name} content verified")

    return True


def fix_orig_mod_prefixes():
    """Fix _orig_mod prefixes in the checkpoint state dictionaries"""
    checkpoint_info = check_checkpoints()

    # Process each checkpoint
    for name, info in checkpoint_info.items():
        if not info.get("exists", False):
            continue

        path = {
            "simple_latest": "checkpoints/simpletransformer/simpletransformer_latest.pt",
            "simple_best": "checkpoints/simpletransformer/simpletransformer_best.pt",
            "latent_latest": "checkpoints/latenttransformer/latenttransformer_latest.pt",
            "latent_best": "checkpoints/latenttransformer/latenttransformer_best.pt",
        }[name]

        modified = False
        checkpoint = torch.load(path, map_location="cpu")

        if "model_state_dict" in checkpoint and checkpoint["model_state_dict"]:
            old_state_dict = checkpoint["model_state_dict"]
            new_state_dict = {}

            for k, v in old_state_dict.items():
                if k.startswith("_orig_mod."):
                    new_key = k[len("_orig_mod.") :]
                    new_state_dict[new_key] = v
                    modified = True
                else:
                    new_state_dict[k] = v

            if modified:
                checkpoint["model_state_dict"] = new_state_dict
                torch.save(checkpoint, path)
                print(f"[FIXED] Removed _orig_mod prefixes from {name}")
            else:
                print(f"[OK] No _orig_mod prefixes found in {name}")
        else:
            print(
                f"[WARNING] Cannot fix prefixes for {name}: No valid model_state_dict"
            )

    return True


def add_optimizer_state_if_missing():
    """Add empty optimizer_state_dict to checkpoints if missing"""
    checkpoint_info = check_checkpoints()

    # Process each checkpoint
    for name, info in checkpoint_info.items():
        if not info.get("exists", False):
            continue

        if not info.get("has_optimizer", False):
            path = {
                "simple_latest": "checkpoints/simpletransformer/simpletransformer_latest.pt",
                "simple_best": "checkpoints/simpletransformer/simpletransformer_best.pt",
                "latent_latest": "checkpoints/latenttransformer/latenttransformer_latest.pt",
                "latent_best": "checkpoints/latenttransformer/latenttransformer_best.pt",
            }[name]

            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["optimizer_state_dict"] = {}
            torch.save(checkpoint, path)
            print(f"[FIXED] Added empty optimizer_state_dict to {name}")

    return True


def main():
    """Main function to parse arguments and execute tasks"""
    parser = argparse.ArgumentParser(description="Checkpoint management utility")
    parser.add_argument(
        "--fix-missing", action="store_true", help="Fix missing checkpoints"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify checkpoint content"
    )
    parser.add_argument(
        "--fix-prefixes",
        action="store_true",
        help="Fix _orig_mod prefixes in checkpoints",
    )
    parser.add_argument(
        "--add-optimizer-state",
        action="store_true",
        help="Add optimizer state if missing",
    )
    parser.add_argument(
        "--sync", action="store_true", help="Synchronize checkpoint steps"
    )
    parser.add_argument("--target-step", type=int, help="Target step to set")
    parser.add_argument(
        "--target-type",
        choices=["latest", "best"],
        help="Which checkpoint type to sync",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup when syncing"
    )
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")

    args = parser.parse_args()

    # Default action is to check checkpoints
    if len(sys.argv) == 1:
        check_checkpoints()
        return

    # Process actions in a specific order
    if args.fix_all or args.fix_missing:
        create_missing_checkpoints()

    if args.fix_all or args.verify:
        verify_checkpoint_content()

    if args.fix_all or args.fix_prefixes:
        fix_orig_mod_prefixes()

    if args.fix_all or args.add_optimizer_state:
        add_optimizer_state_if_missing()

    if args.fix_all or args.sync:
        sync_checkpoints(args.target_step, args.target_type, not args.no_backup)


if __name__ == "__main__":
    main()
