#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to organize and deduplicate scripts in the repository.
"""
import os
import shutil
import sys
import glob


def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def move_file(src, dest):
    """Move a file if it exists"""
    if os.path.exists(src):
        # Create destination directory if needed
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        # Move the file
        shutil.move(src, dest)
        print(f"Moved: {src} -> {dest}")
    else:
        print(f"File not found: {src}")


def make_executable(path):
    """Make a file executable"""
    if os.path.exists(path):
        current_permissions = os.stat(path).st_mode
        os.chmod(path, current_permissions | 0o111)  # Add execute permission
        print(f"Made executable: {path}")
    else:
        print(f"File not found: {path}")


def main():
    # Change to repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working in repository: {repo_root}")

    # Create necessary directories
    create_directory("scripts/archive")
    create_directory("scripts/checkpoint_utils")

    # List of duplicate scripts to archive
    to_archive = [
        # Deprecated/redundant scripts
        ("scripts/sync_checkpoints.py", "scripts/archive/sync_checkpoints.py"),
        ("scripts/check_checkpoints.sh", "scripts/archive/check_checkpoints.sh"),
        (
            "scripts/check_checkpoint_status.sh",
            "scripts/archive/check_checkpoint_status.sh",
        ),
        (
            "scripts/checkpoint_utils/check_sync_checkpoints.py.bak",
            "scripts/archive/check_sync_checkpoints.py.bak",
        ),
    ]

    # Scripts to move to checkpoint_utils
    to_checkpoint_utils = [
        # Root files that are checkpoint utilities
        ("check_checkpoints.py", "scripts/checkpoint_utils/check_checkpoints.py"),
        ("resume_training.py", "scripts/checkpoint_utils/resume_training.py"),
    ]

    # Root level scripts to archive
    root_scripts_to_archive = [
        # Archived scripts
        ("sync_to_4550.py", "scripts/archive/sync_to_4550.py"),
        ("parallel_comparison.py", "scripts/archive/parallel_comparison.py"),
        ("test_larger_model.py", "scripts/archive/test_larger_model.py"),
        ("test_refactoring.py", "scripts/archive/test_refactoring.py"),
        ("small_test.py", "scripts/archive/small_test.py"),
    ]

    # Move files to archive
    print("\nArchiving duplicate/redundant scripts:")
    for src, dest in to_archive:
        move_file(src, dest)

    # Move scripts to checkpoint_utils
    print("\nMoving checkpoint utilities to scripts/checkpoint_utils:")
    for src, dest in to_checkpoint_utils:
        move_file(src, dest)

    # Move root scripts to archive
    print("\nArchiving test and obsolete scripts:")
    for src, dest in root_scripts_to_archive:
        move_file(src, dest)

    # Make sure all Python scripts are executable
    print("\nMaking scripts executable:")
    script_paths = [
        # Python scripts in scripts directory
        *glob.glob("scripts/*.py"),
        # Python scripts in checkpoint_utils directory
        *glob.glob("scripts/checkpoint_utils/*.py"),
        # Shell scripts
        *glob.glob("scripts/*.sh"),
        *glob.glob("scripts/checkpoint_utils/*.sh"),
        # Main script
        "run_training.sh",
    ]

    for script_path in script_paths:
        make_executable(script_path)

    # Check for any remaining duplicate utilities
    print("\nChecking for additional checkpoint utilities in root directory:")
    for root_file in glob.glob("*.py"):
        if "checkpoint" in root_file.lower() and not root_file.startswith(
            (
                "test_",
                "main.py",
                "parallel_",
                "utils.py",
                "config.py",
                "metrics.py",
                "losses.py",
                "Dataset.py",
                "stable_comparison",
            )
        ):
            dest = f"scripts/checkpoint_utils/{root_file}"
            if not os.path.exists(dest):
                print(f"Found additional checkpoint utility: {root_file}")
                move_file(root_file, dest)

    # Print summary
    print("\nFile organization complete!")
    print("The repository is now organized with:")
    print("- Main scripts in the scripts/ directory")
    print("- Checkpoint utilities in scripts/checkpoint_utils/")
    print("- Obsolete and duplicate scripts in scripts/archive/")

    # Print directory contents for verification
    print("\nCurrent structure:")
    print("Scripts directory:")
    for f in sorted(glob.glob("scripts/*")):
        print(f"  {f}")

    print("\nCheckpoint utilities:")
    for f in sorted(glob.glob("scripts/checkpoint_utils/*")):
        print(f"  {f}")

    print("\nArchived scripts:")
    for f in sorted(glob.glob("scripts/archive/*")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
