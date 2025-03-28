#!/usr/bin/env python3
"""
Patch main.py to respect the TENSORBOARD_LOG_DIR environment variable
"""
import os
import sys
import re
import shutil


def patch_main_py():
    """Patch main.py to respect TENSORBOARD_LOG_DIR environment variable"""
    print("Patching main.py to respect TENSORBOARD_LOG_DIR environment variable")

    # Backup original file
    shutil.copy("main.py", "main.py.bak")
    print("Created backup: main.py.bak")

    # Read main.py
    with open("main.py", "r") as f:
        lines = f.readlines()

    # Find the line that sets log_dir
    for i, line in enumerate(lines):
        if "log_dir = f" in line and "run_id" in line:
            # Replace the line with one that checks for environment variable
            lines[
                i
            ] = """    # Check for custom log directory from environment
    if os.environ.get('TENSORBOARD_LOG_DIR'):
        log_dir = os.environ.get('TENSORBOARD_LOG_DIR')
        logger.info(f"Using custom log directory from environment: {log_dir}")
    else:
        log_dir = f"runs/parallel_comparison/{run_id}"
"""

            # Also find and modify the line that deletes the log dir
            for j, inner_line in enumerate(lines[i:], start=i):
                if "shutil.rmtree(log_dir)" in inner_line:
                    # Add a condition to not delete if it's a custom log dir
                    lines[
                        j
                    ] = """        # Don't delete custom log directories
        if not os.environ.get('TENSORBOARD_LOG_DIR'):
            shutil.rmtree(log_dir)
"""
                    break

            break

    # Write modified content back
    with open("main.py", "w") as f:
        f.writelines(lines)

    print("Successfully patched main.py")
    print("To restore original file: mv main.py.bak main.py")


if __name__ == "__main__":
    patch_main_py()
