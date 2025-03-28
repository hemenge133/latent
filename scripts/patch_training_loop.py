#!/usr/bin/env python3
"""
Patch TrainingLoop.py to use our custom SummaryWriter
"""
import os
import sys
import re
import shutil


def patch_training_loop():
    """Patch TrainingLoop.py to use our custom SummaryWriter"""
    print("Patching TrainingLoop.py to use our custom SummaryWriter")

    # Backup original file
    shutil.copy("src/TrainingLoop.py", "src/TrainingLoop.py.bak")
    print("Created backup: src/TrainingLoop.py.bak")

    # Read TrainingLoop.py
    with open("src/TrainingLoop.py", "r") as f:
        content = f.read()

    # Replace the import line
    content = re.sub(
        r"from torch\.utils\.tensorboard import SummaryWriter",
        "from src.SummaryWriter import SummaryWriter",
        content,
    )

    # Write modified content back
    with open("src/TrainingLoop.py", "w") as f:
        f.write(content)

    print("Successfully patched src/TrainingLoop.py")
    print("To restore original file: mv src/TrainingLoop.py.bak src/TrainingLoop.py")


if __name__ == "__main__":
    patch_training_loop()
