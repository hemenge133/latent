#!/usr/bin/env python
import torch
import os
import sys


def check_checkpoint(path):
    print(f"Checking: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu")
        step = checkpoint.get("step", "Not found")
        val_loss = checkpoint.get("val_loss", "Not found")
        print(f"  Step: {step}")
        print(f"  Val loss: {val_loss}")
    except Exception as e:
        print(f"  Error: {e}")


# Check all checkpoint files
for root, dirs, files in os.walk("checkpoints"):
    for file in files:
        if file.endswith(".pt"):
            check_checkpoint(os.path.join(root, file))
