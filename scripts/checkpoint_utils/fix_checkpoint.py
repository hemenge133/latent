#!/usr/bin/env python
import torch

# Load the latent checkpoint
checkpoint_path = "checkpoints/latenttransformer/latenttransformer_best.pt"
output_path = "checkpoints/latenttransformer/latenttransformer_latest.pt"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Set the step count
original_step = checkpoint.get("step", "Not found")
print(f"Original step: {original_step}")

checkpoint["step"] = 3889
print(f"Updated step to: 3889")

# Save the checkpoint
torch.save(checkpoint, output_path)
print(f"Saved updated checkpoint to: {output_path}")
