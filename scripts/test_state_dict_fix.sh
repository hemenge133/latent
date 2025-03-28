#!/bin/bash

# Go to project directory
cd /home/h/Devel/latent

# Activate virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "=== Testing state dict loading fix for mismatched architectures ==="

# Phase 1: Create checkpoint with 1 layer
echo "Phase 1: Creating checkpoint with 1 layer model"
python main.py --d-model 32 --num-layers 1 --num-latent 1 --min-digits 1 --max-digits 1 --batch-size 8 --max-steps 2 --seed 42

# Get run ID
LATEST_RUN=$(ls -td runs/parallel_comparison/* | head -n 1 | xargs basename)
echo "Latest run ID: $LATEST_RUN"

# Phase 2: Resume with different architecture 
echo "Phase 2: Resuming with 4 layers (should work with filtering)"
python main.py --d-model 32 --num-layers 4 --num-latent 1 --min-digits 1 --max-digits 1 --batch-size 8 --max-steps 4 --seed 42 --resume --run-id $LATEST_RUN

echo "=== Test complete ===" 