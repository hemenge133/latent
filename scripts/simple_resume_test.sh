#!/bin/bash
# Script to test the step counter fix

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "============================================="
echo "     TESTING RESUME WITH STEP COUNT FIX      "
echo "============================================="

# Initial training (only 10 steps)
echo "PHASE 1: Training for 10 steps..."
python main.py \
    --d-model 32 \
    --num-layers 1 \
    --num-latent 1 \
    --min-digits 1 \
    --max-digits 2 \
    --batch-size 8 \
    --max-steps 10 \
    --save-every 5 \
    --seed 42

# Get the latest run ID
latest_run_id=$(python scripts/list_runs.py --limit 1 | grep -oP '\d{8}-\d{6}_d\d+_l\d+_n\d+' | head -1)

echo "Got latest run ID: $latest_run_id"

# Resume training for 10 more steps (to reach 20 total)
echo "PHASE 2: Resuming for 10 more steps (to reach 20 total)..."
python main.py \
    --resume \
    --run-id $latest_run_id \
    --max-steps 20 \
    --save-every 5 \
    --seed 42

# Verify by examining logs
echo "Check TensorBoard to verify that training continued properly."
echo "Run: tensorboard --logdir=runs/parallel_comparison" 