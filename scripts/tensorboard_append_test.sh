#!/bin/bash
# Script to test TensorBoard log continuity by using the same log directory

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "==============================================="
echo "     TENSORBOARD CONTINUOUS LOGGING TEST       "
echo "==============================================="

# Define a common run ID for both phases
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COMMON_RUN_ID="${TIMESTAMP}_continuous_test"
LOG_DIR="runs/parallel_comparison/$COMMON_RUN_ID"

# Create log directories manually
mkdir -p "$LOG_DIR/simple"
mkdir -p "$LOG_DIR/latent"

# Define model parameters (smaller model for quicker training)
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=2
BATCH_SIZE=8
SEED=42

# PHASE 1: Initial Training
echo ""
echo "PHASE 1: Initial training for 5 steps"
echo "------------------------------------"

# Run the first training phase with custom log directory
python -c "
import os
import sys
import subprocess

# Construct the command
cmd = [
    'python', 'main.py',
    '--d-model', '$D_MODEL',
    '--num-layers', '$NUM_LAYERS',
    '--num-latent', '$NUM_LATENT',
    '--min-digits', '$MIN_DIGITS',
    '--max-digits', '$MAX_DIGITS',
    '--batch-size', '$BATCH_SIZE',
    '--max-steps', '5',
    '--save-every', '5',
    '--seed', '$SEED'
]

# Set up environment variables for custom log directory
env = os.environ.copy()
env['TENSORBOARD_LOG_DIR'] = '$LOG_DIR'

# Execute command
process = subprocess.Popen(cmd, env=env)
process.wait()
"

# Wait for checkpoint to be written
sleep 2

# Get the latest run ID from normal run
LATEST_RUN_ID=$(ls -t runs/parallel_comparison | grep -E "^[0-9]{8}-[0-9]{6}_d${D_MODEL}_l${NUM_LAYERS}_n${NUM_LATENT}" | head -n 1)
echo ""
echo "Run ID from normal logs: $LATEST_RUN_ID"
echo "Custom continuous log dir: $LOG_DIR"
echo ""

# PHASE 2: Continued Training
echo "PHASE 2: Continuing training for 5 more steps (goal: 10 total)"
echo "-----------------------------------------------------------"

# Run the second training phase, with resume and the same custom log directory
python -c "
import os
import sys
import subprocess

# Construct the command
cmd = [
    'python', 'main.py',
    '--d-model', '$D_MODEL',
    '--num-layers', '$NUM_LAYERS',
    '--num-latent', '$NUM_LATENT',
    '--min-digits', '$MIN_DIGITS',
    '--max-digits', '$MAX_DIGITS',
    '--batch-size', '$BATCH_SIZE',
    '--max-steps', '10',
    '--resume',
    '--run-id', '$LATEST_RUN_ID',
    '--seed', '$SEED'
]

# Set up environment variables for custom log directory
env = os.environ.copy()
env['TENSORBOARD_LOG_DIR'] = '$LOG_DIR'

# Execute command
process = subprocess.Popen(cmd, env=env)
process.wait()
"

echo ""
echo "Test complete! Check TensorBoard logs to see continuous training curve:"
echo "tensorboard --logdir=$LOG_DIR"
echo ""
echo "Run this command to view the continuous logs:"
echo "wsl -e bash -c \"cd /home/h/Devel/latent && source ~/.virtualenvs/latent/bin/activate && tensorboard --logdir=$LOG_DIR\"" 