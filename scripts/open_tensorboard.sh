#!/bin/bash
# Script to open TensorBoard with the latest run

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

# Find the latest run ID if not specified
if [ -z "$1" ]; then
  # Get the latest run folder
  LATEST_RUN=$(ls -t runs/parallel_comparison/ | head -1)
  echo "Using latest run: $LATEST_RUN"
  LOG_DIR="runs/parallel_comparison/$LATEST_RUN"
else
  # Use the specified run ID
  echo "Using specified run: $1"
  LOG_DIR="runs/parallel_comparison/$1"
fi

# Start TensorBoard
echo "Starting TensorBoard with log directory: $LOG_DIR"
echo "If this doesn't work in WSL, you may need to access TensorBoard via your browser at http://localhost:6006"
tensorboard --logdir="$LOG_DIR" 