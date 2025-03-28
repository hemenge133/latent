#!/bin/bash
# Script to test TensorBoard log continuity with clean checkpoints

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "==============================================="
echo "     CLEAN CHECKPOINT CONTINUITY TEST          "
echo "==============================================="

# Create clean run ID and directories
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CLEAN_RUN_ID="${TIMESTAMP}_clean_test"
LOG_DIR="runs/parallel_comparison/$CLEAN_RUN_ID"

# First, let's remove any existing checkpoints to ensure clean test
echo "Cleaning checkpoints directories..."
rm -f checkpoints/simpletransformer/simpletransformer_*.pt
rm -f checkpoints/latenttransformer/latenttransformer_*.pt

# Create necessary directories
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

# Run the first training phase
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 5 \
  --save-every 1 \
  --seed $SEED \
  --run-id $CLEAN_RUN_ID

# Wait for checkpoints to be written
sleep 2

# PHASE 2: Continued Training
echo ""
echo "PHASE 2: Continuing training for 5 more steps (goal: 10 total)"
echo "-----------------------------------------------------------"

# Run the second training phase with resume using the same run-id
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 10 \
  --resume \
  --run-id $CLEAN_RUN_ID \
  --seed $SEED

echo ""
echo "Test complete! Check TensorBoard logs to see continuous training curve:"
echo "tensorboard --logdir=$LOG_DIR"
echo ""
echo "Run this command to view the continuous logs:"
echo "wsl -e bash -c \"cd /home/h/Devel/latent && source ~/.virtualenvs/latent/bin/activate && tensorboard --logdir=$LOG_DIR\""

# Restore original main.py if needed
if [ -f main.py.bak ]; then
  echo "Restoring original main.py..."
  mv main.py.bak main.py
fi 