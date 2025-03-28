#!/bin/bash
# Script to test continuous TensorBoard curves with custom SummaryWriter

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "==============================================="
echo "     CONTINUOUS TENSORBOARD CURVES TEST        "
echo "==============================================="

# First, let's apply our patches
echo "Applying patches..."
python scripts/patch_training_loop.py
python scripts/patch_main_custom_log.py

# Define run ID and directory for testing
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="runs/continuous_test_$TIMESTAMP"

# Define model parameters (smaller model for quicker training)
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=2
BATCH_SIZE=8
SEED=42

# Clean checkpoints
rm -f checkpoints/simpletransformer/simpletransformer_*.pt
rm -f checkpoints/latenttransformer/latenttransformer_*.pt

# Create the log directories
mkdir -p "$RUN_DIR/simple"
mkdir -p "$RUN_DIR/latent"

# Create a unique run ID for TensorBoard
export CUSTOM_LOG_DIR="$RUN_DIR"

# PHASE 1: Initial Training
echo ""
echo "PHASE 1: Initial training for 5 steps"
echo "------------------------------------"

python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 5 \
  --save-every 1 \
  --seed $SEED

# Wait for checkpoints to be written
sleep 2

# PHASE 2: Continued Training
echo ""
echo "PHASE 2: Continuing training for 5 more steps (goal: 10 total)"
echo "-----------------------------------------------------------"

python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 10 \
  --save-every 1 \
  --seed $SEED

echo ""
echo "PHASE 3: Final 5 steps (goal: 15 total)"
echo "-----------------------------------"

python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 15 \
  --save-every 1 \
  --seed $SEED

echo ""
echo "Test complete! Check TensorBoard to see continuous curves:"
echo "tensorboard --logdir=$RUN_DIR"
echo ""
echo "Run this command to view TensorBoard:"
echo "wsl -e bash -c \"cd /home/h/Devel/latent && source ~/.virtualenvs/latent/bin/activate && tensorboard --logdir=$RUN_DIR\""

# Restore original files
echo "Restoring original files..."
if [ -f src/TrainingLoop.py.bak ]; then
  mv src/TrainingLoop.py.bak src/TrainingLoop.py
fi

if [ -f main.py.bak ]; then
  mv main.py.bak main.py
fi 