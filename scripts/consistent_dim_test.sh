#!/bin/bash
cd /home/h/Devel/latent
source ~/.virtualenvs/latent/bin/activate

echo "====================== CONSISTENT DIMENSIONS TEST ======================="
echo "Testing training continuation with dimensions from checkpoints"
echo "==================================================================="

# Define initial model parameters
D_MODEL=64
NUM_LAYERS=2
NUM_LATENT=2
MIN_DIGITS=1
MAX_DIGITS=3
BATCH_SIZE=16
SEED=42

# Step 1: Initial training for 5 steps
echo ""
echo "PHASE 1: Initial training for 5 steps with d_model=$D_MODEL, layers=$NUM_LAYERS"
echo "-----------------------------------------------------------------------"
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 5 \
  --save-every 5 \
  --seed $SEED

# Get the latest run-id
LATEST_RUN_ID=$(ls -t runs/parallel_comparison | grep -E "^[0-9]{8}-[0-9]{6}_d${D_MODEL}_l${NUM_LAYERS}_n${NUM_LATENT}" | head -n 1)
echo ""
echo "Run ID from Phase 1: $LATEST_RUN_ID"
echo ""

# Step 2: Resume training for another 10 steps (total 15)
# IMPORTANT: Use the exact same dimensions as in the first training phase
echo "PHASE 2: Resuming training for 10 more steps (target: 15 total)"
echo "----------------------------------------------------------------"
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 15 \
  --resume \
  --run-id $LATEST_RUN_ID \
  --seed $SEED

echo ""
echo "Test complete. Check TensorBoard logs to verify continuous training:"
echo "tensorboard --logdir=runs/parallel_comparison" 