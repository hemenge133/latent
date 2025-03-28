#!/bin/bash
# Script to test resuming with consistent model parameters

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "====================== RESUME TEST SCRIPT ======================="
echo "Testing training continuation with consistent model parameters"
echo "================================================================"

# Define model parameters
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=2
BATCH_SIZE=8
SEED=42

# Step 1: Initial training for 5 steps
echo ""
echo "PHASE 1: Initial training for 5 steps"
echo "-------------------------------------"
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
echo "PHASE 2: Resuming training for 10 more steps (target: 15 total)"
echo "--------------------------------------------------------------"
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

# Get the latest run-id (should be from the resumed run)
RESUMED_RUN_ID=$(ls -t runs/parallel_comparison | grep -E "^[0-9]{8}-[0-9]{6}_d${D_MODEL}_l${NUM_LAYERS}_n${NUM_LATENT}" | head -n 1)
echo ""
echo "Run ID from Phase 2: $RESUMED_RUN_ID"
echo ""

# Step 3: Resume one more time for an additional 10 steps (total 25)
echo "PHASE 3: Resuming training for 10 more steps (target: 25 total)"
echo "--------------------------------------------------------------"
python main.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-latent $NUM_LATENT \
    --min-digits $MIN_DIGITS \
    --max-digits $MAX_DIGITS \
    --batch-size $BATCH_SIZE \
    --max-steps 25 \
    --resume \
    --run-id $RESUMED_RUN_ID \
    --seed $SEED

echo ""
echo "Test complete. Check TensorBoard logs to verify continuous training:"
echo "tensorboard --logdir=runs/parallel_comparison" 