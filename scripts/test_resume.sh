#!/bin/bash
# Script to test checkpoint resuming with a small model

# Change to the project directory
cd /home/h/Devel/latent

# Activate the Python 3 virtual environment if it exists
if [ -f ~/.virtualenvs/latent/bin/activate ]; then
    echo "Activating virtual environment"
    source ~/.virtualenvs/latent/bin/activate
fi

# ==================== MULTI-STAGE RESUME TEST =====================
# This script tests the resume functionality by training models in multiple stages
# It ensures that the training continues properly across each stage
# and that TensorBoard graphs align seamlessly
# ================================================================

# Set parameters for training
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=1
BATCH_SIZE=8
SEED=42

echo "=== Phase 1: Initial Training ==="
# Run initial training for 3 steps
python3 main.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-latent $NUM_LATENT \
    --min-digits $MIN_DIGITS \
    --max-digits $MAX_DIGITS \
    --batch-size $BATCH_SIZE \
    --max-steps 3 \
    --seed $SEED

# Get the latest run ID
LATEST_RUN=$(ls -td runs/parallel_comparison/* | head -n 1 | xargs basename)
echo "Latest run ID: $LATEST_RUN"

echo "=== Phase 2: Resume Training ==="
# Resume training for additional 3 steps (total of 6)
python3 main.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-latent $NUM_LATENT \
    --min-digits $MIN_DIGITS \
    --max-digits $MAX_DIGITS \
    --batch-size $BATCH_SIZE \
    --max-steps 6 \
    --seed $SEED \
    --resume \
    --run-id $LATEST_RUN

echo "=== Test complete ==="
echo "Check the logs above to verify that training was resumed correctly"

echo "Test complete! Check TensorBoard logs to verify continuous training:"
echo "tensorboard --logdir=runs/parallel_comparison"
echo ""
echo "Compare Stage 1 run: $LATEST_RUN"
echo ""

echo "Test complete! Check TensorBoard logs to verify continuous training:"
echo "tensorboard --logdir=runs/parallel_comparison"
echo ""
echo "Compare Stage 1 run: $LATEST_RUN" 