#!/bin/bash

# Change to the project directory
cd /home/h/Devel/latent

# Activate the virtual environment
source ~/.virtualenvs/latent/bin/activate

# Set parameters for a quick test
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=2
BATCH_SIZE=8
MAX_STEPS=5
SEED=42

echo "=== Phase 1: Initial Training ==="

# Run initial training with minimal parameters
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps $MAX_STEPS \
  --seed $SEED \
  --save-every 5

# Get the latest run ID
LATEST_RUN=$(ls -t runs/parallel_comparison/ | grep -E "^[0-9]{8}-[0-9]{6}_d${D_MODEL}_l${NUM_LAYERS}_n${NUM_LATENT}$" | head -n 1)

echo "=== Phase 2: Resume Training ==="
echo "Resuming from run: $LATEST_RUN"

# Resume training for additional steps
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 10 \
  --seed $SEED \
  --resume \
  --run-id $LATEST_RUN

echo "=== Test complete ==="
echo "Check the logs above for any 'stability_window' errors" 