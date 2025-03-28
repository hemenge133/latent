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
MAX_STEPS=10
SEED=42

echo "=== Running test with stability_window fix ==="

# Run training with minimal parameters to test if stability_window is working
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps $MAX_STEPS \
  --seed $SEED

echo "=== Test complete ==="
echo "Check the logs above for any 'stability_window' errors" 