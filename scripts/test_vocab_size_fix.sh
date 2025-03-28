#!/bin/bash

# Change to the project directory
cd /home/h/Devel/latent

# Activate the virtual environment
source ~/.virtualenvs/latent/bin/activate

# This script will test resuming from a specific run ID where we know there's a vocabulary size mismatch
# Replace this with the run ID that has a vocabulary size of 11 but the current code uses 12
RUN_ID="20250325-195248_d64_l2_n4"

echo "=== Testing resume with vocabulary size mismatch fix ==="
echo "Resuming from run ID: $RUN_ID"

# Resume training
python main.py \
  --d-model 768 \
  --num-layers 8 \
  --num-latent 16 \
  --min-digits 3 \
  --max-digits 3 \
  --batch-size 16 \
  --max-steps 5 \
  --seed 42 \
  --resume \
  --run-id $RUN_ID

echo "=== Test complete ==="
echo "Check the logs above to verify that vocabulary size was correctly extracted and used" 