#!/bin/bash

# Activate the Python environment
source ~/.virtualenvs/latent/bin/activate

# Run the simplified model training with single-digit multiplication
python main.py \
    --d-model 64 \
    --num-layers 2 \
    --num-latent 4 \
    --min-digits 1 \
    --max-digits 1 \
    --batch-size 16 \
    --max-steps 100 \
    --save-every 25 \
    > simple_test_output.log 2>&1

echo "Training completed. Check simple_test_output.log for details." 