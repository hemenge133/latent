#!/bin/bash
# Script to run training with modified parameters to prevent overfitting

cd /home/h/Devel/latent

# Activate virtual environment
source ~/.virtualenvs/latent/bin/activate

# Run training with updated parameters
python main.py \
    --use-checkpointing \
    --d-model 768 \
    --num-layers 8 \
    --num-latent 16 \
    --min-digits 3 \
    --max-steps 20000 \
    --batch-size 512
