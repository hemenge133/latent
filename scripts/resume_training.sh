#!/bin/bash
# Script to resume training from checkpoints or run ID

cd /home/h/Devel/latent

# Activate virtual environment
source ~/.virtualenvs/latent/bin/activate

# Get the run ID from command line argument
RUN_ID=$1

# Get whether to force command-line config
FORCE_CONFIG=$2

if [ -z "$RUN_ID" ]; then
    echo "No run ID provided, using most recent run"
fi

ADDITIONAL_ARGS=""
if [ ! -z "$RUN_ID" ]; then
    ADDITIONAL_ARGS="--run-id $RUN_ID"
    
    # Add force-config if specified
    if [ "$FORCE_CONFIG" = "force" ]; then
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS --force-config"
        echo "Forcing command-line config over checkpoint config"
    else
        echo "Using configuration from checkpoint ($RUN_ID)"
    fi
else
    echo "Using configuration from command line"
fi

# Resume training with updated parameters
# These will only be used if --force-config is specified or no config in checkpoint
python main.py \
    --resume \
    --use-checkpointing \
    --d-model 768 \
    --num-layers 8 \
    --num-latent 16 \
    --min-digits 3 \
    --max-steps 20000 \
    --batch-size 512 \
    $ADDITIONAL_ARGS

# Usage examples:
# ./scripts/resume_training.sh                            # Resume without specifying run ID (uses most recent)
# ./scripts/resume_training.sh 20250325-152939_d768_l8_n16  # Resume specific run by ID with checkpoint config
# ./scripts/resume_training.sh 20250325-152939_d768_l8_n16 force  # Resume specific run but use command line config 