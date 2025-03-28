#!/bin/bash
# Script to test checkpoint resuming while preserving TensorBoard logs

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

echo "============================================="
echo "     RESUME TEST WITH PRESERVED LOGS         "
echo "============================================="

# Temporarily modify main.py to not delete the log directory
echo "Temporarily modifying main.py to preserve logs..."
cp main.py main.py.bak
sed -i 's/shutil.rmtree(log_dir)/# Disabled: shutil.rmtree(log_dir)/' main.py
sed -i 's/os.makedirs(log_dir, exist_ok=True)/os.makedirs(log_dir, exist_ok=True)  # Always create dir/' main.py

# Define model parameters (smaller model for quicker training)
D_MODEL=32
NUM_LAYERS=1
NUM_LATENT=1
MIN_DIGITS=1
MAX_DIGITS=2
BATCH_SIZE=8
SEED=42

# Stage 1: Initial training for 5 steps
echo ""
echo "STAGE 1: Initial training for 5 steps"
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
echo "Run ID from Stage 1: $LATEST_RUN_ID"
echo ""

# Stage 2: Continue training for another 5 steps (10 total)
echo "STAGE 2: Continuing training for 5 more steps (goal: 10 total)"
echo "-------------------------------------------------------------"
python main.py \
  --d-model $D_MODEL \
  --num-layers $NUM_LAYERS \
  --num-latent $NUM_LATENT \
  --min-digits $MIN_DIGITS \
  --max-digits $MAX_DIGITS \
  --batch-size $BATCH_SIZE \
  --max-steps 10 \
  --resume \
  --run-id $LATEST_RUN_ID \
  --seed $SEED

# Restore original main.py
echo "Restoring original main.py..."
mv main.py.bak main.py

echo ""
echo "Test complete! Check TensorBoard logs to verify continuous training:"
echo "tensorboard --logdir=runs/parallel_comparison/$LATEST_RUN_ID"
echo ""
echo "To run TensorBoard, use:"
echo "wsl -e bash -c \"cd /home/h/Devel/latent && source ~/.virtualenvs/latent/bin/activate && tensorboard --logdir=runs/parallel_comparison/$LATEST_RUN_ID\"" 