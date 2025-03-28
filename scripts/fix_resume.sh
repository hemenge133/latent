#!/bin/bash
# Script to comprehensively test and fix resume functionality

# Go to project root
cd /home/h/Devel/latent

# Activate the latent virtual environment
source ~/.virtualenvs/latent/bin/activate

# Display banner
echo "============================================="
echo "     COMPREHENSIVE RESUME FIX & TEST         "
echo "============================================="

# Define fixed seed and configuration for consistency
SEED=42
D_MODEL=64
NUM_LAYERS=2
NUM_LATENT=2
BATCH_SIZE=16
MIN_DIGITS=1
MAX_DIGITS=3
MAX_STEPS_INITIAL=20
MAX_STEPS_RESUME=40
SAVE_FREQ=5

echo "Using fixed configuration:"
echo "  Seed: $SEED"
echo "  Model: d=$D_MODEL, layers=$NUM_LAYERS, latent=$NUM_LATENT"
echo "  Data: min_digits=$MIN_DIGITS, max_digits=$MAX_DIGITS, batch=$BATCH_SIZE"
echo "  Training: initial=$MAX_STEPS_INITIAL steps, resume to $MAX_STEPS_RESUME total"

# Clean up any previous runs to avoid confusion
echo "Cleaning up previous test runs..."
rm -rf runs/parallel_comparison/*test_resume*
rm -f debug_data_*.pt

# Save the seed and data order info for initial training to verify consistency
cat > src/debug_dataloading.py << 'EOF'
import torch
import torch.utils.data
import numpy as np
import random
import os
import sys

# This script helps debug data ordering issues by saving DataLoader state
seed = int(sys.argv[1])
run_name = sys.argv[2]

# Set all seeds
print(f"Setting seed {seed} for data loader debugging")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Import after setting seed
from src.Dataset import MultiplicationDataset
from src.Utils import collate_fn
MultiplicationDataset.set_fixed_seed(seed)

# Create dataset with same params as our test
dataset = MultiplicationDataset(
    num_samples=5000,
    split='train',
    min_value=10**(int(sys.argv[3])-1),
    max_value=10**int(sys.argv[4])-1
)

# Create data loader with deterministic behavior
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=int(sys.argv[5]),
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    generator=torch.Generator().manual_seed(seed),
    worker_init_fn=lambda worker_id: random.seed(seed + worker_id)
)

# Collect first few batches to verify ordering
batches = []
for i, batch in enumerate(loader):
    if i >= 50:  # Just sample first 50 batches
        break
    # Store just the input tensors to save space
    batches.append((i, [b[0].clone().detach() for b in zip(*batch)]))

# Save sample data
torch.save(batches, f"debug_data_{run_name}.pt")
print(f"Saved {len(batches)} batches to debug_data_{run_name}.pt")
EOF

# Run the data debugging script for initial training
echo "Saving initial data ordering pattern..."
python src/debug_dataloading.py $SEED "initial" $MIN_DIGITS $MAX_DIGITS $BATCH_SIZE

# Train a small model for initial steps
echo "Training small model for $MAX_STEPS_INITIAL steps with checkpoints every $SAVE_FREQ steps..."
python main.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-latent $NUM_LATENT \
    --min-digits $MIN_DIGITS \
    --max-digits $MAX_DIGITS \
    --batch-size $BATCH_SIZE \
    --max-steps $MAX_STEPS_INITIAL \
    --save-every $SAVE_FREQ \
    --seed $SEED

# Check if the first training was successful
if [ $? -ne 0 ]; then
    echo "First training failed."
    exit 1
fi

# Get the latest run ID
latest_run_id=$(python scripts/list_runs.py --limit 1 | grep -oP '\d{8}-\d{6}_d\d+_l\d+_n\d+' | head -1)

# Check if we got a valid run ID
if [ -z "$latest_run_id" ]; then
    echo "No valid run ID found."
    exit 1
fi

echo "Got latest run ID: $latest_run_id"

# Run the data debugging script for resumed training
echo "Saving resumed data ordering pattern..."
python src/debug_dataloading.py $SEED "resume" $MIN_DIGITS $MAX_DIGITS $BATCH_SIZE

# Compare data patterns
echo "Comparing data patterns between initial and resumed runs..."
cat > src/compare_data_patterns.py << 'EOF'
import torch
import numpy as np

# Load data patterns
initial_data = torch.load("debug_data_initial.pt")
resume_data = torch.load("debug_data_resume.pt")

print(f"Initial data has {len(initial_data)} batches")
print(f"Resume data has {len(resume_data)} batches")

# Compare first few batches
num_compare = min(len(initial_data), len(resume_data), 10)
matches = 0

for i in range(num_compare):
    initial_idx, initial_batch = initial_data[i]
    resume_idx, resume_batch = resume_data[i]
    
    # Check if batches match
    batch_matches = True
    for j in range(len(initial_batch)):
        if not torch.all(initial_batch[j] == resume_batch[j]):
            batch_matches = False
            break
    
    if batch_matches:
        matches += 1
        print(f"Batch {i}: MATCH ✓")
    else:
        print(f"Batch {i}: DIFFERENT ✗")

print(f"Match rate: {matches}/{num_compare} ({matches/num_compare*100:.1f}%)")

# Conclusion
if matches == num_compare:
    print("PERFECT MATCH - Data ordering is consistent between runs")
else:
    print("INCONSISTENT - Data ordering differs between runs")
EOF

python src/compare_data_patterns.py

# Create special resume tracking script
cat > src/resume_tracker.py << 'EOF'
import torch
import os
import random
import time
import numpy as np
import logging
import argparse
import sys

# Parse arguments
parser = argparse.ArgumentParser(description="Track model states during resuming")
parser.add_argument("--run-id", required=True, help="Run ID to resume from")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResumeTracker")

# Set seeds for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set all seeds to {seed}")

set_all_seeds(args.seed)

# Get current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Get run info
from src.RunManagement import get_run_info
run_info = get_run_info(args.run_id)
if not run_info:
    logger.error(f"Run ID {args.run_id} not found")
    sys.exit(1)

logger.info(f"Found run: {run_info['id']}")

# Load checkpoints
if "checkpoint_paths" not in run_info:
    logger.error("No checkpoint paths found in run info")
    sys.exit(1)

checkpoint_paths = run_info["checkpoint_paths"]
simple_checkpoint_path = checkpoint_paths.get("simple_latest", "")
latent_checkpoint_path = checkpoint_paths.get("latent_latest", "")

logger.info(f"Loading checkpoints:")
logger.info(f"  SimpleTransformer: {simple_checkpoint_path}")
logger.info(f"  LatentTransformer: {latent_checkpoint_path}")

# Load checkpoint data
simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

# Display model steps
simple_step = simple_checkpoint.get('step', 0)
latent_step = latent_checkpoint.get('step', 0)
logger.info(f"Model steps: SimpleTransformer: {simple_step}, LatentTransformer: {latent_step}")

# Check seed information
simple_seed = simple_checkpoint.get('seed', simple_checkpoint.get('config', {}).get('seed', 'Not found'))
latent_seed = latent_checkpoint.get('seed', latent_checkpoint.get('config', {}).get('seed', 'Not found'))
logger.info(f"Seeds: SimpleTransformer: {simple_seed}, LatentTransformer: {latent_seed}")

# Extract and save first batch of parameters for comparison
def extract_first_params(checkpoint):
    params = []
    for i, (name, param) in enumerate(checkpoint['model_state_dict'].items()):
        if i < 20 and isinstance(param, torch.Tensor):  # Get first 20 parameters
            params.append((name, param.float().mean().item(), param.float().std().item()))
    return params

simple_params = extract_first_params(simple_checkpoint)
latent_params = extract_first_params(latent_checkpoint)

logger.info(f"Extracted {len(simple_params)} SimpleTransformer parameters")
logger.info(f"Extracted {len(latent_params)} LatentTransformer parameters")

# Save for comparison after resume
torch.save({
    'simple_params': simple_params,
    'latent_params': latent_params,
    'simple_step': simple_step,
    'latent_step': latent_step
}, "pre_resume_state.pt")

logger.info("Saved pre-resume state to pre_resume_state.pt")
EOF

# Track pre-resume state
echo "Tracking pre-resume model state..."
python src/resume_tracker.py --run-id $latest_run_id --seed $SEED

# Resume training with the same configuration and seed
echo "Resuming training for additional steps to reach $MAX_STEPS_RESUME total..."
python main.py \
    --resume \
    --run-id $latest_run_id \
    --max-steps $MAX_STEPS_RESUME \
    --save-every $SAVE_FREQ \
    --seed $SEED

# Check if the resume was successful
if [ $? -ne 0 ]; then
    echo "Resume training failed."
    exit 1
fi

# Create script to check model state after resuming
cat > src/check_resumed_state.py << 'EOF'
import torch
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResumeStateChecker")

# Load pre-resume state
logger.info("Loading pre-resume state...")
pre_state = torch.load("pre_resume_state.pt")

# Get most recent run ID
logger.info("Getting latest run...")
from src.RunManagement import list_runs
runs = list_runs(limit=1)
if not runs:
    logger.error("No runs found")
    exit(1)

latest_run = runs[0]
logger.info(f"Latest run: {latest_run['id']}")

# Load post-resume checkpoints
logger.info("Loading post-resume checkpoints...")
checkpoint_paths = latest_run.get("checkpoint_paths", {})
simple_checkpoint_path = checkpoint_paths.get("simple_latest", "")
latent_checkpoint_path = checkpoint_paths.get("latent_latest", "")

# Load checkpoint data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simple_checkpoint = torch.load(simple_checkpoint_path, map_location=device)
latent_checkpoint = torch.load(latent_checkpoint_path, map_location=device)

# Display resume steps
simple_step = simple_checkpoint.get('step', 0)
latent_step = latent_checkpoint.get('step', 0)
logger.info(f"Post-resume steps: SimpleTransformer: {simple_step}, LatentTransformer: {latent_step}")
logger.info(f"Pre-resume steps: SimpleTransformer: {pre_state['simple_step']}, LatentTransformer: {pre_state['latent_step']}")

# Step increases should match
simple_step_diff = simple_step - pre_state['simple_step']
latent_step_diff = latent_step - pre_state['latent_step']
logger.info(f"Step increases: SimpleTransformer: +{simple_step_diff}, LatentTransformer: +{latent_step_diff}")

# Extract current parameters to compare
def extract_first_params(checkpoint):
    params = []
    for i, (name, param) in enumerate(checkpoint['model_state_dict'].items()):
        if i < 20 and isinstance(param, torch.Tensor):  # Get first 20 parameters
            params.append((name, param.float().mean().item(), param.float().std().item()))
    return params

# Get post-resume parameters
post_simple_params = extract_first_params(simple_checkpoint)
post_latent_params = extract_first_params(latent_checkpoint)

# Compare parameters
def compare_params(pre_params, post_params, model_name):
    logger.info(f"Comparing {model_name} parameters...")
    
    # Check if we have the same parameters
    pre_names = [p[0] for p in pre_params]
    post_names = [p[0] for p in post_params]
    
    if pre_names != post_names:
        logger.warning(f"{model_name} parameter names don't match")
        return False
    
    # Compare values
    changes = []
    for i, ((name, pre_mean, pre_std), (_, post_mean, post_std)) in enumerate(zip(pre_params, post_params)):
        mean_diff = abs(post_mean - pre_mean)
        mean_pct = 100 * mean_diff / (abs(pre_mean) + 1e-10)
        
        if mean_pct > 5.0:  # More than 5% change
            changes.append((name, pre_mean, post_mean, mean_pct))
    
    if changes:
        logger.info(f"Found {len(changes)} parameters with significant changes (>5%)")
        for name, pre, post, pct in changes[:5]:  # Show first 5
            logger.info(f"  {name}: {pre:.6f} -> {post:.6f} ({pct:.1f}% change)")
        return True
    else:
        logger.warning(f"No significant parameter changes found in {model_name}")
        return False

# Check both models
simple_changed = compare_params(pre_state['simple_params'], post_simple_params, "SimpleTransformer")
latent_changed = compare_params(pre_state['latent_params'], post_latent_params, "LatentTransformer")

# Final verdict
if simple_changed and latent_changed:
    logger.info("✅ RESUME WORKING: Both models show parameter changes after resuming")
elif simple_changed or latent_changed:
    logger.warning("⚠️ PARTIAL RESUME: Only one model shows parameter changes")
else:
    logger.error("❌ RESUME FAILED: No parameter changes detected after resuming")
EOF

# Check post-resume state
echo "Checking post-resume model state..."
python src/check_resumed_state.py

# Clean up debug files
echo "Cleaning up debug files..."
rm -f src/debug_dataloading.py
rm -f src/compare_data_patterns.py
rm -f src/resume_tracker.py
rm -f src/check_resumed_state.py
rm -f debug_data_*.pt
rm -f pre_resume_state.pt

echo "Resume testing complete!" 