#!/bin/bash
# Script to setup the environment and install dependencies

cd /home/h/Devel/latent

# Check if virtual environment exists
if [ ! -d ~/.virtualenvs/latent ]; then
    echo "Creating virtual environment..."
    mkdir -p ~/.virtualenvs
    python3 -m venv ~/.virtualenvs/latent
fi

# Activate virtual environment
source ~/.virtualenvs/latent/bin/activate

# Install required packages
echo "Installing dependencies..."
pip install torch numpy matplotlib tqdm tabulate tensorboard

# Create required directories
echo "Creating directory structure..."
mkdir -p checkpoints/simpletransformer
mkdir -p checkpoints/latenttransformer
mkdir -p runs/parallel_comparison

# Initialize run index if needed
python -c "
try:
    from run_management import initialize_run_index
    initialize_run_index()
    print('Run index initialized.')
except ImportError:
    print('Warning: run_management.py not found. Run index not initialized.')
"

echo "Environment setup complete. Use the following to activate:"
echo "source ~/.virtualenvs/latent/bin/activate" 