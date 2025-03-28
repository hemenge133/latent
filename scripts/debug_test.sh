#!/bin/bash

# Change to the project directory
cd /home/h/Devel/latent

# Activate the Python 3 virtual environment if it exists
if [ -f ~/.virtualenvs/latent/bin/activate ]; then
    echo "Activating virtual environment"
    source ~/.virtualenvs/latent/bin/activate
fi

# Get Python version
echo "Using Python: $(python3 --version)"

# Run with minimal parameters to reproduce and debug the error
echo "=== Running minimal test with Python 3 ==="
python3 main.py --d-model 32 --num-layers 1 --num-latent 1 --min-digits 1 --max-digits 1 --batch-size 8 --max-steps 5

echo "=== Test complete ===" 