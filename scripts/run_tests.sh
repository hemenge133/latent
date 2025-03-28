#!/bin/bash
# Run all tests with coverage reporting

# Ensure the script fails on errors
set -e

# Activate the virtual environment if it exists
if [ -d "latent" ]; then
    echo "Activating virtual environment..."
    source latent/bin/activate
fi

# Install test dependencies if needed
echo "Checking for test dependencies..."
pip install pytest pytest-cov

# Run pytest with coverage
echo "Running tests with coverage..."
pytest tests/ -v --cov=src --cov-report=term-missing

# Exit with the pytest status code
exit $? 