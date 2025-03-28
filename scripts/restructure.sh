#!/bin/bash
# Script to restructure the project, moving Python files to src/ and renaming them

cd /home/h/Devel/latent

# Ensure src directory exists
mkdir -p src

# Rename and move files to src with capitalized names
cp stable_comparison_with_accuracy.py src/Models.py
cp dataset.py src/Dataset.py
cp config.py src/Config.py
cp losses.py src/Losses.py
cp metrics.py src/Metrics.py
cp run_management.py src/RunManagement.py
cp training.py src/Training.py
cp training_loop.py src/TrainingLoop.py
cp utils.py src/Utils.py
cp calculate_efficiency.py src/CalculateEfficiency.py

echo "Project files have been restructured!"
echo "Original files remain in their current location."
echo "Update main.py and other scripts to use the new src directory structure." 