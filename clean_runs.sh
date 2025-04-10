#!/bin/bash
# Remove tensorboard run data
rm -rf ./runs/parallel_comparison/*
rm -rf ./runs/test_tensorboard/*
rm -rf ./runs/test*
rm -rf ./runs/standard_run/*
rm -rf ./runs/grid_search/*
rm -rf ./runs/optuna_study_*/*

# Remove checkpoint files
rm -rf ./checkpoints/latenttransformer/*
rm -rf ./checkpoints/simpletransformer/*

# Remove log files
rm -f ./logs/*.log
rm -f ./*.log
rm -f ./test_*.log

# Remove Optuna databases and best parameter files
rm -f ./*.db
rm -f ./best_params_*.json

# Remove grid search results
rm -f ./grid_search_results_*.json

echo "All run data, checkpoints, logs, and optimization artifacts have been cleaned."
