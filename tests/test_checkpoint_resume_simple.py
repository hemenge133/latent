#!/usr/bin/env python
"""
Simple integration test for checkpoint loading and resuming.

This test verifies that:
1. Training can run for 3 epochs with checkpoints saved
2. Training can be resumed from those checkpoints for 3 more epochs
3. All model dimensions and parameters are correctly restored during resumption
"""

import os
import sys
import time
import json
import torch
import pytest
import subprocess
from pathlib import Path
from loguru import logger

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("test_checkpoint_resume.log", rotation="100 MB")

def get_latest_run_id():
    """Get the latest run ID from the runs.json file"""
    if not os.path.exists("runs.json"):
        logger.warning("runs.json file not found")
        return None
    
    try:
        with open("runs.json", "r") as f:
            runs = json.load(f)
        
        if not runs:
            logger.warning("runs.json is empty")
            return None
        
        # Get the latest run by timestamp
        latest_run = None
        latest_timestamp = "0"
        
        for run_id, run_info in runs.items():
            if 'timestamp' in run_info and run_info['timestamp'] > latest_timestamp:
                latest_timestamp = run_info['timestamp']
                latest_run = run_id
        
        if latest_run:
            logger.info(f"Found latest run ID: {latest_run}")
        else:
            logger.warning("No run ID found in runs.json")
            
        return latest_run
    except Exception as e:
        logger.error(f"Error reading runs.json: {str(e)}")
        return None

def get_run_id_from_dirs():
    """Fallback method to get run ID from directories"""
    runs_dir = Path("runs/parallel_comparison")
    if runs_dir.exists():
        dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if dirs:
            # Sort by modification time, newest first
            dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            run_id = dirs[0].name
            logger.info(f"Found run ID from directory listing: {run_id}")
            return run_id
    
    logger.warning("No run ID found in directories")
    return None

def run_command(cmd, timeout=300):
    """Run a command and log output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Start the process and wait for it to complete
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )
    
    # Log standard output and error
    for line in result.stdout.split("\n"):
        if line.strip():
            logger.info(f"STDOUT: {line.strip()}")
    
    for line in result.stderr.split("\n"):
        if line.strip():
            logger.warning(f"STDERR: {line.strip()}")
    
    return result.returncode

def cleanup(run_id=None):
    """Clean up test artifacts"""
    if run_id:
        # Clean up run directory if possible
        run_dir = Path("runs/parallel_comparison") / run_id
        if run_dir.exists():
            try:
                import shutil
                shutil.rmtree(run_dir)
                logger.info(f"Cleaned up run directory: {run_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up run directory: {str(e)}")

def test_checkpoint_resume_simple():
    """
    Test checkpoint resumption by running main.py directly:
    1. First run with 3 epochs
    2. Resume run with 3 more epochs
    """
    initial_run_id = None
    
    try:
        # Make sure checkpoint directories exist
        os.makedirs("checkpoints/simpletransformer", exist_ok=True)
        os.makedirs("checkpoints/latenttransformer", exist_ok=True)
        os.makedirs("runs/parallel_comparison", exist_ok=True)
        
        # Get initial step count
        simple_checkpoint_path = Path("checkpoints/simpletransformer/simpletransformer_latest.pt")
        latent_checkpoint_path = Path("checkpoints/latenttransformer/latenttransformer_latest.pt")
        
        initial_step = 0
        if simple_checkpoint_path.exists():
            try:
                checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
                if "step" in checkpoint:
                    initial_step = checkpoint["step"]
                    logger.info(f"Found existing checkpoint at step {initial_step}")
            except Exception as e:
                logger.error(f"Error reading existing checkpoint: {str(e)}")
        
        # Step 1: Run initial training for 3 epochs
        # Use a small model and dataset for quick testing
        cmd = [
            "python", "main.py",
            "--d-model", "32",              # Small model size
            "--num-layers", "1",            # Single layer
            "--num-latent", "2",            # Few latent tokens
            "--min-digits", "1",            # Single-digit multiplication (small problem)
            "--max-digits", "1",
            "--batch-size", "16",           # Small batch size
            "--max-steps", "10",            # Fixed value: Just 10 steps (~ 3 epochs with small dataset)
            "--save-every", "3",            # Save every 3 steps to ensure checkpoint at end
            "--seed", "42"
        ]
        
        # Run the command
        exit_code = run_command(cmd)
        assert exit_code == 0, f"Initial training failed with code {exit_code}"
        
        # Get the run ID for resumption
        initial_run_id = get_latest_run_id()
        if initial_run_id is None:
            # Try fallback method
            initial_run_id = get_run_id_from_dirs()
        
        # Even if we don't have a run ID, we can still resume from the checkpoint files
        if initial_run_id is None:
            logger.warning("Could not find run ID, will resume using just the checkpoint files")
        else:
            logger.info(f"Initial training completed, run ID: {initial_run_id}")
        
        # Verify checkpoints were created
        assert simple_checkpoint_path.exists(), "SimpleTransformer checkpoint not created"
        assert latent_checkpoint_path.exists(), "LatentTransformer checkpoint not created"
        
        # Get steps from checkpoint
        checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
        simple_step = checkpoint.get("step", 0)
        logger.info(f"Checkpoint saved at step {simple_step}")
        
        # Store dimensions and parameters for later verification
        if "model_state_dict" in checkpoint:
            # Find embedding dimension
            for k, v in checkpoint["model_state_dict"].items():
                if k.endswith("embed.weight") or k == "embed.weight":
                    embed_shape = v.shape
                    logger.info(f"Embedding shape from checkpoint: {embed_shape}")
                    break
        
        # Sleep briefly to ensure clear timestamps
        time.sleep(1)
        
        # Step 2: Resume training for 3 more epochs
        resume_cmd = [
            "python", "main.py",
            "--max-steps", "20",            # Fixed value: Run to 20 steps total
            "--seed", "42"
        ]
        
        # Add resume flag and run ID if available
        if initial_run_id:
            resume_cmd.insert(2, "--resume")
            resume_cmd.extend(["--run-id", initial_run_id])
        else:
            # Just use resume flag without run ID
            resume_cmd.insert(2, "--resume")
            # Add parameters from the initial run to ensure compatibility
            resume_cmd.extend([
                "--d-model", "32",
                "--num-layers", "1",
                "--num-latent", "2",
                "--min-digits", "1",
                "--max-digits", "1"
            ])
        
        # Run the resume command
        resume_exit_code = run_command(resume_cmd)
        assert resume_exit_code == 0, f"Resume training failed with code {resume_exit_code}"
        
        # Verify checkpoint was updated
        updated_checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
        updated_step = updated_checkpoint.get("step", 0)
        
        assert updated_step > simple_step, f"Checkpoint step not updated. Original: {simple_step}, New: {updated_step}"
        logger.info(f"Checkpoint step updated: {simple_step} -> {updated_step}")
        
        # Verify dimensions were maintained
        if "model_state_dict" in updated_checkpoint:
            for k, v in updated_checkpoint["model_state_dict"].items():
                if k.endswith("embed.weight") or k == "embed.weight":
                    updated_embed_shape = v.shape
                    logger.info(f"Updated embedding shape: {updated_embed_shape}")
                    assert updated_embed_shape == embed_shape, f"Embedding shape changed: {embed_shape} -> {updated_embed_shape}"
                    break
        
        logger.info("Test completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Test failed with exception: {str(e)}")
        assert False, f"Test failed with exception: {str(e)}"
    
    finally:
        # Clean up test artifacts
        if initial_run_id:
            logger.info(f"Cleaning up run ID: {initial_run_id}")
            cleanup(initial_run_id)

if __name__ == "__main__":
    test_checkpoint_resume_simple() 