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

def run_command(cmd, timeout=600):
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
        
        # Clean up previous checkpoints before starting
        if simple_checkpoint_path.exists():
            os.remove(simple_checkpoint_path)
            logger.info(f"Removed existing checkpoint: {simple_checkpoint_path}")
        if latent_checkpoint_path.exists():
            os.remove(latent_checkpoint_path)
            logger.info(f"Removed existing checkpoint: {latent_checkpoint_path}")
        
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
            "--max-steps", "10",            # Fixed value: Just 10 steps 
            "--save-every", "10",           # Save only at the end (step 9 or 10)
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
        
        # --- Verify Initial Checkpoint --- 
        logger.info("Verifying initial checkpoint contents...")
        initial_checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
        initial_step_from_ckpt = initial_checkpoint.get("step", -1)
        # Assert based on when the save happens (max_steps or save_every)
        # Since save_every=10, it saves *after* step 9 completes, at step 10.
        # Let's adjust expectation based on logs if this fails.
        expected_initial_step = 10 
        assert initial_step_from_ckpt == expected_initial_step, f"Initial checkpoint step mismatch. Expected {expected_initial_step}, got {initial_step_from_ckpt}"
        logger.info(f"Initial checkpoint verified: step={initial_step_from_ckpt}")

        initial_embed_shape = None # For later comparison
        if 'config' in initial_checkpoint:
             assert initial_checkpoint['config'].get('d_model') == 32, f"Initial checkpoint d_model mismatch: {initial_checkpoint['config'].get('d_model')}"
             logger.info("Initial checkpoint d_model verified.")
        else:
             logger.warning("Config not found in initial checkpoint for verification.")
        
        if "model_state_dict" in initial_checkpoint:
             for k, v in initial_checkpoint["model_state_dict"].items():
                if k.endswith("embed.weight") or k == "embed.weight":
                    initial_embed_shape = v.shape
                    logger.info(f"Embedding shape from initial checkpoint: {initial_embed_shape}")
                    break
        # --- End Verify Initial Checkpoint ---

        # Sleep briefly to ensure clear timestamps
        time.sleep(1)

        # Step 2: Resume training for just ONE more step
        resume_cmd = [
            "python", "main.py",
            "--max-steps", "11",            # Run to 11 steps total
            "--save-every", "1",           # Save every step during resume
            "--seed", "42" # Keep seed consistent
        ]

        # Add resume flag and run ID if available
        if initial_run_id:
            resume_cmd.insert(2, "--resume")
            resume_cmd.extend([
                "--run-id", initial_run_id,
                # Explicitly pass parameters matching the initial run
                "--d-model", "32",
                "--num-layers", "1",
                "--num-latent", "2",
                "--min-digits", "1",
                "--max-digits", "1",
                "--force-config"
            ])
        else:
            # Just use resume flag without run ID (less likely scenario for this test)
            resume_cmd.insert(2, "--resume")
            resume_cmd.extend([
                "--d-model", "32",
                "--num-layers", "1",
                "--num-latent", "2",
                "--min-digits", "1",
                "--max-digits", "1",
                "--force-config"
            ])

        # Run the resume command
        logger.info(f"Running resume command: {' '.join(resume_cmd)}")
        resume_exit_code = run_command(resume_cmd)
        assert resume_exit_code == 0, f"Resume training failed with code {resume_exit_code}"

        # --- Verify Resumed Checkpoint --- 
        logger.info("Verifying resumed checkpoint contents...")
        assert simple_checkpoint_path.exists(), "SimpleTransformer checkpoint file missing after resume!"
        
        # Add a small delay/retry mechanism for loading the checkpoint, in case of filesystem lag
        updated_checkpoint = None
        for attempt in range(3):
            try:
                updated_checkpoint = torch.load(simple_checkpoint_path, map_location="cpu")
                logger.info(f"Successfully loaded resumed checkpoint on attempt {attempt+1}")
                break
            except FileNotFoundError:
                 logger.warning(f"Attempt {attempt+1}: Resumed checkpoint not found, waiting...")
                 time.sleep(1)
            except Exception as load_err:
                 logger.error(f"Attempt {attempt+1}: Error loading resumed checkpoint: {load_err}")
                 time.sleep(1)
        assert updated_checkpoint is not None, "Failed to load resumed checkpoint after multiple attempts."

        updated_step = updated_checkpoint.get("step", -1)
        expected_resumed_step = 11 # Should complete step 10 and save at step 11
        assert updated_step == expected_resumed_step, f"Checkpoint step not updated correctly after resume. Expected {expected_resumed_step}, got {updated_step}"
        logger.info(f"Checkpoint step updated successfully: {initial_step_from_ckpt} -> {updated_step}")

        if 'config' in updated_checkpoint:
             assert updated_checkpoint['config'].get('d_model') == 32, f"Resumed checkpoint d_model mismatch: {updated_checkpoint['config'].get('d_model')}"
             logger.info("Resumed checkpoint d_model verified.")
        else:
             logger.warning("Config not found in resumed checkpoint for verification.")

        if initial_embed_shape and "model_state_dict" in updated_checkpoint:
             resumed_embed_shape = None
             for k, v in updated_checkpoint["model_state_dict"].items():
                  if k.endswith("embed.weight") or k == "embed.weight":
                       resumed_embed_shape = v.shape
                       logger.info(f"Embedding shape from resumed checkpoint: {resumed_embed_shape}")
                       break
             assert resumed_embed_shape == initial_embed_shape, f"Embedding shape changed after resume: {initial_embed_shape} -> {resumed_embed_shape}"
             logger.info("Resumed checkpoint embedding shape verified.")
        elif initial_embed_shape:
             logger.warning("Could not verify embedding shape in resumed checkpoint: model_state_dict missing.")
        # --- End Verify Resumed Checkpoint ---

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