#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for managing runs and checkpoints.
"""
import os
import json
import glob
import torch
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

RUNS_DIRECTORY = "runs/parallel_comparison"
RUNS_INDEX_FILE = "runs/run_index.json"
CHECKPOINTS_DIRECTORY = "checkpoints"


def initialize_run_index():
    """Initialize or load the run index file"""
    os.makedirs("runs", exist_ok=True)

    if not os.path.exists(RUNS_INDEX_FILE):
        # Create empty index
        run_index = {}
        save_run_index(run_index)
        return run_index

    try:
        with open(RUNS_INDEX_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading run index: {e}")
        logger.warning("Creating new run index")
        run_index = {}
        save_run_index(run_index)
        return run_index


def save_run_index(run_index):
    """Save the run index to file"""
    try:
        with open(RUNS_INDEX_FILE, "w") as f:
            json.dump(run_index, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving run index: {e}")


def register_run(run_id, config):
    """Register a run in the index"""
    run_index = initialize_run_index()

    # Store a timestamp with the run
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_info = {
        "timestamp": timestamp,
        "config": config,
        "checkpoint_paths": {
            "simple_latest": f"checkpoints/simpletransformer/simpletransformer_latest.pt",
            "simple_best": f"checkpoints/simpletransformer/simpletransformer_best.pt",
            "latent_latest": f"checkpoints/latenttransformer/latenttransformer_latest.pt",
            "latent_best": f"checkpoints/latenttransformer/latenttransformer_best.pt",
        },
    }

    run_index[run_id] = run_info
    save_run_index(run_index)
    logger.info(f"Registered run: {run_id}")


def update_run_step(run_id, step):
    """Update the current step of a run"""
    run_index = initialize_run_index()

    if run_id not in run_index:
        logger.warning(f"Run ID {run_id} not found in index")
        return

    run_index[run_id]["current_step"] = step
    save_run_index(run_index)


def get_run_info(run_id=None):
    """
    Get information about a specific run.
    If run_id is None, return the most recent run.
    """
    run_index = initialize_run_index()

    if not run_index:
        logger.warning("No runs found in index")
        return None

    if run_id is None:
        # Find most recent run
        most_recent = None
        most_recent_timestamp = None

        for id, info in run_index.items():
            if most_recent is None or info.get("timestamp", "") > most_recent_timestamp:
                most_recent = id
                most_recent_timestamp = info.get("timestamp", "")

        run_id = most_recent

    if run_id not in run_index:
        # Try partial matching
        matching_ids = [id for id in run_index.keys() if run_id in id]
        if matching_ids:
            if len(matching_ids) > 1:
                logger.warning(
                    f"Multiple matching run IDs for '{run_id}': {matching_ids}"
                )
                logger.warning(f"Using the first match: {matching_ids[0]}")
            run_id = matching_ids[0]
        else:
            logger.warning(f"Run ID {run_id} not found in index")
            return None

    return {"id": run_id, **run_index[run_id]}


def scan_runs_directory():
    """Scan the runs directory and update the index"""
    run_index = initialize_run_index()

    # Scan the runs directory for run folders
    run_dirs = glob.glob(f"{RUNS_DIRECTORY}/*")

    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)

        # Skip if already indexed
        if run_id in run_index:
            continue

        # Check if there's a config.json file
        config_path = os.path.join(run_dir, "config.json")
        config = {}

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config for {run_id}: {e}")

        # Register the run
        register_run(run_id, config)

    return run_index


def list_runs():
    """List all available runs"""
    run_index = initialize_run_index()

    if not run_index:
        logger.info("No runs found in index")
        return []

    runs_info = []
    for run_id, info in run_index.items():
        # Extract useful info
        timestamp = info.get("timestamp", "Unknown")
        config = info.get("config", {})
        d_model = config.get("d_model", "Unknown")
        num_layers = config.get("num_layers", "Unknown")
        num_latent = config.get("num_latent", "Unknown")

        runs_info.append(
            {
                "id": run_id,
                "timestamp": timestamp,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_latent": num_latent,
            }
        )

    # Sort by timestamp, newest first
    runs_info.sort(key=lambda x: x["timestamp"], reverse=True)
    return runs_info


def get_run_checkpoints(run_id):
    """Get the checkpoint paths for a specific run"""
    run_info = get_run_info(run_id)

    if not run_info:
        return None

    return run_info.get("checkpoint_paths", {})


def backup_run_checkpoints(run_id):
    """Backup the checkpoints for a specific run"""
    run_info = get_run_info(run_id)

    if not run_info:
        logger.warning(f"Run ID {run_id} not found")
        return False

    # Create backup directory
    backup_dir = f"{CHECKPOINTS_DIRECTORY}/backup/{run_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup each checkpoint
    checkpoint_paths = run_info.get("checkpoint_paths", {})
    for name, path in checkpoint_paths.items():
        if os.path.exists(path):
            backup_path = os.path.join(backup_dir, os.path.basename(path))
            shutil.copy2(path, backup_path)
            logger.info(f"Backed up {name} to {backup_path}")

    return True


def get_run_config_from_id(run_id):
    """
    Extract configuration parameters from a run ID string.
    Example: "20250325-152939_d768_l8_n16" -> {"d_model": 768, "num_layers": 8, "num_latent": 16}
    """
    config = {}

    # First check if this run is in our index
    run_info = get_run_info(run_id)
    if run_info and "config" in run_info:
        return run_info["config"]

    # If not in index or no config, try to parse from the ID
    parts = run_id.split("_")

    for part in parts:
        if part.startswith("d") and part[1:].isdigit():
            config["d_model"] = int(part[1:])
        elif part.startswith("l") and part[1:].isdigit():
            config["num_layers"] = int(part[1:])
        elif part.startswith("n") and part[1:].isdigit():
            config["num_latent"] = int(part[1:])

    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run management utilities")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--info", type=str, help="Get info for a specific run ID")
    parser.add_argument(
        "--scan", action="store_true", help="Scan runs directory and update index"
    )
    parser.add_argument(
        "--backup", type=str, help="Backup checkpoints for a specific run ID"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.list:
        runs = list_runs()
        print(f"Found {len(runs)} runs:")
        for run in runs:
            print(
                f"{run['id']} - {run['timestamp']} - d{run['d_model']}_l{run['num_layers']}_n{run['num_latent']}"
            )

    if args.info:
        run_info = get_run_info(args.info)
        if run_info:
            print(f"Run ID: {run_info['id']}")
            print(f"Timestamp: {run_info.get('timestamp', 'Unknown')}")
            print(f"Config: {json.dumps(run_info.get('config', {}), indent=2)}")
            print(
                f"Checkpoint paths: {json.dumps(run_info.get('checkpoint_paths', {}), indent=2)}"
            )
        else:
            print(f"Run ID {args.info} not found")

    if args.scan:
        print("Scanning runs directory...")
        run_index = scan_runs_directory()
        print(f"Updated index with {len(run_index)} runs")

    if args.backup:
        print(f"Backing up checkpoints for run {args.backup}...")
        if backup_run_checkpoints(args.backup):
            print("Backup completed")
        else:
            print("Backup failed")
