#!/usr/bin/env python
"""
Script to list available runs and their configuration.
"""
import os
import sys
import json
import datetime
import argparse
from tabulate import tabulate

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.RunManagement import list_runs, get_run_info, scan_runs_directory


def format_config(config):
    """Format configuration for display"""
    if not config:
        return "N/A"

    # Select key parameters to display
    key_params = [
        "d_model",
        "num_layers",
        "num_latent",
        "min_digits",
        "max_digits",
        "batch_size",
    ]
    return ", ".join([f"{k}={config.get(k, 'N/A')}" for k in key_params if k in config])


def main():
    parser = argparse.ArgumentParser(description="List available training runs")
    parser.add_argument("--rescan", action="store_true", help="Rescan runs directory")
    parser.add_argument("--id", type=str, help="Show details for specific run ID")
    parser.add_argument(
        "--sort",
        type=str,
        default="date",
        choices=["date", "step", "loss"],
        help="Sort method: date, step, or loss",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit number of runs to show"
    )
    args = parser.parse_args()

    # Rescan runs directory if requested
    if args.rescan:
        print("Rescanning runs directory...")
        scan_runs_directory()

    # Show details for specific run ID
    if args.id:
        run_info = get_run_info(args.id)
        if not run_info:
            print(f"Run with ID {args.id} not found.")
            sys.exit(1)

        print(f"\nDetails for run {run_info['id']}:")
        print(f"Date: {run_info.get('date', 'N/A')}")
        print(f"Current step: {run_info.get('current_step', 'N/A')}")

        # Print checkpoint paths
        if "checkpoint_paths" in run_info:
            print("\nCheckpoint paths:")
            for name, path in run_info["checkpoint_paths"].items():
                print(f"  {name}: {path}")

        # Print configuration
        if "config" in run_info:
            print("\nConfiguration:")
            for key, value in run_info["config"].items():
                print(f"  {key}: {value}")

        # Print run directory
        if "directory" in run_info:
            print(f"\nRun directory: {run_info['directory']}")

        sys.exit(0)

    # Get all runs
    runs = list_runs()

    # Sort runs
    if args.sort == "date":
        # Default is already sorted by date
        pass
    elif args.sort == "step":
        runs = sorted(runs, key=lambda x: x.get("current_step", 0), reverse=True)
    elif args.sort == "loss":
        # Sort by validation loss if available
        runs = sorted(runs, key=lambda x: x.get("val_loss", float("inf")))

    # Limit number of runs
    if args.limit > 0:
        runs = runs[: args.limit]

    # Prepare table data
    table_data = []
    for run in runs:
        row = [
            run.get("id", "N/A"),
            run.get("date", "N/A"),
            run.get("current_step", "N/A"),
            format_config(run.get("config", {})),
            run.get("val_loss", "N/A"),
        ]
        table_data.append(row)

    # Display runs in table format
    headers = ["Run ID", "Date", "Current Step", "Configuration", "Val Loss"]

    if table_data:
        print("\nAvailable runs:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(
            f"\nShowing {len(table_data)} runs. Use --limit to change or --id <run_id> for details."
        )
    else:
        print("No runs found.")


if __name__ == "__main__":
    main()
