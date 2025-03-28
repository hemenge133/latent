"""
Custom SummaryWriter to preserve TensorBoard logs across runs
"""
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

logger = logging.getLogger(__name__)

class SummaryWriter(TorchSummaryWriter):
    """
    Custom SummaryWriter that preserves step counters between runs and ensures smooth continuity
    in TensorBoard graphs when resuming training from checkpoints.
    """

    def __init__(self, log_dir=None, *args, **kwargs):
        """Initialize with preserved step counters"""
        super().__init__(log_dir=log_dir, *args, **kwargs)

        # Track highest step seen for each tag
        self.max_steps = {}
        
        # Track the last value seen for each tag to ensure continuity
        self.last_values = {}

        # Path for storing persistence data
        if log_dir:
            self.persistence_file = os.path.join(log_dir, ".steps_data")
            self.values_file = os.path.join(log_dir, ".values_data")
            self._load_persistence()
            
            # Log when we initialize the writer
            logger.info(f"Initialized SummaryWriter with log_dir={log_dir}")
            if self.max_steps:
                logger.info(f"Loaded persisted step data: {self.max_steps}")

    def _load_persistence(self):
        """Load persisted step data if available"""
        # Load step data
        if hasattr(self, "persistence_file") and os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    for line in f:
                        if line.strip():
                            tag, step = line.strip().split(":", 1)
                            self.max_steps[tag] = int(step)
                logger.info(f"Loaded TensorBoard step data from {self.persistence_file}")
            except Exception as e:
                logger.error(f"Error loading step persistence data: {e}")
        
        # Load value data
        if hasattr(self, "values_file") and os.path.exists(self.values_file):
            try:
                with open(self.values_file, "r") as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(":", 1)
                            if len(parts) == 2:
                                tag, value = parts
                                try:
                                    self.last_values[tag] = float(value)
                                except ValueError:
                                    # Skip if value can't be converted to float
                                    pass
                logger.info(f"Loaded TensorBoard value data from {self.values_file}")
            except Exception as e:
                logger.error(f"Error loading value persistence data: {e}")

    def _save_persistence(self):
        """Save persisted step and value data"""
        # Save step data
        if hasattr(self, "persistence_file"):
            try:
                with open(self.persistence_file, "w") as f:
                    for tag, step in self.max_steps.items():
                        f.write(f"{tag}:{step}\n")
            except Exception as e:
                logger.error(f"Error saving step persistence data: {e}")
        
        # Save value data
        if hasattr(self, "values_file"):
            try:
                with open(self.values_file, "w") as f:
                    for tag, value in self.last_values.items():
                        f.write(f"{tag}:{value}\n")
            except Exception as e:
                logger.error(f"Error saving value persistence data: {e}")

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        """
        Add scalar with adjusted global_step to preserve continuity.
        
        When resuming training, this ensures:
        1. The global step continues from where it left off
        2. There's a smooth transition between runs by recording the last value
        3. Values close to zero are handled properly to avoid rounding issues
        """
        # Skip if global_step is None
        if global_step is None:
            return super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        
        # Avoid extremely small negative values which show as -0 in TensorBoard
        if isinstance(scalar_value, float) and -1e-7 < scalar_value < 0:
            scalar_value = 0.0
        
        # Store the actual value (not placeholder)
        if scalar_value is not None and (tag not in self.last_values or scalar_value != 0):
            self.last_values[tag] = scalar_value
        
        # First, handle step adjustment to ensure continuity
        adjusted_step = global_step
        
        # If we've seen this tag before, check for step continuity
        if tag in self.max_steps:
            max_seen_step = self.max_steps[tag]
            
            # Case 1: This is likely a training resumption (step reset or much lower than max)
            if global_step < max_seen_step - 5:
                # Get the next sequential step number
                adjusted_step = max_seen_step + 1
                
                # If this is the resumption point, add a bridging point using the last known value
                # This creates a visual continuity in the graph
                if tag in self.last_values and global_step <= 1:
                    # Bridge the gap with the last known value at max_seen_step
                    last_value = self.last_values[tag]
                    super().add_scalar(tag, last_value, max_seen_step, *args, **kwargs)
            
            # Case 2: Current step is same as max (duplicate log)
            elif global_step == max_seen_step:
                # Keep using the same step
                adjusted_step = global_step
            
            # Case 3: Normal progression (global_step > max_seen_step)
            else:
                # Use global_step directly
                adjusted_step = global_step
        
        # Update the max step for this tag
        self.max_steps[tag] = max(self.max_steps.get(tag, 0), adjusted_step)
        
        # Log significant step adjustments
        if global_step != adjusted_step and abs(global_step - adjusted_step) > 2:
            logger.debug(f"Adjusted step for {tag}: {global_step} -> {adjusted_step}")
        
        # Save persistence data every 25 steps to avoid excessive disk I/O
        if adjusted_step % 25 == 0:
            self._save_persistence()
        
        # Call parent implementation with adjusted step
        return super().add_scalar(tag, scalar_value, adjusted_step, *args, **kwargs)
    
    def close(self):
        """Save persistence data and close the writer"""
        self._save_persistence()
        super().close()
