# Latent Transformer Project Planning Document

## Bug Fixes

### Checkpoint Resume Test Fixes (2025-03-28)

1. **Command Order Bug**:
   - Issue: The test was incorrectly placing the `--resume` flag before the script name, resulting in `python --resume main.py` instead of `python main.py --resume`.
   - Fix: Modified the command insertion order in `tests/test_checkpoint_resume_simple.py` to insert the `--resume` flag at index 2 instead of index 1.

2. **Incrementing Max Steps Bug**:
   - Issue: The test was reading the step value from the previously created checkpoint and adding 10 to it, causing the max_steps to increase with each test run (resulting in values like 76 instead of the intended 10-20).
   - Fix: Modified the test to use explicit fixed values (10 for the initial run, 20 for the resume run) rather than incrementing based on checkpoint values.

## Implementation Notes

- The checkpoint system stores training progress including step count, model parameters, and optimizer state.
- When resuming training, the system needs to properly restore all these components to enable seamless continuation.
- Tests verify that training can be resumed correctly with all model dimensions and parameters preserved.

## Usage Examples

- Example for quick testing: `python main.py --d-model 64 --num-layers 2 --num-latent 4 --min-digits 1 --max-digits 1 --batch-size 16 --max-steps 200 --save-every 10`
- Example for resuming training: `python main.py --resume --run-id [RUN_ID] --max-steps 400` 