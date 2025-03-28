# Planning Document: Enhancements for Latent Transformer Project

## 1. Fix TensorBoard Integration and Checkpoint Persistence Issues

**Current Status:** ✅ Implemented
- Created improved SummaryWriter implementation to handle step continuity
- Added proper value persistence between runs
- Fixed negative value display issues in TensorBoard
- Implemented visual bridging points between runs

## 2. Develop Tests for Optimizer/Scheduler State Persistence

**Current Status:** ✅ Implemented
- Created test_checkpoint_resume.py to verify state persistence
- Added verification of optimizer momentum values
- Ensured scheduler state properly advances between runs

## 3. Increase Dataset Size to 1 Million 3-digit Multiplication Examples

**Current Status:** To be implemented
- Need to update dataset generation logic to handle 3-digit numbers (100-999)
- Optimize the MultiplicationDataset class for larger ranges
- Implement memory-efficient caching for large datasets

## 4. Identify Duplicated Code and Library Opportunities

**Current Status:** To be implemented
- Scan codebase for duplicate functionality
- Look for opportunities to use standard libraries

## 5. Fix Max Iteration Handling

**Current Status:** To be implemented
- The max_steps argument is being provided but not properly respected during training
- Need to modify training loop to:
  - Respect max_steps when provided (stop at that exact point)
  - Allow indefinite training if max_steps is not specified
- Update the progress tracking to correctly show progress relative to max_steps
- Ensure checkpoint resume logic correctly handles remaining steps based on max_steps

## 6. Common Issues and Critical Components

**Training Loop Issues:**
- Loss function (criterion) needs a fallback mechanism as it can be None in some execution paths
- The global_step variable must be derived from model-specific step counts
- Model-specific training state (teacher forcing parameters, stability windows) must be included in checkpoints
- Training loop breaks correctly only when both models reach max_steps

**Command-Line Usage:**
- Main.py requires specific model architecture parameters (d-model, num-layers, num-latent)
- Dataset parameters control problem complexity (min-digits, max-digits)
- Training parameters control batch size, max steps, and checkpointing frequency
- Example for quick testing: `python main.py --d-model 64 --num-layers 2 --num-latent 4 --min-digits 1 --max-digits 1 --batch-size 16 --max-steps 200 --save-every 10`

**SequenceAccuracyLoss:**
- Combines token-level cross-entropy with sequence-level correctness metrics
- Has parameters for balancing token vs sequence accuracy weights
- Properly handles masking for padding tokens
- Returns both loss and binary sequence accuracy for metrics tracking

**Checkpoint Persistence:**
- Must include optimizer and scheduler states for proper momentum continuation
- Must include teacher forcing parameters (schedule, current probability, start step)
- Must include step counts separately for each model (simple and latent)
- Must include recent loss history and stability windows
- When resuming from checkpoints, model dimensions must be checked for consistency
- Models should be recreated with the dimensions from checkpoints when mismatches are detected
- Dimension mismatch most commonly occurs with d_model, num_layers, and num_latent parameters

**Common Dimension Mismatch Issues:**
- When checkpoint has d_model=64 but attempting to create model with d_model=320 (default)
- When checkpoint has num_latent=4 but attempting to create model with num_latent=8 (default)
- The fix requires extracting dimensions from checkpoint state_dict before creating models
- If dimensions don't match after model creation, recreating the models with the correct dimensions

## 7. Test Run Cleanup

**Current Status:** To be implemented
- Test runs accumulate in the `runs/` directory and runs JSON, cluttering the visualization and tracking
- Need to develop a cleanup mechanism that:
  - Detects test runs (based on naming patterns or metadata)
  - Removes test runs from the filesystem (deleting TensorBoard logs)
  - Removes test runs from the runs JSON index
  - Implements a pruning strategy for old test runs
- Best practices:
  - Tests should use a specific prefix or tag to make them easily identifiable
  - Clean up test runs after each test suite execution
  - Implement an age-based cleanup for any remaining test runs

## Storage and Configuration Information

**Checkpoints:** Saved in `checkpoints/{model_name}/` directory
**TensorBoard logs:** Saved in `runs/parallel_comparison/{run_id}` with subdirectories for each model
**Metrics persistence:** `.steps_data` and `.values_data` files in the TensorBoard log directories 