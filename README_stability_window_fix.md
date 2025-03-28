# Stability Window Fix

## Issue Description
The training loop was encountering errors related to a missing `stability_window` key in the model dictionary:
```
Error in training step for SimpleTransformer: 'stability_window'
Error in training step for LatentTransformer: 'stability_window'
```

The code was trying to access `model_info["stability_window"]` in the loop that monitors training stability, but this key was never defined in the model information dictionaries.

## Fix Applied
Added the `stability_window` parameter to both model dictionaries in the `models` initialization in `src/TrainingLoop.py`:

```python
models = {
    'simple': {
        # ... existing keys ...
        'stability_window': 50  # Added stability window parameter
    },
    'latent': {
        # ... existing keys ...
        'stability_window': 50  # Added stability window parameter
    }
}
```

This parameter is used to limit the number of recent loss values stored for stability monitoring, preventing unbounded growth of the `recent_losses` list.

## Testing
Two test scripts were created to verify the fix:

1. `scripts/test_stability_window.sh` - A simple test that runs training for 10 steps to verify no `stability_window` errors occur
2. `scripts/test_resume_stability.sh` - A more comprehensive test that runs initial training for 5 steps, then resumes for 5 more steps (total 10), verifying that the fix works with resumed training as well

Both tests completed successfully without any `stability_window` errors.

## Code Impact
The fix is minimal and non-intrusive, simply adding the missing configuration parameter that was already being used elsewhere in the code. This ensures the training loop can properly manage its list of recent loss values for stability monitoring without encountering KeyError exceptions. 