#!/usr/bin/env python
"""
Wrapper script to call the CalculateEfficiency module from src directory
"""
from src.CalculateEfficiency import (
    clean_compiled_state_dict,
    get_best_checkpoint,
    calculate_efficiency,
    main,
)

if __name__ == "__main__":
    main()
