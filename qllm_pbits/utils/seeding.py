"""
Utilities for reproducible random number generation.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Set seeds for reproducibility across all random number generators.
    
    Args:
        seed: Random seed (integer)
        deterministic: If True, enable torch.use_deterministic_algorithms.
            WARNING: This may cause errors with some CUDA operations.
            Use only when full reproducibility is critical.
    
    Examples:
        >>> set_seed(42)
        >>> # All random operations are now deterministic
        >>> torch.randn(3)
        tensor([ 0.3367,  0.1288,  0.2345])
    
    Notes:
        - Deterministic mode may fail with certain CUDA operations
        - Some operations (like multinomial on CUDA) may still have variance
        - For true reproducibility, also control threading and device selection
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Note: This may cause runtime errors with operations like
        # torch.nn.functional.interpolate, scatter_add_, etc.

