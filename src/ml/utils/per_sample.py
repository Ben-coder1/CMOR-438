import numpy as np
from typing import Callable, Sequence

from ml.utils._errors_and_warnings._general_error_handling import InputShapeError
from ml.utils._errors_and_warnings._general_error_handling import _ensure_callable, _ensure_array_like


def _apply_per_sample(func: Callable, *arrays: Sequence) -> np.ndarray:
    """
    Apply a function per sample, with support for both vectorized
    and scalar-only functions.

    Parameters
    ----------
    func : callable
        A function that can operate either on full arrays or on
        individual samples.
    arrays : array-like
        One or more arrays of shape (n_samples, ...). Each must have
        the same leading dimension.

    Returns
    -------
    np.ndarray
        Results per sample, shape (n_samples,).
    """
    # Ensure inputs are numpy arrays
    arrays = [np.asarray(a) for a in arrays]
    
    # Validation: Ensure all inputs have the same number of samples
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("All inputs must have the same number of samples.")

    # 1. Try Vectorized Call (Fast Path)
    try:
        result = np.asarray(func(*arrays))
        # Check if the function returned a valid array of the correct length.
        # We perform strict checking to ensure we didn't get a scalar or wrong shape
        # by accident (e.g., if n=1 and function returned a scalar).
        if result.shape[0] == n:
            return result
    except Exception:
        # If vectorized call fails (AttributeError, ValueError, or Shape mismatch),
        # fall back to row-wise.
        pass

    # 2. Row-wise Fallback (Slow Path)
    # FIX: Removed `dtype=float`. Let NumPy infer the data type automatically.
    # This supports integers, strings, objects, etc.
    return np.array([func(*[a[i] for a in arrays]) for i in range(n)])