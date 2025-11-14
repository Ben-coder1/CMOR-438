import numpy as np
from typing import Callable, Sequence

from ml.utils._errors_and_warnings._general_error_handling import InputShapeError
from ml.utils._errors_and_warnings._general_error_handling import _ensure_callable, _ensure_array_like


import numpy as np
from typing import Callable, Sequence

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
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("All inputs must have the same number of samples.")

    try:
        # Try vectorized call
        result = np.asarray(func(*arrays))
        if result.shape[0] == n:
            return result
    except Exception:
        pass

    # Row-wise fallback
    return np.array([func(*[a[i] for a in arrays]) for i in range(n)], dtype=float)
