import numpy as np
from typing import Callable, Sequence

from ml.utils._errors_and_warnings.error_handling import InputShapeError
from ml.utils._errors_and_warnings.error_handling import _ensure_callable, _ensure_array_like


import numpy as np
from typing import Callable, Sequence

def _call_distance_for_samples(distance: Callable, y_true: Sequence, y_pred: Sequence) -> np.ndarray:
    """
    Apply a distance function per sample, with support for both vectorized
    and scalar-only distance functions.

    Parameters
    ----------
    distance : callable
        A function taking two vectors (u, v) and returning a scalar, or a
        vectorized function that can operate on full arrays.
    y_true : array-like of shape (n_samples, n_features)
        True samples.
    y_pred : array-like of shape (n_samples, n_features)
        Predicted samples.

    Returns
    -------
    np.ndarray
        Distances per sample, shape (n_samples,).

    Raises
    ------
    InputShapeError
        If `y_true` and `y_pred` have different shapes.
    ValueError
        If `distance` is None or inputs are None/empty.
    TypeError
        If `distance` is not callable or inputs are not array-like.

    Examples
    --------
    >>> import numpy as np
    >>> def l1(u, v): return np.sum(np.abs(u - v))
    >>> y_true = np.array([[1, 2], [3, 4]])
    >>> y_pred = np.array([[2, 2], [2, 5]])
    >>> _call_distance_for_samples(l1, y_true, y_pred)
    array([1., 2.])

    Vectorized distance functions also work:

    >>> def l2(u, v): return np.linalg.norm(u - v, axis=-1)
    >>> _call_distance_for_samples(l2, y_true, y_pred)
    array([1.        , 1.41421356])
    """
    distance = _ensure_callable(distance, "distance")
    y_true = _ensure_array_like(y_true, "y_true")
    y_pred = _ensure_array_like(y_pred, "y_pred")

    if y_true.shape != y_pred.shape:
        raise InputShapeError("y_true and y_pred must have the same shape")

    try:
        # Try vectorized call
        result = np.asarray(distance(y_true, y_pred))
        if result.shape == (y_true.shape[0],):
            return result.astype(float)
        # If scalar or wrong shape, fall back to row-wise
    except Exception:
        pass

    # Row-wise fallback
    return np.array([distance(u, v) for u, v in zip(y_true, y_pred)], dtype=float)
