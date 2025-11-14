import numpy as np
import numbers
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance
from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan, _ensure_callable

def compute_inertia(
    X: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    distance_func: callable = EuclideanDistance,
) -> float:
    """
    Compute inertia (within-cluster sum of squares).

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Data points.
    centroids : array_like of shape (k, n_features)
        Cluster centroids.
    labels : array_like of shape (n_samples,)
        Cluster assignments for each sample.
    distance_func : callable, optional
        Distance function. Defaults to EuclideanDistance.

    Returns
    -------
    float
        Inertia value.

    Raises
    ------
    ValueError
        If X and labels lengths mismatch.
    TypeError
        If distance_func is not callable or does not return numeric.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> labels = np.array([0, 1])
    >>> compute_inertia(X, centroids, labels)
    0.0

    Error case: labels length mismatch
    >>> compute_inertia(X, centroids, np.array([0]))
    Traceback (most recent call last):
        ...
    ValueError: X and labels must have the same length.
    """
    X = _ensure_numeric_array(X, name="X", ndim=2)
    centroids = _ensure_numeric_array(centroids, name="centroids", ndim=2)
    labels = _ensure_numeric_array(labels, name="labels", ndim=1)
    _ensure_no_nan(X, "X")
    _ensure_no_nan(centroids, "centroids")

    _ensure_callable(distance_func, "distance_func")

    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length.")

    inertia = 0.0
    for i, x in enumerate(X):
        d = distance_func(x, centroids[labels[i]])
        if not isinstance(d, numbers.Real):
            raise TypeError(f"distance_func must return a numeric scalar, got {type(d).__name__}.")
        inertia += d**2
    return inertia


