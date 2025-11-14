import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array, _ensure_no_nan
)

def compute_noise_fraction(labels: np.ndarray) -> float:
    """
    Compute the fraction of points labeled as noise (-1) in DBSCAN.

    Parameters
    ----------
    labels : array_like of shape (n_samples,)
        Cluster labels assigned by DBSCAN. Noise points must be labeled -1.

    Returns
    -------
    float
        Fraction of points labeled as noise, in [0.0, 1.0].

    Raises
    ------
    TypeError
        If labels is not array-like or contains non-numeric values.
    ValueError
        If labels array is empty.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([0, 0, 1, 1, -1, -1])
    >>> compute_noise_fraction(labels)
    0.3333333333333333

    Edge case: no noise
    >>> labels = np.array([0, 1, 1, 2])
    >>> compute_noise_fraction(labels)
    0.0

    Error case: empty labels
    >>> compute_noise_fraction(np.array([]))
    Traceback (most recent call last):
        ...
    ValueError: labels array must not be empty.
    """
    # Ensure labels is a numeric 1D array
    labels = _ensure_numeric_array(labels, name="labels", ndim=1)
    _ensure_no_nan(labels, "labels")

    if labels.size == 0:
        raise ValueError("labels array must not be empty.")

    # Count how many points are labeled as noise (-1)
    noise_count = np.sum(labels == -1)

    # Compute fraction of noise points
    fraction = noise_count / labels.size

    return float(fraction)
