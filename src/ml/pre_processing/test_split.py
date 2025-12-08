import numpy as np
from typing import Sequence, Tuple
from ml.utils._errors_and_warnings._general_error_handling import _ensure_array_like


def train_test_split(data, train_ratio=0.7, seed=None):
    """
    Split a dataset into training and testing sets.

    Shuffles the data and splits it into training and testing subsets
    according to the specified ratio.

    Parameters
    ----------
    data : sequence
        Input dataset (list, tuple, or array).
    train_ratio : float, optional
        Proportion of data to include in the training set (default 0.7).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of lists
        (train, test) subsets.

    Raises
    ------
    ValueError
        If input is empty or train_ratio is not between 0 and 1.

    Examples
    --------
    >>> train, test = train_test_split([1, 2, 3, 4, 5], train_ratio=0.6, seed=42)
    >>> len(train), len(test)
    (3, 2)
    """
    data = list(data)  # Ensure sequence type
    if len(data) == 0:
        raise ValueError("Input must not be empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    # Shuffle indices reproducibly
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)

    # Split into train/test
    split_index = int(len(data) * train_ratio)
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]

    return train, test


def train_test_split_arrays(
    data: Sequence,
    labels: Sequence,
    train_ratio: float = 0.7,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split aligned data and labels into training and testing sets.

    Parameters
    ----------
    data : sequence
        Feature data (array-like), shape (n_samples, ...).
    labels : sequence
        Corresponding labels (array-like), shape (n_samples,).
    train_ratio : float, optional
        Proportion of data to include in the training set (default 0.7).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (X_train, y_train, X_test, y_test)

    Raises
    ------
    ValueError
        If inputs are empty, misaligned, or train_ratio is invalid.

    Examples
    --------
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> X_train, y_train, X_test, y_test = train_test_split_arrays(X, y, train_ratio=0.6, seed=42)
    >>> X_train.shape, y_train.shape
    ((3, 1), (3,))
    """
    X = _ensure_array_like(data, "data")
    y = _ensure_array_like(labels, "labels")

    if X.shape[0] != y.shape[0]:
        raise ValueError("data and labels must have the same number of samples.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    split = int(X.shape[0] * train_ratio)
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
