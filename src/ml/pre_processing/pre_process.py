import numpy as np
from ml.utils._errors_and_warnings.error_handling import _ensure_numeric_array


def normalize_by_max_abs(data: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D array by its maximum absolute value.

    Parameters
    ----------
    data : array_like
        One-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Normalized array.

    Raises
    ------
    ValueError
        If the maximum absolute value is zero.

    Examples
    --------
    >>> normalize_by_max_abs([1, -2, 3])
    array([ 0.33333333, -0.66666667,  1.        ])
    """
    # Ensure input is numeric 1-D array
    arr = _ensure_numeric_array(data, name="data", ndim=1)

    # Compute maximum absolute value
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        raise ValueError("Maximum absolute value must not be zero.")

    # Normalize by max absolute value
    return arr / max_val


def normalize_by_average_abs(data: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D array by its average absolute value.

    Parameters
    ----------
    data : array_like
        One-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Normalized array.

    Raises
    ------
    ValueError
        If the average absolute value is zero.

    Examples
    --------
    >>> normalize_by_average_abs([1, -2, 3])
    array([ 0.5, -1. ,  1.5])
    """
    arr = _ensure_numeric_array(data, name="data", ndim=1)

    # Compute average absolute value
    avg_abs = np.mean(np.abs(arr))
    if avg_abs == 0:
        raise ValueError("Average absolute value must not be zero.")

    return arr / avg_abs


def normalize_vectors_by_max_abs(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D array column-wise by the maximum absolute value of each column.

    Each column is divided by its maximum absolute value. This ensures that
    the largest absolute entry in each column becomes 1.0.

    Parameters
    ----------
    vectors : array_like
        Two-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Column-wise normalized array.

    Raises
    ------
    ValueError
        If any column has maximum absolute value zero.

    Examples
    --------
    >>> import numpy as np
    >>> normalize_vectors_by_max_abs([[1, -2], [3, 4]])
    array([[ 0.33333333, -0.5       ],
           [ 1.        ,  1.        ]])

    >>> normalize_vectors_by_max_abs([[0, 0], [0, 0]])
    Traceback (most recent call last):
        ...
    ValueError: One or more columns have maximum absolute value zero.

    """
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)

    # Compute max absolute values per column
    max_vals = np.max(np.abs(arr), axis=0)
    if np.any(max_vals == 0):
        raise ValueError("One or more columns have maximum absolute value zero.")

    return arr / max_vals


def normalize_vectors_by_average_abs(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D array column-wise by the average absolute value of each column.

    Each column is divided by the mean of its absolute values. This ensures that
    the average absolute magnitude of each column becomes 1.0.

    Parameters
    ----------
    vectors : array_like
        Two-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Column-wise normalized array.

    Raises
    ------
    ValueError
        If any column has average absolute value zero.

    Examples
    --------
    >>> import numpy as np
    >>> normalize_vectors_by_average_abs([[1, -2], [3, 4]])
    array([[ 0.5       , -0.66666667],
           [ 1.5       ,  1.33333333]])

    >>> normalize_vectors_by_average_abs([[2, -2], [2, 2]])
    array([[ 1., -1.],
           [ 1.,  1.]])
    """
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)

    # Compute average absolute values per column
    avg_abs = np.mean(np.abs(arr), axis=0)
    if np.any(avg_abs == 0):
        raise ValueError("One or more columns have average absolute value zero.")

    return arr / avg_abs



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
