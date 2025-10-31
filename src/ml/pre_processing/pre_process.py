import numpy as np

def _ensure_numeric_array(data, name="data", ndim=None) -> np.ndarray:
    """
    Convert input to a numeric NumPy array with explicit checks.

    Parameters
    ----------
    data : array_like
        Input data to validate.
    name : str, optional
        Name of the variable (for error messages).
    ndim : int, optional
        If provided, require array to have this number of dimensions.

    Returns
    -------
    np.ndarray
        Float array.

    Raises
    ------
    ValueError
        If array is empty or ndim does not match.
    TypeError
        If array contains non-numeric values.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain only numeric values.")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array.")
    return arr.astype(float)


def normalize_by_max_abs(data: np.ndarray) -> np.ndarray:
    """Normalize a 1D array by the maximum absolute value."""
    arr = _ensure_numeric_array(data, name="data", ndim=1)
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        raise ValueError("Maximum absolute value must not be zero.")
    return arr / max_val


def normalize_by_average_abs(data: np.ndarray) -> np.ndarray:
    """Normalize a 1D array by the average absolute value."""
    arr = _ensure_numeric_array(data, name="data", ndim=1)
    avg_abs = np.mean(np.abs(arr))
    if avg_abs == 0:
        raise ValueError("Average absolute value must not be zero.")
    return arr / avg_abs


def normalize_vectors_by_max_abs(vectors: np.ndarray) -> np.ndarray:
    """Normalize a 2D array column-wise by maximum absolute values."""
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)
    max_vals = np.max(np.abs(arr), axis=0)
    if np.any(max_vals == 0):
        raise ValueError("One or more columns have maximum absolute value zero.")
    return arr / max_vals


def normalize_vectors_by_average_abs(vectors: np.ndarray) -> np.ndarray:
    """Normalize a 2D array column-wise by average absolute values."""
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)
    avg_abs = np.mean(np.abs(arr), axis=0)
    if np.any(avg_abs == 0):
        raise ValueError("One or more columns have average absolute value zero.")
    return arr / avg_abs


def train_test_split(data, train_ratio=0.7, seed=None):
    """
    Split a dataset into training and testing sets.

    Shuffles the data and splits it into training and testing subsets
    according to the specified ratio.
    """
    data = list(data)  # ensure sequence, works for arrays, lists, tuples
    if len(data) == 0:
        raise ValueError("Input must not be empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)

    split_index = int(len(data) * train_ratio)
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test
