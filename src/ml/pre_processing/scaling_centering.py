import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import _ensure_no_nan, _ensure_numeric_array, _ensure_nonzero


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
    _ensure_nonzero(max_val, "maximum absolute value")

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
    _ensure_nonzero(avg_abs, "average absolute value")

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
    _ensure_nonzero(max_vals, "maximum absolute value")

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
    _ensure_nonzero(avg_abs, "average absolute value")

    return arr / avg_abs



def center_data(data: np.ndarray) -> np.ndarray:
    """
    Center a 1D array by subtracting its mean.

    Parameters
    ----------
    data : array_like
        One-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Centered array (mean becomes 0).

    Raises
    ------
    ValueError
        If the input array is empty.

    Examples
    --------
    >>> center_data([1, 2, 3])
    array([-1.,  0.,  1.])
    """
    arr = _ensure_numeric_array(data, name="data", ndim=1)
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    return arr - np.mean(arr)


def center_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Center a 2D array column-wise by subtracting the mean of each column.

    Parameters
    ----------
    vectors : array_like
        Two-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Column-wise centered array (each column mean becomes 0).

    Raises
    ------
    ValueError
        If any column is empty.

    Examples
    --------
    >>> import numpy as np
    >>> center_vectors([[1, 2], [3, 4]])
    array([[-1., -1.],
           [ 1.,  1.]])
    """
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)
    if arr.shape[0] == 0:
        raise ValueError("Input matrix must not be empty.")
    return arr - np.mean(arr, axis=0)

def center_and_normalize(data: np.ndarray,
                         method: str = "max_abs",
                         axis: int = 0) -> np.ndarray:
    """
    Center and normalize a numeric array.

    Parameters
    ----------
    data : array_like
        Numeric array (1D or 2D).
    method : {"max_abs", "average_abs"}, optional
        Normalization method. Default is "max_abs".
    axis : {0, 1}, optional
        For 2D arrays, axis along which to normalize (0 = columns, 1 = rows).
        Default is 0 (columns).

    Returns
    -------
    np.ndarray
        Centered and normalized array.

    Raises
    ------
    ValueError
        If method is invalid or normalization divisor is zero.

    Examples
    --------
    >>> center_and_normalize([1, -2, 3], method="max_abs")
    array([ 0.125, -1.   ,  0.875])

    >>> import numpy as np
    >>> center_and_normalize(np.array([[1, -2], [3, 4]]), method="average_abs")
    array([[-1., -1.],
           [ 1.,  1.]])
    """
    arr = _ensure_numeric_array(data, name="data")

    # Center
    if arr.ndim == 1:
        arr = center_data(arr)
        if method == "max_abs":
            return normalize_by_max_abs(arr)
        elif method == "average_abs":
            return normalize_by_average_abs(arr)
        else:
            raise ValueError(f"Unknown method '{method}'.")
    elif arr.ndim == 2:
        arr = center_vectors(arr)
        if method == "max_abs":
            return normalize_vectors_by_max_abs(arr)
        elif method == "average_abs":
            return normalize_vectors_by_average_abs(arr)
        else:
            raise ValueError(f"Unknown method '{method}'.")
    else:
        raise ValueError("Input must be 1D or 2D array.")



def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardize a 1D array using z-score normalization.

    Each value is transformed as (x - mean) / std, so the resulting
    array has mean 0 and standard deviation 1.

    Parameters
    ----------
    data : array_like
        One-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Standardized array.

    Raises
    ------
    ValueError
        If the standard deviation is zero (constant array).

    Examples
    --------
    >>> np.allclose(standardize_data([10, 20, 30]), [-1.22474487, 0., 1.22474487])
    True


    >>> standardize_data([5, 5, 5])
    Traceback (most recent call last):
        ...
    ValueError: Standard deviation must not be zero.
    """
    arr = _ensure_numeric_array(data, name="data", ndim=1)
    _ensure_no_nan(arr, "data")

    mean = np.mean(arr)
    std = np.std(arr)
    _ensure_nonzero(std, "standard deviation")

    return (arr - mean) / std


def standardize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Standardize a 2D array column-wise using z-score normalization.

    Each column is transformed so that its mean becomes 0 and its
    standard deviation becomes 1.

    Parameters
    ----------
    vectors : array_like
        Two-dimensional numeric array.

    Returns
    -------
    np.ndarray
        Column-wise standardized array.

    Raises
    ------
    ValueError
        If any column has zero standard deviation.

    Examples
    --------
    >>> import numpy as np
    >>> standardize_vectors([[1, 2], [3, 4]])
    array([[-1., -1.],
           [ 1.,  1.]])

    >>> standardize_vectors([[5, 5], [5, 5]])
    Traceback (most recent call last):
        ...
    ValueError: One or more columns have zero standard deviation.
    """
    arr = _ensure_numeric_array(vectors, name="vectors", ndim=2)
    _ensure_no_nan(arr, "vectors")

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    _ensure_nonzero(std, "standard deviation")

    return (arr - mean) / std
