import numbers
import numpy as np
from typing import Any, Callable, Hashable, Optional, Sequence



#some error handling functions


def _ensure_positive_numeric(value: float, name: str) -> float:
    """
    Ensure that a parameter is a positive numeric value.

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        The name of the parameter (for error messages).

    Returns
    -------
    float
        The validated numeric value.

    Raises
    ------
    TypeError
        If the value is not numeric.
    ValueError
        If the value is not strictly positive.
    """
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a numeric value, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}.")
    return value

def _ensure_numeric_array(value, name: str = "array", ndim: int | None = None) -> np.ndarray:
    """
    Convert input to a NumPy array and ensure it contains only numeric values.

    Parameters
    ----------
    value : array_like
        Input data to validate. Can be a list, tuple, or NumPy array.
    name : str, optional
        Name used in error messages (default "array").
    ndim : int, optional
        Required number of dimensions. If provided, raises if mismatch.

    Returns
    -------
    np.ndarray
        A NumPy array containing numeric values.

    Raises
    ------
    ValueError
        If the array is empty or does not match the required dimensionality.
    TypeError
        If the array contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> _ensure_numeric_array([1, 2, 3], name="vec", ndim=1)
    array([1, 2, 3])
    >>> _ensure_numeric_array(np.array([[1.0, 2.0], [3.0, 4.0]]), name="X", ndim=2)
    array([[1., 2.],
           [3., 4.]])
    >>> _ensure_numeric_array([], name="empty")
    Traceback (most recent call last):
        ...
    ValueError: empty must be non-empty.
    >>> _ensure_numeric_array(["a", "b"], name="bad")
    Traceback (most recent call last):
        ...
    TypeError: bad must contain only numeric values.
    >>> _ensure_numeric_array([1, 2, 3], name="vec", ndim=2)
    Traceback (most recent call last):
        ...
    ValueError: vec must be 2-D, got 1-D.
    """

    if value is None:
        raise ValueError(f"{name} must be provided and non-None.")


    arr = np.asarray(value)

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain only numeric values.")

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {arr.ndim}-D.")

    return arr

def _ensure_numeric_scalar(value, name: str = "value"):
    """
    Ensure that a given value is a numeric scalar.

    Parameters
    ----------
    value : any
        The value to validate.
    name : str, optional
        Name used in error messages (default "value").

    Returns
    -------
    numbers.Real
        The validated numeric scalar.

    Raises
    ------
    TypeError
        If the value is not a numeric scalar.

    Examples
    --------
    >>> _ensure_numeric_scalar(3.14, name="pi")
    3.14
    >>> _ensure_numeric_scalar(42, name="answer")
    42
    >>> _ensure_numeric_scalar("not a number", name="bad")
    Traceback (most recent call last):
        ...
    TypeError: bad must be a numeric scalar, got str.
    """
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a numeric scalar, got {type(value).__name__}.")
    return value




def _ensure_array_like(a: Sequence, name: str) -> np.ndarray:
    if a is None:
        raise ValueError(f"{name} must be provided and non-None.")
    try:
        arr = np.asarray(a)
    except Exception as exc:
        raise TypeError(f"{name} is not array-like: {exc}") from exc
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr

def _check_sample_shapes_match(y: np.ndarray, preds: np.ndarray) -> None:
    """
    Ensure y and preds have the same shape for each sample.
    Valid shapes:
      - both have same overall shape
      - or same leading dimension (n_samples) and identical per-sample shapes (rest of axes)
    """
    if y.shape == preds.shape:
        return
    if y.ndim >= 1 and preds.ndim >= 1 and y.shape[0] == preds.shape[0]:
        # Compare trailing shapes after the sample axis
        if y.shape[1:] == preds.shape[1:]:
            return
    raise InputShapeError(
        f"Per-sample shapes disagree: y.shape={y.shape}, preds.shape={preds.shape}"
    )

def _ensure_no_nan(arr, name="array"):
    """
    Ensure that the array contains no NaN values.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    name : str, optional
        Name of the array for error messages.

    Raises
    ------
    ValueError
        If NaN values are found.
    """
    arr = np.asarray(arr)
    # Only check for NaN if dtype is numeric
    if np.issubdtype(arr.dtype, np.number):
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values.")
    return arr


def _ensure_positive_int(value: int, name: str) -> int:
    """
    Ensure that a parameter is a positive integer.

    Parameters
    ----------
    value : int
        The value to validate.
    name : str
        The name of the parameter (for error messages).
    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    TypeError
        If the value is not an integer.
    ValueError
        If the value is not strictly positive.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value

def _ensure_nonzero(value, name: str):
    """
    Ensure that a computed value (scalar or array) is non-zero.

    Parameters
    ----------
    value : float or np.ndarray
        The numeric value(s) to validate.
    name : str
        The name of the value (for error messages).
    
    Returns
    -------
    float or np.ndarray
        The validated value(s).

    Raises
    ------
    ValueError
        If any element of the value is zero.
    """
    # Handle both scalars and arrays uniformly
    if np.any(value == 0):
        raise ValueError(f"{name} must not be zero, got {value}.")
    return value


def _ensure_same_shape_1d(arr1: np.ndarray, name1: str, arr2: np.ndarray, name2: str) -> None:
    """
    Ensure two arrays are 1-D and have the same shape.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        Arrays to validate.
    name1, name2 : str
        Names of the arrays (for error messages).
    
    Raises
    ------
    ValueError
        If either array is not 1-D.
    InputShapeError
        If arrays are not the same shape.
    """
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError(f"{name1} and {name2} must be 1-D vectors.")
    if arr1.shape != arr2.shape:
        raise InputShapeError(
            f"{name1} and {name2} must have the same shape, got {arr1.shape} vs {arr2.shape}."
        )


def _ensure_string(value, name: str):
    """
    Ensure that a value is a string.

    Parameters
    ----------
    value : Any
        Value to validate.
    name : str
        Name of the parameter (for error messages).

    Raises
------
ValueError
    If value is None.
TypeError
    If value is not a string.

    """
    if value is None:
        raise ValueError(f"{name} must be provided and non-None.")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    return value




def _ensure_callable(value, name: str):
    """Ensure that a parameter is callable."""
    if value is None:
        raise ValueError(f"{name} must be provided and non-None.")
    if not callable(value):
        raise TypeError(f"{name} must be callable.")
    return value





def _ensure_non_empty(value, name: str) -> np.ndarray:
    """Ensure that a sequence or array is non-empty."""
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr

def _ensure_hashable_labels(labels, name: str) -> np.ndarray:
    """
    Ensure that all labels are hashable.

    Parameters
    ----------
    labels : array_like
        Sequence of labels.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        Array of labels.

    Raises
    ------
    TypeError
        If any label is unhashable.
    ValueError
        If labels are empty.
    """
    arr = _ensure_array_like(labels, name)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")

    for i, lbl in enumerate(arr):
        if not isinstance(lbl, Hashable):
            raise TypeError(f"{name} must contain only hashable elements; element {i} is not.")

    return arr


def _ensure_numeric_labels(labels, name: str) -> np.ndarray:
    """Ensure that all labels are numeric."""
    arr = _ensure_non_empty(labels, name)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    return arr.astype(float)

def _ensure_ndim(value, name: str, ndim: int) -> np.ndarray:
    """
    Ensure that an array has the required number of dimensions.

    Parameters
    ----------
    value : array_like
        Input data.
    name : str
        Name used in error messages.
    ndim : int
        Required number of dimensions.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    InputShapeError
        If the array does not have the required number of dimensions.

    Examples
    --------
    >>> _ensure_ndim([[1, 2], [3, 4]], "X", 2).shape
    (2, 2)
    >>> _ensure_ndim([1, 2, 3], "vec", 1).shape
    (3,)
    """
    arr = np.asarray(value)
    if arr.ndim != ndim:
        raise InputShapeError(f"{name} must be {ndim}-D, got {arr.ndim}-D.")
    return arr


def _ensure_same_length(arr1: np.ndarray, name1: str, arr2: np.ndarray, name2: str) -> None:
    """
    Ensure that two arrays have the same length along the leading axis.

    Raises
    ------
    InputShapeError
        If lengths differ.

    Examples
    --------
    >>> _ensure_same_length([1,2,3], "X", [0,1,0], "y")
    >>> _ensure_same_length([1,2], "X", [0,1,0], "y")
    Traceback (most recent call last):
        ...
    InputShapeError: X and y must have the same length (2 vs 3).

    """
    len1, len2 = len(arr1), len(arr2)
    if len1 != len2:
        raise InputShapeError(f"{name1} and {name2} must have the same length ({len1} vs {len2}).")



def _ensure_in_range(value: float, name: str, min_val=None, max_val=None, inclusive=True) -> float:
    """
    Ensure that a numeric value lies within a specified range.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name used in error messages.
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.
    inclusive : bool, default=True
        Whether bounds are inclusive.

    Raises
    ------
    ValueError
        If value is outside the allowed range.

    Examples
    --------
    >>> _ensure_in_range(0.5, "train_ratio", 0, 1)
    0.5
    >>> _ensure_in_range(1.5, "train_ratio", 0, 1)
    Traceback (most recent call last):
        ...
    ValueError: train_ratio must be <= 1, got 1.5.
    """
    if not isinstance(value, numbers.Real): 
        raise TypeError(f"{name} must be a numeric value.")
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}.")
        if not inclusive and value <= min_val:
            raise ValueError(f"{name} must be > {min_val}, got {value}.")
    if max_val is not None:
        if inclusive and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}.")
        if not inclusive and value >= max_val:
            raise ValueError(f"{name} must be < {max_val}, got {value}.")
    return value





class ModelInterfaceError(TypeError):
    pass

class InputShapeError(ValueError):
    pass

class InvalidSignatureError(ValueError):
    """Raised when a model signature string refers to a missing method."""
    pass