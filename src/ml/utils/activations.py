"""
Activation functions for neural networks and machine learning models.

Each activation validates its input using centralized error-handling utilities:
- `_ensure_numeric_array`: ensures the input is a NumPy numeric array.
- `_ensure_no_nan`: ensures the input contains no NaN values.

Raises
------
ValueError
    If NaN values are present in the input.
InputShapeError
    If the input shape is invalid (e.g., softmax requires 1D or 2D input).
"""

import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array,
    _ensure_no_nan,
    InputShapeError,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Maps real-valued inputs to the range (0, 1).
    Uses a numerically stable implementation to avoid overflow.

    Parameters
    ----------
    z : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Array of same shape as input, with values in (0, 1).

    Raises
    ------
    ValueError
        If input contains NaN values.

    Examples
    --------
    >>> sigmoid(np.array([-10.0, 0.0, 10.0]))
    array([4.53978687e-05, 5.00000000e-01, 9.99954602e-01])
    """
    arr = _ensure_numeric_array(z, name="z")
    _ensure_no_nan(arr, name="z")

    # Numerically stable computation
    out = np.empty_like(arr)
    pos = arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    expz = np.exp(arr[neg])
    out[neg] = expz / (1.0 + expz)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.

    Converts raw scores into probabilities that sum to 1 across classes.
    Uses a numerically stable implementation by subtracting the row max.

    Parameters
    ----------
    z : np.ndarray
        Input array of shape (n_classes,) or (n_samples, n_classes).

    Returns
    -------
    np.ndarray
        Probabilities of same shape, rows summing to 1.

    Raises
    ------
    ValueError
        If input contains NaN values.
    InputShapeError
        If input is not 1D or 2D.

    Examples
    --------
    >>> softmax(np.array([1.0, 2.0, 3.0]))
    array([[0.09003057, 0.24472847, 0.66524096]])
    """
    arr = _ensure_numeric_array(z, name="z")
    _ensure_no_nan(arr, name="z")

    # Ensure correct shape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise InputShapeError("softmax input must be 1D or 2D.")

    # Stability trick: subtract max per row
    arr = arr - np.max(arr, axis=1, keepdims=True)
    expz = np.exp(arr)
    return expz / np.sum(expz, axis=1, keepdims=True)


def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU).

    Maps negative inputs to 0, positive inputs unchanged.

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Elementwise max(0, z).

    Raises
    ------
    ValueError
        If input contains NaN values.

    Examples
    --------
    >>> relu(np.array([-2.0, -0.5, 0.0, 1.5]))
    array([0. , 0. , 0. , 1.5])
    """
    arr = _ensure_numeric_array(z, name="z")
    _ensure_no_nan(arr, name="z")
    return np.maximum(0.0, arr)


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation.

    Maps real-valued inputs to (-1, 1).

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of same shape, values in (-1, 1).

    Raises
    ------
    ValueError
        If input contains NaN values.

    Examples
    --------
    >>> tanh(np.array([-2.0, 0.0, 2.0]))
    array([-0.96402758,  0.        ,  0.96402758])
    """
    arr = _ensure_numeric_array(z, name="z")
    _ensure_no_nan(arr, name="z")
    return np.tanh(arr)


def step(z: np.ndarray) -> np.ndarray:
    """
    Heaviside step function.

    Outputs 0 for negative inputs, 1 for non-negative inputs.

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of 0s and 1s.

    Raises
    ------
    ValueError
        If input contains NaN values.

    Examples
    --------
    >>> step(np.array([-1.0, 0.0, 2.0]))
    array([0., 1., 1.])
    """
    arr = _ensure_numeric_array(z, name="z")
    _ensure_no_nan(arr, name="z")
    return (arr >= 0).astype(float)
