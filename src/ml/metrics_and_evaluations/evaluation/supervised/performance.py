import numpy as np
from typing import Any, Callable, Optional, Sequence
from ml.metrics_and_evaluations.metrics.metrics import LnDistanceConstructor
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_array_like,
    InputShapeError,
    ModelInterfaceError,
    _ensure_no_nan,
    _ensure_numeric_array
)
from ml.utils.per_sample import _apply_per_sample
from typing import Any

from ml.utils._errors_and_warnings._general_error_handling import ModelInterfaceError

__all__ = [
    "mean_squared_error",
    "classification_accuracy",
    "NegativeLogLikelihoodBatchConstructor",
    "mean_absolute_error",
]

def _check_model_predict(model: Any) -> None:
    """
    Ensure that a model exposes a callable `predict(X)` method.

    Parameters
    ----------
    model : object
        Model instance to validate.

    Raises
    ------
    ModelInterfaceError
        If the model does not have a callable `predict` method.

    Examples
    --------
    >>> class Dummy: pass
    >>> _check_model_predict(Dummy())
    Traceback (most recent call last):
        ...
    ModelInterfaceError: model must have a callable `predict(X)` method

    >>> class Good:
    ...     def predict(self, X): return X
    >>> _check_model_predict(Good())  # passes
    """
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ModelInterfaceError("model must have a callable `predict(X)` method")





def mean_squared_error(
    model: Any,
    X: Sequence,
    y: Sequence,
    *,
    distance: Optional[Callable[[Sequence, Sequence], Sequence]] = None,
    sample_weight: Optional[Sequence] = None,
) -> float:
    """
    Compute mean squared error (MSE) using a per-sample distance function.

    Parameters
    ----------
    model : object
        Must expose predict(X).
    X : array_like
        Input features, shape (n_samples, ...).
    y : array_like
        Targets, shape (n_samples, ...).
    distance : callable, optional
        Distance function. If None, uses Euclidean norm of residuals.
    sample_weight : array_like, optional
        1D weights of length n_samples.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    InputShapeError
        If shapes are incompatible.
    ValueError
        If sample weights sum to zero, distance returns negatives,
        or distance returns non-numeric values.
    TypeError
        If distance is not callable.

    Examples
    --------
    >>> class Dummy:
    ...     def predict(self, X): return X
    >>> mse = mean_squared_error(Dummy(), [[1],[2]], [[1],[3]])
    >>> round(mse, 2)
    0.5
    """
    _check_model_predict(model)

    # Convert inputs to arrays
    X_arr = _ensure_array_like(X, name="X")
    y_arr = _ensure_array_like(y, name="y")

    preds = model.predict(X_arr)
    preds = _ensure_array_like(preds, name="predictions")

    # Validate leading axis lengths
    if X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] != preds.shape[0]:
        raise InputShapeError("X, y, and predictions must have the same number of samples (leading axis)")

    n_samples = X_arr.shape[0]

    # Normalize shapes for comparison
    y_proc = y_arr.reshape(n_samples, -1) if y_arr.ndim == 1 else y_arr
    preds_proc = preds.reshape(n_samples, -1) if preds.ndim == 1 else preds

    if y_proc.shape[1:] != preds_proc.shape[1:]:
        raise InputShapeError(
            f"Per-sample shapes differ: y per-sample shape={y_proc.shape[1:]} "
            f"vs preds per-sample shape={preds_proc.shape[1:]}"
        )

    # Choose distance function
    if distance is None:
        def _default_distance(y_true, y_pred):
            diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
            # Euclidean norm per sample
            return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
        distance_fn = _default_distance
    else:
        if not callable(distance):
            raise TypeError("distance must be callable")
        distance_fn = distance

    # Compute per-sample distances using unified helper
    per_sample_dist = _apply_per_sample(distance_fn, y_proc, preds_proc)
    per_sample_dist = np.asarray(per_sample_dist)

    # Validate distance output
    if per_sample_dist.shape[0] != n_samples:
        raise InputShapeError("Distance function must return one distance value per sample (length n_samples)")
    if not np.issubdtype(per_sample_dist.dtype, np.number):
        raise ValueError("distance function must return numeric values")
    if np.any(per_sample_dist < 0):
        raise ValueError("distance function returned negative values")

    # Square the distances to get squared errors
    sq_errors = per_sample_dist ** 2

    # Apply sample weights if provided
    if sample_weight is not None:
        sw = _ensure_array_like(sample_weight, name="sample_weight")
        if sw.ndim != 1 or sw.shape[0] != n_samples:
            raise InputShapeError("sample_weight must be 1D and match number of samples")
        sw = sw.astype(float)
        denom = np.sum(sw)
        if denom == 0:
            raise ValueError("sum of sample_weight is zero")
        return float(np.sum(sq_errors * sw) / denom)

    return float(np.mean(sq_errors))



def classification_accuracy(model: Any, X: Sequence, y: Sequence) -> float:
    """
    Compute classification accuracy (fraction of samples predicted exactly correct).

    Parameters
    ----------
    model : object
        Must expose predict(X).
    X : array_like
        Input features, shape (n_samples, ...).
    y : array_like
        True labels, shape (n_samples, ...).

    Returns
    -------
    float
        Accuracy in [0.0, 1.0].

    Raises
    ------
    InputShapeError
        If shapes are incompatible.
    TypeError
        If comparison fails.

    Examples
    --------
    >>> class Dummy:
    ...     def predict(self, X): return X
    >>> acc = classification_accuracy(Dummy(), [0,1,1], [0,1,0])
    >>> round(acc, 2)
    0.67
    >>> acc = classification_accuracy(Dummy(), ["cat","dog"], ["cat","dog"])
    >>> acc
    1.0
    """
    _check_model_predict(model)

    # Convert inputs to arrays
    X_arr = _ensure_array_like(X, name="X")
    y_arr = _ensure_array_like(y, name="y")

    # Validate leading axis
    if X_arr.ndim == 0:
        raise InputShapeError("X must have a leading sample axis (n_samples, ...)")
    n = X_arr.shape[0]
    if y_arr.shape[0] != n:
        raise InputShapeError("X and y must have the same number of samples (leading axis)")

    # Get predictions (robust to scalar-only or vectorized predict)
    preds = _apply_per_sample(model.predict, X_arr)
    preds = _ensure_array_like(preds, name="predictions")
    if preds.shape[0] != n:
        raise InputShapeError("Number of predictions does not match number of samples")

    try:
        # Case 1: both y and preds are 1D (simple labels)
        if y_arr.ndim == 1 and preds.ndim == 1:
            correct = (preds == y_arr)

        # Case 2: multi-dimensional labels (e.g., one-hot vectors)
        else:
            y_flat = np.asarray(y_arr).reshape(n, -1)
            p_flat = np.asarray(preds).reshape(n, -1)

            # Ensure per-sample component counts match
            if y_flat.shape[1] != p_flat.shape[1]:
                raise InputShapeError("Per-sample component counts differ between y and predictions")

            # A sample is correct if all components match
            correct = np.all(p_flat == y_flat, axis=1)

    except Exception as exc:
        raise TypeError(f"Error comparing predictions and targets: {exc}") from exc

    # Compute accuracy as mean of correct matches
    return float(np.mean(correct.astype(float)))



def NegativeLogLikelihoodBatchConstructor(epsilon: float = 1e-8):
    """
    Create a vectorized negative log likelihood (NLL) loss function.

    Parameters
    ----------
    epsilon : float, optional
        Small constant added inside log to avoid log(0).
        Default is 1e-8.

    Returns
    -------
    function
        A function ``f(Y_true, Y_pred)`` that computes per-sample
        negative log likelihoods for a batch.

    Raises
    ------
    ValueError
        If epsilon is not positive.

    Examples
    --------
    >>> NLL_batch = NegativeLogLikelihoodBatchConstructor()
    >>> Y_true = np.array([[0,1,0],[1,0,0]])
    >>> Y_pred = np.array([[0.2,0.7,0.1],[0.9,0.05,0.05]])
    >>> np.round(NLL_batch(Y_true, Y_pred), 4)
    array([0.3567, 0.1054])

    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    def nll_batch(y_true, y_pred):
        # Ensure both inputs are numeric 2-D arrays
        a = _ensure_numeric_array(y_true, name="Y_true", ndim=2)
        b = _ensure_numeric_array(y_pred, name="Y_pred", ndim=2)

        # Ensure no NaN values
        _ensure_no_nan(a, name="Y_true")
        _ensure_no_nan(b, name="Y_pred")

        # Ensure same shape
        if a.shape != b.shape:
            raise InputShapeError(
                f"Shape mismatch: Y_true {a.shape} vs Y_pred {b.shape}"
            )

        # Compute per-sample NLL: - sum(y_true * log(y_pred + epsilon), axis=1)
        return -np.sum(a * np.log(b + epsilon), axis=1).astype(float)

    return nll_batch

def mean_absolute_error(
    model: Any,
    X: Sequence,
    y: Sequence,
    *,
    distance: Optional[Callable[[Sequence, Sequence], Sequence]] = None,
    sample_weight: Optional[Sequence] = None,
) -> float:
    """
    Compute mean absolute error (MAE) using a per-sample distance function.

    This function evaluates a model by comparing its predictions against true targets
    and computing the average absolute error. By default, it uses the L1 (Manhattan)
    norm via `LnDistanceConstructor(1)`, but a custom distance function can be supplied.

    Parameters
    ----------
    model : object
        Must expose predict(X).
    X : array_like
        Input features, shape (n_samples, ...).
    y : array_like
        Targets, shape (n_samples, ...).
    distance : callable, optional
        Distance function. If None, uses L1 norm of residuals (Manhattan distance).
    sample_weight : array_like, optional
        1D weights of length n_samples.

    Returns
    -------
    float
        Mean absolute error.

    Raises
    ------
    InputShapeError
        If shapes are incompatible.
    ValueError
        If sample weights sum to zero, distance returns negatives,
        or distance returns non-numeric values.
    TypeError
        If distance is not callable.

    Examples
    --------
    >>> class Dummy:
    ...     def predict(self, X): return X
    >>> mae = mean_absolute_error(Dummy(), [[1],[2]], [[1],[3]])
    >>> round(mae, 2)
    0.5
    """
    _check_model_predict(model)

    # Convert inputs to arrays
    X_arr = _ensure_array_like(X, name="X")
    y_arr = _ensure_array_like(y, name="y")

    preds = model.predict(X_arr)
    preds = _ensure_array_like(preds, name="predictions")

    # Validate leading axis lengths
    if X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] != preds.shape[0]:
        raise InputShapeError("X, y, and predictions must have the same number of samples (leading axis)")

    n_samples = X_arr.shape[0]

    # Normalize shapes for comparison
    y_proc = y_arr.reshape(n_samples, -1) if y_arr.ndim == 1 else y_arr
    preds_proc = preds.reshape(n_samples, -1) if preds.ndim == 1 else preds

    if y_proc.shape[1:] != preds_proc.shape[1:]:
        raise InputShapeError(
            f"Per-sample shapes differ: y per-sample shape={y_proc.shape[1:]} "
            f"vs preds per-sample shape={preds_proc.shape[1:]}"
        )

    # Choose distance function
    if distance is None:
        distance_fn = LnDistanceConstructor(1)  # L1 norm
    else:
        if not callable(distance):
            raise TypeError("distance must be callable")
        distance_fn = distance

    # Compute per-sample distances using unified helper
    per_sample_dist = _apply_per_sample(distance_fn, y_proc, preds_proc)
    per_sample_dist = np.asarray(per_sample_dist)

    # Validate distance output
    if per_sample_dist.shape[0] != n_samples:
        raise InputShapeError("Distance function must return one distance value per sample (length n_samples)")
    if not np.issubdtype(per_sample_dist.dtype, np.number):
        raise ValueError("distance function must return numeric values")
    if np.any(per_sample_dist < 0):
        raise ValueError("distance function returned negative values")

    # Apply sample weights if provided
    if sample_weight is not None:
        sw = _ensure_array_like(sample_weight, name="sample_weight")
        if sw.ndim != 1 or sw.shape[0] != n_samples:
            raise InputShapeError("sample_weight must be 1D and match number of samples")
        sw = sw.astype(float)
        denom = np.sum(sw)
        if denom == 0:
            raise ValueError("sum of sample_weight is zero")
        return float(np.sum(per_sample_dist * sw) / denom)

    return float(np.mean(per_sample_dist))
