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


