from typing import Optional
import numpy as np
from pyparsing import Callable
from ml.metrics_and_evaluations.metrics.metrics import LnDistanceConstructor
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError, _ensure_callable, _ensure_no_nan, _ensure_numeric_array
from ml.utils.per_sample import _apply_per_sample

class Loss:
    """
    Encapsulates a loss function and its gradient.
    """
    def __init__(self, func, grad, name=None):
        self.func = func
        self.grad = grad
        self.name = name or func.__name__
    """
    calls the function with proper checks
    """
    def __call__(self, y_true, y_pred):
        """Compute per-sample scalar losses with validation."""
        y_true = _ensure_numeric_array(y_true, name="y_true")
        y_pred = _ensure_numeric_array(y_pred, name="y_pred")
        _ensure_no_nan(y_true, name="y_true")
        _ensure_no_nan(y_pred, name="y_pred")

        out = self.func(y_true, y_pred)
        self._validate_loss_output(out, y_true.shape[0])
        return out

    def gradient(self, y_true, y_pred):
        """Compute gradient w.r.t. predictions with validation."""
        y_true = _ensure_numeric_array(y_true, name="y_true")
        y_pred = _ensure_numeric_array(y_pred, name="y_pred")
        _ensure_no_nan(y_true, name="y_true")
        _ensure_no_nan(y_pred, name="y_pred")

        grad = self.grad(y_true, y_pred)
        self._validate_grad_output(grad, y_pred.shape)
        return grad

    def _validate_loss_output(self, out, n_samples):
        if not isinstance(out, np.ndarray):
            raise ValueError("Loss must return a NumPy array.")
        if out.shape != (n_samples,):
            raise ValueError(
                f"Loss must return per-sample scalars. Expected shape ({n_samples},), got {out.shape}."
            )
        if np.isnan(out).any():
            raise ValueError("Loss output contains NaN values.")

    def _validate_grad_output(self, out, expected_shape):
        if not isinstance(out, np.ndarray):
            raise ValueError("Gradient must return a NumPy array.")
        if out.shape != expected_shape:
            raise ValueError(
                f"Gradient must match prediction shape. Expected {expected_shape}, got {out.shape}."
            )
        if np.isnan(out).any():
            raise ValueError("Gradient output contains NaN values.")
    def mean_loss(self, y_true, y_pred):
        """Compute mean loss over all samples."""
        losses = self.__call__(y_true, y_pred)
        return np.mean(losses)

    def __repr__(self):
        return f"Loss(name={self.name})"



def CrossEntropyLossConstructor(epsilon: float = 1e-8):
    """
    Construct a Loss object for cross-entropy.

    Parameters
    ----------
    epsilon : float, optional
        Small constant added inside log to avoid log(0).
        Default is 1e-8.

    Returns
    -------
    Loss
        A Loss object with func and grad methods for cross-entropy.

    Raises
    ------
    ValueError
        If epsilon is not positive.

    Examples
    --------
    >>> ce_loss = CrossEntropyLossConstructor()
    >>> Y_true = np.array([[0,1,0],[1,0,0]])
    >>> Y_pred = np.array([[0.2,0.7,0.1],[0.9,0.05,0.05]])
    >>> ce_loss(Y_true, Y_pred)
    array([0.35667494, 0.10536052])
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    def ce_func(y_true, y_pred):
        a = _ensure_numeric_array(y_true, name="Y_true", ndim=2)
        b = _ensure_numeric_array(y_pred, name="Y_pred", ndim=2)
        _ensure_no_nan(a, name="Y_true")
        _ensure_no_nan(b, name="Y_pred")
        if a.shape != b.shape:
            raise InputShapeError(f"Shape mismatch: Y_true {a.shape} vs Y_pred {b.shape}")
        b = np.clip(b, epsilon, 1.0 - epsilon)
        return -np.sum(a * np.log(b), axis=1).astype(float)

    def ce_grad(y_true, y_pred):
        a = _ensure_numeric_array(y_true, name="Y_true", ndim=2)
        b = _ensure_numeric_array(y_pred, name="Y_pred", ndim=2)
        _ensure_no_nan(a, name="Y_true")
        _ensure_no_nan(b, name="Y_pred")
        if a.shape != b.shape:
            raise InputShapeError(f"Shape mismatch: Y_true {a.shape} vs Y_pred {b.shape}")
        b = np.clip(b, epsilon, 1.0 - epsilon)
        return -(a / b).astype(float)

    return Loss(ce_func, ce_grad, name="cross_entropy")
cross_entropy = CrossEntropyLossConstructor()

# --- Mean Squared Error (MSE) ---
def mse_func(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise InputShapeError(
            f"Shape mismatch: Y_true {y_true.shape} vs Y_pred {y_pred.shape}"
        )
    residual = y_true - y_pred
    # Per-sample mean squared error
    return np.mean(residual ** 2, axis=1)

def mse_grad(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise InputShapeError(
            f"Shape mismatch: Y_true {y_true.shape} vs Y_pred {y_pred.shape}"
        )
    # Gradient wrt predictions: 2 * (y_pred - y_true) / n_features
    n_features = y_true.shape[1]
    return 2.0 * (y_pred - y_true) / n_features

MSE = Loss(mse_func, mse_grad, name="mse")


# --- Mean Absolute Error (MAE) ---
def mae_func(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise InputShapeError(
            f"Shape mismatch: Y_true {y_true.shape} vs Y_pred {y_pred.shape}"
        )
    residual = y_true - y_pred
    # Per-sample mean absolute error
    return np.mean(np.abs(residual), axis=1)

def mae_grad(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise InputShapeError(
            f"Shape mismatch: Y_true {y_true.shape} vs Y_pred {y_pred.shape}"
        )
    n_features = y_true.shape[1]
    # Gradient wrt predictions: sign(y_pred - y_true) / n_features
    return np.sign(y_pred - y_true) / n_features

MAE = Loss(mae_func, mae_grad, name="mae")


mean_squared_error = MSE

# Approved losses dictionary
# Do note that these errors are not actually the averages, they are per sample pre-averaged losses.
# i.e, MSE is just the squared error per sample, not averaged over samples. I keep the name MSE because it is used for that purpose

APPROVED_LOSSES = {
    "MAE": MAE,                # Mean Absolute Error (default L1 distance)
    "mse": mean_squared_error,  # Mean Squared Error
    "cross_entropy": cross_entropy,  # Cross-Entropy / NLL
    "mae": MAE,
    "MSE": mean_squared_error,
    "Cross_Entropy": cross_entropy,
    # Add more as needed, e.g. hinge, huber, etc.
}
