from typing import Optional
import numpy as np
from ml.distances.evaluation_metrics import classification_accuracy
from ml.utils._errors_and_warnings.error_handling import (
    _ensure_numeric_array,
    InputShapeError,
    _ensure_array_like,
)


class Perceptron:
    """
    Binary Perceptron classifier.

    Supports arbitrary binary labels (not just -1 and +1).
    Internally maps them to {-1, +1} for training.

    Parameters
    ----------
    n_features : int
        Number of input features.

    Attributes
    ----------
    w : ndarray of shape (n_features,)
        Weight vector.
    b : float
        Bias term.
    classes_ : ndarray of shape (2,)
        Original class labels, in the order mapped to {-1, +1}.
    history : list of float
        Training accuracy per epoch.

    Examples
    --------
    >>> p = Perceptron(n_features=2)
    >>> p.weights.shape
    (2,)
    """

    def __init__(self, n_features: int):
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive integer.")

        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        # Training history and class mapping
        self.history: list[float] = []
        self.classes_: Optional[np.ndarray] = None

        # Random generator (for reproducibility in training)
        self._rng: Optional[np.random.Generator] = None

    @property
    def weights(self) -> np.ndarray:
        """Return a copy of the current weight vector."""
        return self.w.copy()

    @property
    def bias(self) -> float:
        """Return the current bias term."""
        return float(self.b)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Compute linear activation (dot product + bias)."""
        x = _ensure_numeric_array(x, name="x")
        return np.dot(x, self.w) + self.b

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Map arbitrary labels to {-1, +1}.
        """
        y = _ensure_array_like(y, name="y")
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("Perceptron supports exactly 2 distinct classes.")

        # Store original class labels
        self.classes_ = classes

        # Map first class to -1, second to +1
        mapping = {classes[0]: -1, classes[1]: 1}
        return np.vectorize(mapping.get)(y)

    def _decode_labels(self, y_internal: np.ndarray) -> np.ndarray:
        """
        Map {-1, +1} back to original labels.

        Parameters
        ----------
        y_internal : array-like of shape (n_samples,)
            Encoded labels in {-1, +1}.

        Returns
        -------
        ndarray of shape (n_samples,)
            Original labels.

        Raises
        ------
        RuntimeError
            If the model has not been trained (classes_ is None).
        """
        if self.classes_ is None:
            raise RuntimeError("Model has not been trained yet.")
        mapping = {-1: self.classes_[0], 1: self.classes_[1]}
        return np.vectorize(mapping.get)(y_internal)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input samples.
        """
        X = _ensure_numeric_array(X, name="X", ndim=2)
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"X must have shape (n_samples, {self.w.shape[0]}).")

        scores = self._activation(X)
        internal_preds = np.where(scores >= 0.0, 1, -1)
        return self._decode_labels(internal_preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy using external utility.
        """
        return classification_accuracy(self, X, y)

    def train(
        self,
        X,
        y,
        *,
        lr: float = 1.0,
        epochs: int = 10,
        seed: Optional[int] = None,
        verbose: bool = False,
        stochastic: bool = True,
        shuffle: bool = True,
    ):
        """
        Train the perceptron.
        """
        # --- Input validation ---
        X = _ensure_numeric_array(X, name="X", ndim=2)
        y = _ensure_array_like(y, name="y")

        if X.shape[1] != self.w.shape[0]:
            raise InputShapeError(f"X must have shape (n_samples, {self.w.shape[0]}).")
        if X.shape[0] != y.shape[0]:
            raise InputShapeError("X and y must have the same number of samples (leading axis).")

        if not (isinstance(lr, (int, float)) and lr > 0):
            raise ValueError("Learning rate must be a positive number.")
        if not (isinstance(epochs, int) and epochs > 0):
            raise ValueError("Epochs must be a positive integer.")

        # --- Initialize RNG and weights ---
        self._rng = np.random.RandomState(seed)
        self.w = self._rng.normal(loc=0.0, scale=0.01, size=self.w.shape)
        self.b = 0.0

        # --- Encode labels ---
        y_internal = self._encode_labels(y)
        self.history = []

        # --- Training loop ---
        for epoch in range(1, epochs + 1):
            if stochastic:
                indices = np.arange(X.shape[0])
                if shuffle:
                    self._rng.shuffle(indices)
                for i in indices:
                    xi, yi = X[i], y_internal[i]
                    pred = 1 if self._activation(xi) >= 0 else -1
                    if pred != yi:
                        self.w += lr * yi * xi
                        self.b += lr * yi
            else:
                scores = self._activation(X)
                preds = np.where(scores >= 0.0, 1, -1)
                mis_idx = np.where(preds != y_internal)[0]
                if mis_idx.size > 0:
                    self.w += lr * np.sum(y_internal[mis_idx, None] * X[mis_idx], axis=0)
                    self.b += lr * np.sum(y_internal[mis_idx])

            acc = self.score(X, y)
            self.history.append(acc)
            if verbose:
                print(f"Epoch {epoch}/{epochs} - accuracy: {acc:.4f}")

    def reset(self, *, seed: Optional[int] = None):
        """
        Reinitialize weights, bias, RNG, and training history.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        if self._rng is None:
            self._rng = np.random.RandomState(seed)

        self.w = self._rng.normal(loc=0.0, scale=0.01, size=self.w.shape)
        self.b = 0.0
        self.history = []
