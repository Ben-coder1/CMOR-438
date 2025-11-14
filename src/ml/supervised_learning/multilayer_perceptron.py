import numpy as np
from ml.metrics_and_evaluations.evaluation.supervised.performance import NegativeLogLikelihoodBatchConstructor, mean_squared_error
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array,
    _ensure_no_nan,
    _ensure_positive_int,
    InputShapeError,
    _validate_activation,
)
from ml.utils import activations


class DenseLayer:
    """
    A single fully-connected (dense) neural network layer.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_outputs : int
        Number of output units.
    activation : str or callable, optional
        Activation function. Defaults to 'sigmoid'.
        Can be a string name of a pre-approved function
        or a custom callable that maps np.ndarray -> np.ndarray.

    Attributes
    ----------
    W : np.ndarray
        Weight matrix of shape (n_inputs, n_outputs).
    b : np.ndarray
        Bias vector of shape (1, n_outputs).
    activation : callable
        Activation function.
    last_input : np.ndarray
        Cached input from last forward pass.
    last_output : np.ndarray
        Cached output from last forward pass.

    Examples
    --------
    >>> layer = DenseLayer(4, 3, activation="relu")
    >>> X = np.random.randn(5, 4)
    >>> out = layer.forward(X)
    >>> out.shape
    (5, 3)
    """

    def __init__(self, n_inputs, n_outputs, activation="sigmoid"):
        # Validate input/output sizes
        self.n_inputs = _ensure_positive_int(n_inputs, "n_inputs")
        self.n_outputs = _ensure_positive_int(n_outputs, "n_outputs")

        # Initialize weights and biases with appropriate scaling
        scale = np.sqrt(2.0 / n_inputs) if activation == "relu" else np.sqrt(1.0 / n_inputs)
        self.W = np.random.randn(n_inputs, n_outputs) * scale
        self.b = np.zeros((1, n_outputs))

        # Approved activations
        approved = {
            "sigmoid": activations.sigmoid,
            "relu": activations.relu,
            "tanh": activations.tanh,
            "softmax": activations.softmax,
            "step": activations.step,
        }

        # Validate activation function
        self.activation = _validate_activation(activation, approved)

        # Cache for forward pass
        self.last_input = None
        self.last_output = None

    def forward(self, X):
        """
        Forward pass through the layer.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_inputs)
            Input data.

        Returns
        -------
        np.ndarray
            Output of the layer after applying weights, biases, and activation.

        Raises
        ------
        InputShapeError
            If input feature dimension does not match layer weights.

        Examples
        --------
        >>> layer = DenseLayer(2, 2, activation="tanh")
        >>> X = np.array([[1.0, -1.0], [0.5, 0.5]])
        >>> out = layer.forward(X)
        >>> out.shape
        (2, 2)
        """
        # Validate input
        X = _ensure_numeric_array(X, name="X")
        _ensure_no_nan(X, name="X")

        # Check feature dimension
        if X.shape[1] != self.W.shape[0]:
            raise InputShapeError(
                f"Input has {X.shape[1]} features, expected {self.W.shape[0]}."
            )

        # Linear transformation + activation
        z = np.dot(X, self.W) + self.b
        a = self.activation(z)

        # Cache for backprop
        self.last_input = X
        self.last_output = a
        return a


class MultilayerPerceptron:
    """
    A simple multilayer perceptron (MLP) model composed of dense layers.

    Parameters
    ----------
    layers : list of DenseLayer or list of tuples
        Either a list of pre-constructed DenseLayer objects,
        or a list of specifications (n_inputs, n_outputs, activation).
    activations : list of str, optional
        Activation functions for each layer. Defaults to "sigmoid" for all layers
        if not provided. Ignored if `layers` is given as specifications.
    learning_rate : float, default=0.01
        Learning rate for weight updates.

    Attributes
    ----------
    layers : list of DenseLayer
        Sequence of dense layers.
    activations : list of str
        Activation functions for each layer.
    learning_rate : float
        Learning rate for weight updates.

    Examples
    --------
    >>> mlp = MultilayerPerceptron([(2, 3, "relu"), (3, 2, "softmax")], learning_rate=0.1)
    >>> X = np.array([[0,0],[1,1]])
    >>> preds = mlp.predict(X)
    """

    def __init__(self, layers, activations=None, learning_rate=0.01):
        # Allow either DenseLayer objects or specifications
        if all(isinstance(layer, tuple) for layer in layers):
            self.layers = [DenseLayer(*spec) for spec in layers]
            self.activations = [spec[2] for spec in layers]
        else:
            self.layers = layers
            if activations is None:
                activations = ["sigmoid"] * len(layers)
            if len(activations) != len(layers):
                raise ValueError("Number of activations must match number of layers")
            self.activations = activations

        self.learning_rate = learning_rate

        # Default loss: Negative Log Likelihood (vectorized)
        self.loss_fn = NegativeLogLikelihoodBatchConstructor()

    def forward(self, X):
        """Forward pass through the network."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        """Predict class labels for input data."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=100, verbose=True, stochastic=False, close_enough=None):
        """
        Train the network using gradient descent.

        Parameters
        ----------
        X : array_like
            Training features.
        y : array_like
            One-hot encoded training labels.
        epochs : int, default=100
            Number of training iterations.
        verbose : bool, default=True
            Whether to print loss during training.
        stochastic : bool, default=False
            If True, use stochastic gradient descent (SGD).
            If False, use batch gradient descent.
        close_enough : float, optional
            Early stopping threshold. If improvement in loss between epochs
            is less than this value, training stops.
        """
        X = _ensure_numeric_array(X, name="X")
        _ensure_no_nan(X, name="X")

        prev_loss = None
        for epoch in range(epochs):
            if stochastic:
                # SGD: update per sample
                for i in range(X.shape[0]):
                    xi = X[i:i+1]
                    yi = y[i:i+1]

                    y_pred = self.forward(xi)
                    loss = self.loss_fn(yi, y_pred).mean()

                    delta = y_pred - yi
                    for j in reversed(range(len(self.layers))):
                        layer = self.layers[j]
                        grads_W = np.dot(layer.last_input.T, delta)
                        grads_b = delta
                        layer.W -= self.learning_rate * grads_W
                        layer.b -= self.learning_rate * grads_b
                        if j > 0:
                            prev_out = self.layers[j - 1].last_output
                            delta = np.dot(delta, layer.W.T) * (prev_out > 0)
            else:
                # Batch GD
                y_pred = self.forward(X)
                loss = self.loss_fn(y, y_pred).mean()

                delta = y_pred - y
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    grads_W = np.dot(layer.last_input.T, delta) / X.shape[0]
                    grads_b = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
                    layer.W -= self.learning_rate * grads_W
                    layer.b -= self.learning_rate * grads_b
                    if i > 0:
                        prev_out = self.layers[i - 1].last_output
                        delta = np.dot(delta, layer.W.T) * (prev_out > 0)

            # Early stopping check
            if close_enough is not None and prev_loss is not None:
                if abs(prev_loss - loss) < close_enough:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
                    break
            prev_loss = loss

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def MSE(self, X, y_true, *, distance=None, sample_weight=None):
        """
        Compute mean squared error (MSE) for the network on given data.

        This method delegates to the standalone `mean_squared_error` function,
        passing the current model (`self`) as the predictor. It supports optional
        custom per-sample distance functions and sample weighting.

        Parameters
        ----------
        X : array_like
            Input features, shape (n_samples, ...).
        y_true : array_like
            Targets, shape (n_samples, ...).
        distance : callable, optional
            Custom distance function between y_true and predictions. If None,
            uses Euclidean norm of residuals.
        sample_weight : array_like, optional
            1D weights of length n_samples. If provided, errors are weighted.

        Returns
        -------
        float
            Mean squared error over the dataset.

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
        >>> mlp = Dummy()
        >>> X = np.array([[0],[1]])
        >>> y = np.array([[0],[1]])
        >>> round(mean_squared_error(mlp, X, y), 3)
        0.0

        """
        return mean_squared_error(self, X, y_true,
                                  distance=distance,
                                  sample_weight=sample_weight)



# Public API for this module
__all__ = ["MultilayerPerceptron"]



