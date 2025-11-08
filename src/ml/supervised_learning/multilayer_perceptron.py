import numpy as np
from ml.utils._errors_and_warnings.error_handling import (
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
        self.n_inputs = _ensure_positive_int("n_inputs", n_inputs)
        self.n_outputs = _ensure_positive_int("n_outputs", n_outputs)

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

    Attributes
    ----------
    layers : list of DenseLayer
        Sequence of dense layers.
    activations : list of str
        Activation functions for each layer. Defaults to "sigmoid" for all layers
        if not provided.
    learning_rate : float
        Learning rate for weight updates.

    Examples
    --------
    >>> import numpy as np
    >>> # Define a toy MLP with one dense layer; activations default to "sigmoid"
    >>> mlp = MultilayerPerceptron([DenseLayer(2, 2)], learning_rate=0.1)
    >>> X = np.array([[0,0],[1,1]])
    >>> preds = mlp.predict(X)
    >>> isinstance(preds, np.ndarray)
    True

    >>> # Explicitly specify activations if desired
    >>> mlp2 = MultilayerPerceptron([DenseLayer(2, 2)], activations=["sigmoid"], learning_rate=0.1)
    >>> preds2 = mlp2.predict(X)
    >>> isinstance(preds2, np.ndarray)
    True
    """

    def __init__(self, layers, activations=None, learning_rate=0.01):
        if activations is None:
            # Default all activations to "sigmoid"
            activations = ["sigmoid"] * len(layers)
        if len(activations) != len(layers):
            raise ValueError("Number of activations must match number of layers")

        self.layers = layers
        self.activations = activations
        self.learning_rate = learning_rate


    def forward(self, X):
        """
        Forward pass through the network.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray
            Network output.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : array_like
            Input data.

        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=100, verbose=True):
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
        """
        X = _ensure_numeric_array(X, name="X")
        _ensure_no_nan(X, name="X")

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))

            # Backpropagation
            delta = y_pred - y
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]

                # Gradients
                grads_W = np.dot(layer.last_input.T, delta) / X.shape[0]
                grads_b = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

                # Update weights
                layer.W -= self.learning_rate * grads_W
                layer.b -= self.learning_rate * grads_b

                # Propagate error backward (ReLU derivative for hidden layers)
                if i > 0:
                    prev_out = self.layers[i - 1].last_output
                    delta = np.dot(delta, layer.W.T) * (prev_out > 0)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def score(self, X, y_true, scoring_fn):
        """
        Evaluate the network using a scoring function.

        Parameters
        ----------
        X : array_like
            Input features.
        y_true : array_like
            True labels.
        scoring_fn : callable
            Function that computes a score given (y_true, y_pred).

        Returns
        -------
        float
            Score computed by scoring_fn.
        """
        y_pred = self.predict(X)
        return scoring_fn(y_true, y_pred)


# Public API for this module
__all__ = ["MultilayerPerceptron"]
