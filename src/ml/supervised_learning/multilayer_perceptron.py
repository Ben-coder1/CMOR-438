import numpy as np
from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import APPROVED_LOSSES, mean_squared_error
from ml.metrics_and_evaluations.evaluation.supervised.performance import classification_accuracy
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array,
    _ensure_no_nan,
    _ensure_positive_int,
    InputShapeError
)
from ml.utils import activations, APPROVED_ACTIVATIONS
from ml.utils._errors_and_warnings.activations_and_losses_specific import _validate_activation, _validate_loss_fn


class DenseLayer:
    """
    A single fully-connected (dense) neural network layer.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_outputs : int
        Number of output units.
    activation : str or Activation, optional
        Activation function. Defaults to 'sigmoid'.
        Can be a string name of a pre-approved activation
        or a custom Activation object.

    Attributes
    ----------
    W : np.ndarray
        Weight matrix of shape (n_inputs, n_outputs).
    b : np.ndarray
        Bias vector of shape (1, n_outputs).
    activation : Activation
        Activation object with func and gradient.
    last_input : np.ndarray
        Cached input from last forward pass.
    last_z : np.ndarray
        Cached pre-activation values.
    last_output : np.ndarray
        Cached output from last forward pass.
    """

    def __init__(self, n_inputs, n_outputs, activation="sigmoid"):
        self.n_inputs = _ensure_positive_int(n_inputs, "n_inputs")
        self.n_outputs = _ensure_positive_int(n_outputs, "n_outputs")

        # Initialize weights and biases with appropriate scaling
        scale = np.sqrt(2.0 / n_inputs) if activation == "relu" else np.sqrt(1.0 / n_inputs)
        self.W = np.random.randn(n_inputs, n_outputs) * scale
        self.b = np.zeros((1, n_outputs))

        # Validate activation
        self.activation = _validate_activation(activation, APPROVED_ACTIVATIONS)

        # Cache for forward pass
        self.last_input = None
        self.last_z = None
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

        if X.shape[1] != self.W.shape[0]:
            raise InputShapeError(
                f"Input has {X.shape[1]} features, expected {self.W.shape[0]}."
            )

        # Linear transformation + activation
        z = np.dot(X, self.W) + self.b
        a = self.activation(z)

        # Cache for backprop
        self.last_input = X
        self.last_z = z
        self.last_output = a
        return a



class MultilayerPerceptron:
    """
        A fully connected feedforward neural network (dense architecture).

        This implementation supports arbitrary layer specifications, configurable
        learning rate, and pluggable loss functions. Layers can be provided either
        as tuples of specifications (to construct DenseLayer objects internally)
        or as pre-built DenseLayer instances.

        Parameters
        ----------
        layers : list
            Either a list of DenseLayer instances or a list of tuples specifying
            layer construction arguments for DenseLayer.
        learning_rate : float, optional
            Step size for gradient descent updates. Default is 0.01.
        loss_fn : Loss or str, optional
            Loss function to optimize. Must be either:
            - None (defaults to Negative Log Likelihood),
            - A Loss instance,
            - A string key referring to an entry in APPROVED_LOSSES.

        Attributes
        ----------
        layers : list of DenseLayer
            The sequence of layers in the network.
        learning_rate : float
            Step size for parameter updates.
        loss_fn : Loss
            The loss function object used for training.
        """


    def __init__(self, layers, learning_rate=0.01, loss_fn=None):
        """
    Initialize a MultilayerPerceptron.

    Parameters
    ----------
    layers : list
        Either:
        - A list of DenseLayer instances, or
        - A list of tuples specifying DenseLayer construction arguments.
          Each tuple is passed to DenseLayer(*spec).
    learning_rate : float, optional
        Step size for gradient descent updates. Default is 0.01.
    loss_fn : Loss or str, optional
        Loss function to optimize. If None, defaults to Negative Log Likelihood.
        If a string is provided, it must be a key in APPROVED_LOSSES.
        If a Loss instance is provided, it will be validated.

    Raises
    ------
    TypeError
        If loss_fn is neither a Loss instance nor a string.
    ValueError
        If a string loss_fn is not found in APPROVED_LOSSES.

    Notes
    -----
    After initialization, `self.loss_fn` is always a Loss object, ensuring
    consistent usage in training and evaluation.

    Examples
    --------
    >>> mlp = MultilayerPerceptron(
    ...     layers=[(2, 4, "relu"), (4, 1, "sigmoid")],
    ...     learning_rate=0.05,
    ...     loss_fn="MAE"
    ... )
    """
        if not layers or len(layers) == 0:
            raise ValueError("MultilayerPerceptron must be constructed with at least one layer.")
        if all(isinstance(layer, tuple) for layer in layers):
            # Build layers from specs
            self.layers = [DenseLayer(*spec) for spec in layers]
        else:
            self.layers = layers

        self.learning_rate = learning_rate
        if loss_fn is None:
            self.loss_fn = APPROVED_LOSSES["cross_entropy"]
        else:
            self.loss_fn = _validate_loss_fn(loss_fn, APPROVED_LOSSES)

    def forward(self, X):
        """
        Compute the forward pass through the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        out : ndarray
            Network output after passing through all layers.
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
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class indices.
        """

        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=100, verbose=True, stochastic=False, close_enough=None):
        """
        Train the network using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input data.
        y : ndarray of shape (n_samples, n_classes)
            One-hot encoded target labels.
        epochs : int, optional
            Number of training iterations. Default is 100.
        verbose : bool, optional
            If True, prints loss updates during training. Default is True.
        stochastic : bool, optional
            If True, performs stochastic gradient descent (SGD). Default is False.
        close_enough : float, optional
            Early stopping threshold. If the change in loss between epochs
            is smaller than this value, training stops early.

        Notes
        -----
        - Supports both batch gradient descent and stochastic gradient descent.
        - Early stopping is triggered if `close_enough` is provided and the
          loss improvement falls below the threshold.
        """

        X = _ensure_numeric_array(X, name="X")
        _ensure_no_nan(X, name="X")

        prev_loss = None
        for epoch in range(epochs):
            if stochastic:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)

                for i in indices:
                    xi = X[i:i+1]
                    yi = y[i:i+1]

                    y_pred = self.forward(xi)
                    loss = self.loss_fn(yi, y_pred).mean()
                    delta = self.loss_fn.gradient(yi, y_pred)

                    # Backpropagation
                    for j in reversed(range(len(self.layers))):
                        layer = self.layers[j]
                        grads_W = np.dot(layer.last_input.T, delta)
                        grads_b = np.sum(delta, axis=0, keepdims=True)
                        layer.W -= self.learning_rate * grads_W
                        layer.b -= self.learning_rate * grads_b

                        if j > 0:
                            prev_z = self.layers[j - 1].last_z
                            deriv = self.layers[j - 1].activation.gradient(prev_z)
                            delta = np.dot(delta, layer.W.T) * deriv
            else:
                y_pred = self.forward(X)
                loss = self.loss_fn(y, y_pred).mean()
                delta = self.loss_fn.gradient(y, y_pred)

                # Backpropagation
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    grads_W = np.dot(layer.last_input.T, delta) / X.shape[0]
                    grads_b = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
                    layer.W -= self.learning_rate * grads_W
                    layer.b -= self.learning_rate * grads_b

                    if i > 0:
                        prev_z = self.layers[i - 1].last_z
                        deriv = self.layers[i - 1].activation.gradient(prev_z)
                        delta = np.dot(delta, layer.W.T) * deriv

            # Early stopping
            if close_enough is not None and prev_loss is not None:
                if abs(prev_loss - loss) < close_enough:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
                    break
            prev_loss = loss

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")





    def score(self, X, y):
        """
        Compute classification accuracy of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly predicted samples.
        """

        return classification_accuracy(self, X, y)



# Public API for this module
__all__ = ["MultilayerPerceptron"]



