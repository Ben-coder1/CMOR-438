import numpy as np

from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan

class Activation:
    """
    Encapsulates an activation function and its derivative,
    with validation to ensure numeric outputs.

    Activation functions are applied after the linear transformation
    in neural networks. This class provides:
      - A callable interface for applying the activation.
      - A method for computing its derivative.
      - Validation to ensure outputs are numeric and shape-preserving.
    """


    def __init__(self, func, deriv, name=None):
        """
        Initialize an Activation object.

        Parameters
        ----------
        func : callable
            The activation function, mapping ndarray -> ndarray.
        deriv : callable
            The derivative of the activation function, mapping ndarray -> ndarray.
        name : str, optional
            A human-readable name for the activation. Defaults to func.__name__.
        """

        self.func = func
        self.deriv = deriv
        self.name = name or func.__name__

    def __call__(self, z):
        """
        Apply the activation function to input values.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values (linear transformation outputs).

        Returns
        -------
        np.ndarray
            Activation outputs, same shape as `z`.

        Raises
        ------
        ValueError
            If the output is not numeric or does not preserve input shape.
        """

        arr = _ensure_numeric_array(z, name="z")
        _ensure_no_nan(arr, name="z")
        out = self.func(arr)
        return self._validate_output(out, arr.shape, "activation")
        

    def gradient(self, z):
        """
        Compute the derivative of the activation function
        with respect to pre-activation values.

        Parameters
        ----------
        z : np.ndarray
            Pre-activation values.

        Returns
        -------
        np.ndarray
            Derivative values, same shape as `z`.

        Notes
        -----
        Custom derivatives must follow the convention of being defined
        with respect to the pre-activation input `z`.
        """


        arr = _ensure_numeric_array(z, name="z")
        _ensure_no_nan(arr, name="z")
        grad = self.deriv(arr)
        return self._validate_output(grad, arr.shape, "derivative")
        

    def _validate_output(self, out, expected_shape, kind):
        """
        Validate that an activation or derivative output is numeric,
        contains no NaNs, and preserves input shape.

        Parameters
        ----------
        out : np.ndarray
            Output array to validate.
        expected_shape : tuple
            Expected shape of the output.
        kind : str
            Description of the output type ("activation" or "derivative").

        Returns
        -------
        np.ndarray
            Validated output array.

        Raises
        ------
        ValueError
            If the output shape does not match the expected shape.
        """

        hold = _ensure_numeric_array(out, name=f"{kind} output")
        if hold.shape != expected_shape:
            raise ValueError(
                f"{kind.capitalize()} must preserve input shape. "
                f"Expected {expected_shape}, got {hold.shape}."
            )
        
        return _ensure_no_nan(out, name=f"{kind} output, function is f{self.name}")

    def __repr__(self):
        """
        Return a string representation of the Activation object.

        Returns
        -------
        str
            Representation including the activation name.
        """

        return f"Activation(name={self.name})"



# --- Sigmoid with error checks ---

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Element-wise sigmoid values in (0, 1).
    """

    return 1.0 / (1.0 + np.exp(-z))

def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array (pre-activation values).

    Returns
    -------
    np.ndarray
        Element-wise derivative values.
    """

    sig = _sigmoid(z)
    return sig * (1.0 - sig)

sigmoid = Activation(_sigmoid, _sigmoid_derivative, name="sigmoid")


# --- Tanh ---

def _tanh(z: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Element-wise tanh values in (-1, 1).
    """

    return np.tanh(z)

def _tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the tanh activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array (pre-activation values).

    Returns
    -------
    np.ndarray
        Element-wise derivative values.
    """

    return 1.0 - np.tanh(z) ** 2

tanh = Activation(_tanh, _tanh_derivative, name="tanh")


# --- ReLU ---

def _relu(z: np.ndarray) -> np.ndarray:
    """
    Compute the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Element-wise ReLU values: max(0, z).
    """

    return np.maximum(0, z)

def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array (pre-activation values).

    Returns
    -------
    np.ndarray
        Element-wise derivative values: 1 if z > 0, else 0.
    """

    return (z > 0).astype(float)

relu = Activation(_relu, _relu_derivative, name="relu")



# --- Softmax ---
def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the softmax activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array of shape (n_samples, n_outputs).

    Returns
    -------
    np.ndarray
        Softmax probabilities, same shape as z.
    """
    # subtract max for numerical stability
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the softmax activation function.

    Parameters
    ----------
    z : np.ndarray
        Input array (pre-activation values).

    Returns
    -------
    np.ndarray
        Derivative values. For simplicity, this returns the elementwise
        form softmax(z) * (1 - softmax(z)), which is correct for the
        diagonal entries of the Jacobian. In practice, when combined
        with cross-entropy loss, the gradient simplifies to
        softmax(z) - y, so you rarely need the full Jacobian.
    """
    s = _softmax(z)
    return s * (1.0 - s)  # diagonal approximation


softmax = Activation(_softmax, _softmax_derivative, name="softmax")


APPROVED_ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "softmax": softmax,
}

