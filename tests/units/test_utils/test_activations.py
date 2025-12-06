import numpy as np
import pytest
from ml.utils import activations
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError
from ml.utils.activations import Activation

def test_sigmoid_basic():
    z = np.array([-1.0, 0.0, 1.0])
    expected = 1 / (1 + np.exp(-z))
    result = activations.sigmoid(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
def test_sigmoid_derivative_basic():
    z = np.array([-1.0, 0.0, 1.0])
    sig = 1 / (1 + np.exp(-z))
    expected = sig * (1 - sig)
    result = activations.sigmoid.gradient(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
def test_sigmoid_non_numeric_input():
    with pytest.raises(TypeError):
        activations.sigmoid(["a", "b", "c"])

def test_sigmoid_nan_input():
    z = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        activations.sigmoid(z)
def test_sigmoid_derivative_non_numeric_input():
    with pytest.raises(TypeError):
        activations.sigmoid.gradient(["x", "y", "z"])
def test_sigmoid_derivative_nan_input():
    z = np.array([np.nan, 0.0, 1.0])
    with pytest.raises(ValueError):
        activations.sigmoid.gradient(z)
def test_sigmoid_output_shape():
    z = np.random.randn(5, 3)
    result = activations.sigmoid(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"
def test_sigmoid_derivative_output_shape():
    z = np.random.randn(4, 2)
    result = activations.sigmoid.gradient(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"

def test_sigmoid_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.sigmoid(z)
def test_sigmoid_derivative_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.sigmoid.gradient(z)


def test_tanh_basic():
    z = np.array([-1.0, 0.0, 1.0])
    expected = np.tanh(z)
    result = activations.tanh(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_tanh_derivative_basic():
    z = np.array([-1.0, 0.0, 1.0])
    tanh_val = np.tanh(z)
    expected = 1 - tanh_val**2
    result = activations.tanh.gradient(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_tanh_non_numeric_input():
    with pytest.raises(TypeError):
        activations.tanh(["a", "b", "c"])

def test_tanh_nan_input():
    z = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        activations.tanh(z)

def test_tanh_derivative_non_numeric_input():
    with pytest.raises(TypeError):
        activations.tanh.gradient(["x", "y", "z"])

def test_tanh_derivative_nan_input():
    z = np.array([np.nan, 0.0, 1.0])
    with pytest.raises(ValueError):
        activations.tanh.gradient(z)

def test_tanh_output_shape():
    z = np.random.randn(5, 3)
    result = activations.tanh(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"

def test_tanh_derivative_output_shape():
    z = np.random.randn(4, 2)
    result = activations.tanh.gradient(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"

def test_tanh_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.tanh(z)

def test_tanh_derivative_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.tanh.gradient(z)



def test_relu_basic():
    z = np.array([-1.0, 0.0, 1.0])
    expected = np.maximum(0, z)
    result = activations.relu(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_relu_derivative_basic():
    z = np.array([-1.0, 0.0, 1.0])
    expected = np.where(z > 0, 1.0, 0.0)
    result = activations.relu.gradient(z)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_relu_non_numeric_input():
    with pytest.raises(TypeError):
        activations.relu(["a", "b", "c"])

def test_relu_nan_input():
    z = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        activations.relu(z)

def test_relu_derivative_non_numeric_input():
    with pytest.raises(TypeError):
        activations.relu.gradient(["x", "y", "z"])

def test_relu_derivative_nan_input():
    z = np.array([np.nan, 0.0, 1.0])
    with pytest.raises(ValueError):
        activations.relu.gradient(z)

def test_relu_output_shape():
    z = np.random.randn(5, 3)
    result = activations.relu(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"

def test_relu_derivative_output_shape():
    z = np.random.randn(4, 2)
    result = activations.relu.gradient(z)
    assert result.shape == z.shape, f"Expected shape {z.shape}, got {result.shape}"

def test_relu_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.relu(z)

def test_relu_derivative_empty_input():
    z = np.array([])
    with pytest.raises(ValueError):
        activations.relu.gradient(z)


def test_activation_init_and_run_good():
    # Define a simple square activation
    def square(x): return np.asarray(x) ** 2
    def square_grad(x): return 2 * np.asarray(x)

    act = Activation(square, square_grad, name="square")
    z = np.array([1.0, 2.0, -3.0])
    result = act(z)
    grad = act.gradient(z)

    expected_result = np.array([1.0, 4.0, 9.0])
    expected_grad = np.array([2.0, 4.0, -6.0])

    assert np.allclose(result, expected_result)
    assert np.allclose(grad, expected_grad)
    assert act.name == "square"



def test_activation_forward_returns_non_numeric_error():
    def bad_forward(x): return np.array(["a", "b", "c"])  # non-numeric dtype
    def grad(x): return np.ones_like(x)

    act = Activation(bad_forward, grad, name="bad_forward_non_numeric")
    z = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        act(z)



def test_activation_gradient_returns_non_numeric_error():
    def forward(x): return np.asarray(x)
    def bad_grad(x): return np.array(["x", "y", "z"])  # non-numeric dtype

    act = Activation(forward, bad_grad, name="bad_grad_non_numeric")
    z = np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        act.gradient(z)


def test_softmax_forward_probabilities_sum_to_one():
    z = np.array([[1.0, 2.0, 3.0],
                  [0.1, 0.2, 0.3]])
    out = activations.softmax(z)
    # Each row should sum to 1
    row_sums = np.sum(out, axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums)), "Softmax outputs must sum to 1"
    # Shape preserved
    assert out.shape == z.shape


def test_softmax_forward_numerical_stability():
    z = np.array([[1000.0, 1001.0, 1002.0]])  # large values
    out = activations.softmax(z)
    # Should not produce NaNs or infs
    assert np.all(np.isfinite(out)), "Softmax must be numerically stable"
    # Still sums to 1
    assert np.isclose(np.sum(out), 1.0)


def test_softmax_derivative_shape_and_values():
    z = np.array([[1.0, 2.0, 3.0]])
    grad = activations.softmax.gradient(z)
    # Shape preserved
    assert grad.shape == z.shape
    # Values between 0 and 1
    assert np.all((grad >= 0) & (grad <= 1)), "Softmax derivative values must be in [0,1]"


def test_softmax_in_activation_class_repr_and_call():
    z = np.array([[0.5, 1.5]])
    act = activations.softmax(z)
    grad = activations.softmax.gradient(z)
    # Check repr
    assert "softmax" in repr(activations.softmax)
    # Forward output sums to 1
    assert np.isclose(np.sum(act), 1.0)
    # Gradient shape preserved
    assert grad.shape == z.shape


