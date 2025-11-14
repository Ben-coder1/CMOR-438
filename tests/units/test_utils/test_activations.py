import numpy as np
import pytest
from ml.utils import activations
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError

# --- Sigmoid tests ---
def test_sigmoid_basic_and_extremes():
    z = np.array([-10.0, -1.5, 0.0, 1.5, 10.0])
    out = activations.sigmoid(z)
    # Values should be between 0 and 1
    assert np.all((out > 0) & (out < 1))
    # Symmetry: sigmoid(-x) = 1 - sigmoid(x)
    assert np.allclose(out[1], 1 - out[3], atol=1e-6)

def test_sigmoid_empty_input_raises():
    with pytest.raises(ValueError):
        activations.sigmoid([])

def test_sigmoid_nan_input_raises():
    arr = np.array([np.nan, 1.0])
    with pytest.raises(ValueError):
        activations.sigmoid(arr)


# --- ReLU tests ---
def test_relu_negatives_and_floats():
    z = np.array([-2.5, -0.1, 0.0, 0.1, 3.7])
    out = activations.relu(z)
    # Negative inputs should map to 0
    assert out[0] == 0.0 and out[1] == 0.0
    # Positive floats should pass through
    assert np.allclose(out[-2:], [0.1, 3.7])

def test_relu_empty_input_raises():
    with pytest.raises(ValueError):
        activations.relu([])


# --- Tanh tests ---
def test_tanh_negatives_and_floats():
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    out = activations.tanh(z)
    # Odd symmetry: tanh(-x) = -tanh(x)
    assert np.allclose(out[1], -out[3], atol=1e-6)
    # Values should be bounded between -1 and 1
    assert np.all((out > -1) & (out < 1))

def test_tanh_non_numeric_raises():
    with pytest.raises(TypeError):
        activations.tanh(["a", "b"])


# --- Softmax tests ---
def test_softmax_row_sums_to_one_with_negatives_and_floats():
    z = np.array([[1.5, -2.0, 0.3], [0.1, 0.2, 0.3]])
    out = activations.softmax(z)
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums))

def test_softmax_accepts_1d_input():
    z = np.array([-1.0, 0.0, 1.0])
    out = activations.softmax(z)
    assert out.shape == (1, 3)
    assert np.allclose(out.sum(), 1.0)

def test_softmax_empty_input_raises():
    with pytest.raises(ValueError):
        activations.softmax([])

def test_softmax_invalid_ndim_raises():
    z = np.ones((2, 2, 2))  # 3D input
    with pytest.raises(InputShapeError):
        activations.softmax(z)

def test_softmax_nan_input_raises():
    z = np.array([[1.0, np.nan, 2.0]])
    with pytest.raises(ValueError):
        activations.softmax(z)

def test_step_negatives_zero_positives():
    z = np.array([-2.5, -0.1, 0.0, 0.1, 3.7])
    out = activations.step(z)
    # Negative inputs should map to 0
    assert np.all(out[:2] == 0.0)
    # Zero and positives should map to 1
    assert out[2] == 1.0 and out[3] == 1.0 and out[4] == 1.0

def test_step_integer_and_float_inputs():
    z = np.array([-1, 0, 2.5])
    out = activations.step(z)
    assert np.array_equal(out, np.array([0.0, 1.0, 1.0]))

def test_step_empty_input_raises():
    with pytest.raises(ValueError):
        activations.step([])

def test_step_non_numeric_input_raises():
    with pytest.raises(TypeError):
        activations.step(["a", "b"])
