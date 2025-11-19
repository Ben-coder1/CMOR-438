
import numpy as np
import pytest

from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import Loss, MSE, MAE, CrossEntropyLossConstructor


def test_mse_basic_positive_negative():
    y_true = np.array([[1.0, -1.0], [2.0, 3.0]])
    y_pred = np.array([[0.0, -2.0], [2.5, 2.5]])
    # residuals: [[1,-1] - [0,-2]] = [[1,1],[ -0.5,0.5]]
    residual = y_true - y_pred
    expected = np.mean(residual**2, axis=1)
    result = MSE(y_true, y_pred)
    assert np.allclose(result, expected)

def test_mse_int_inputs():
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[1, 1], [4, 4]])
    result = MSE(y_true, y_pred)
    assert result.shape == (2,)
    assert np.all(result >= 0)

def test_mse_empty_input():
    y_true = np.array([]).reshape(0, 2)
    y_pred = np.array([]).reshape(0, 2)
    with pytest.raises(ValueError):
        MSE(y_true, y_pred)

def test_mse_nan_input():
    y_true = np.array([[1.0, np.nan]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        MSE(y_true, y_pred)

def test_mse_string_input():
    y_true = np.array([["a", "b"]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(TypeError):
        MSE(y_true, y_pred)

def test_mse_gradient_shape_and_values():
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[2.0, 4.0]])
    grad = MSE.gradient(y_true, y_pred)
    assert grad.shape == y_pred.shape
    expected = 2.0 * (y_pred - y_true) / y_true.shape[1]
    assert np.allclose(grad, expected)
def test_mse_gradient_nan_input():
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[np.nan, 4.0]])
    with pytest.raises(ValueError):
        MSE.gradient(y_true, y_pred)
def test_mse_gradient_string_input():
    y_true = np.array([["a", "b"]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(TypeError):
        MSE.gradient(y_true, y_pred)

def test_mae_basic_positive_negative():
    y_true = np.array([[1.0, -1.0], [2.0, 3.0]])
    y_pred = np.array([[0.0, -2.0], [2.5, 2.5]])
    residual = y_true - y_pred
    expected = np.mean(np.abs(residual), axis=1)
    result = MAE(y_true, y_pred)
    assert np.allclose(result, expected)

def test_mae_int_inputs():
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[1, 1], [4, 4]])
    result = MAE(y_true, y_pred)
    assert result.shape == (2,)
    assert np.all(result >= 0)

def test_mae_empty_input():
    y_true = np.array([]).reshape(0, 2)
    y_pred = np.array([]).reshape(0, 2)
    with pytest.raises(ValueError):
        MAE(y_true, y_pred)

def test_mae_nan_input():
    y_true = np.array([[1.0, np.nan]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        MAE(y_true, y_pred)

def test_mae_string_input():
    y_true = np.array([["a", "b"]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(TypeError):
        MAE(y_true, y_pred)

def test_mae_gradient_shape_and_values():
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[2.0, 4.0]])
    grad = MAE.gradient(y_true, y_pred)
    assert grad.shape == y_pred.shape
    expected = np.sign(y_pred - y_true) / y_true.shape[1]
    assert np.allclose(grad, expected)

def test_cross_entropy_basic():
    ce_loss = CrossEntropyLossConstructor()
    Y_true = np.array([[0,1,0],[1,0,0]])
    Y_pred = np.array([[0.2,0.7,0.1],[0.9,0.05,0.05]])
    result = ce_loss(Y_true, Y_pred)
    expected = np.array([0.35667494, 0.10536052])
    assert np.allclose(result, expected)

def test_cross_entropy_gradient_shape_and_values():
    ce_loss = CrossEntropyLossConstructor()
    Y_true = np.array([[0,1,0],[1,0,0]])
    Y_pred = np.array([[0.2,0.7,0.1],[0.9,0.05,0.05]])
    grad = ce_loss.gradient(Y_true, Y_pred)
    assert grad.shape == Y_pred.shape
    expected = -(Y_true / np.clip(Y_pred, 1e-8, 1.0-1e-8))
    assert np.allclose(grad, expected)

def test_cross_entropy_empty_input():
    ce_loss = CrossEntropyLossConstructor()
    Y_true = np.array([]).reshape(0, 3)
    Y_pred = np.array([]).reshape(0, 3)
    with pytest.raises(ValueError):
        ce_loss(Y_true, Y_pred)

def test_cross_entropy_nan_input():
    ce_loss = CrossEntropyLossConstructor()
    Y_true = np.array([[1.0, np.nan, 0.0]])
    Y_pred = np.array([[0.2, 0.7, 0.1]])
    with pytest.raises(ValueError):
        ce_loss(Y_true, Y_pred)

def test_cross_entropy_string_input():
    ce_loss = CrossEntropyLossConstructor()
    Y_true = np.array([["a","b","c"]])
    Y_pred = np.array([[0.2,0.7,0.1]])
    with pytest.raises(TypeError):
        ce_loss(Y_true, Y_pred)

def test_cross_entropy_invalid_epsilon():
    with pytest.raises(ValueError):
        CrossEntropyLossConstructor(epsilon=-1.0)



#Class tests


def test_loss_constructor_and_call_good():
    # Define a simple squared error loss
    def func(y_true, y_pred):
        residual = y_true - y_pred
        return np.mean(residual**2, axis=1)

    def grad(y_true, y_pred):
        return 2.0 * (y_pred - y_true) / y_true.shape[1]

    loss = Loss(func, grad, name="squared_error")

    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.5, 2.5], [2.5, 3.5]])

    # __call__ returns per-sample scalars
    result = loss(y_true, y_pred)
    assert result.shape == (2,)
    assert np.all(result >= 0)

    # gradient returns same shape as y_pred
    grad_result = loss.gradient(y_true, y_pred)
    assert grad_result.shape == y_pred.shape

    # mean_loss aggregates correctly
    mean_val = loss.mean_loss(y_true, y_pred)
    assert np.isclose(mean_val, np.mean(result))


def test_loss_output_not_numpy_array():
    def bad_func(y_true, y_pred):
        return [1.0, 2.0]  # returns list, not np.ndarray
    def grad(y_true, y_pred):
        return np.zeros_like(y_pred)

    loss = Loss(bad_func, grad, name="bad_loss")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss(y_true, y_pred)


def test_loss_output_wrong_shape():
    def bad_func(y_true, y_pred):
        return np.array([[1.0, 2.0]])  # wrong shape
    def grad(y_true, y_pred):
        return np.zeros_like(y_pred)

    loss = Loss(bad_func, grad, name="bad_shape")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss(y_true, y_pred)


def test_loss_output_contains_nan():
    def bad_func(y_true, y_pred):
        return np.array([np.nan])
    def grad(y_true, y_pred):
        return np.zeros_like(y_pred)

    loss = Loss(bad_func, grad, name="nan_loss")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss(y_true, y_pred)


def test_gradient_not_numpy_array():
    def func(y_true, y_pred):
        return np.mean((y_true - y_pred)**2, axis=1)
    def bad_grad(y_true, y_pred):
        return [[1.0, 2.0]]  # list, not np.ndarray

    loss = Loss(func, bad_grad, name="bad_grad")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss.gradient(y_true, y_pred)


def test_gradient_wrong_shape():
    def func(y_true, y_pred):
        return np.mean((y_true - y_pred)**2, axis=1)
    def bad_grad(y_true, y_pred):
        return np.array([1.0, 2.0])  # wrong shape

    loss = Loss(func, bad_grad, name="bad_grad_shape")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss.gradient(y_true, y_pred)


def test_gradient_contains_nan():
    def func(y_true, y_pred):
        return np.mean((y_true - y_pred)**2, axis=1)
    def bad_grad(y_true, y_pred):
        return np.array([[np.nan, 0.0]])

    loss = Loss(func, bad_grad, name="nan_grad")
    y_true = np.array([[1.0, 2.0]])
    y_pred = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        loss.gradient(y_true, y_pred)
