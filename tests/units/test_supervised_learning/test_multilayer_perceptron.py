import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.supervised.performance import mean_squared_error
from ml.supervised_learning.multilayer_perceptron import DenseLayer, MultilayerPerceptron

def test_dense_forward_basic():
    layer = DenseLayer(2, 3, activation="relu")
    X = np.array([[1.0, -1.0]])
    out = layer.forward(X)
    assert out.shape == (1, 3)
    assert np.all(out >= 0)

def test_dense_forward_list_input():
    layer = DenseLayer(2, 2, activation="sigmoid")
    out = layer.forward([[1, 2]])
    assert out.shape == (1, 2)

def test_dense_forward_empty_input():
    layer = DenseLayer(2, 2, activation="sigmoid")
    with pytest.raises(ValueError):
        layer.forward([])

def test_dense_forward_nan_input():
    layer = DenseLayer(2, 2, activation="sigmoid")
    X = np.array([[np.nan, 1]])
    with pytest.raises(ValueError):
        layer.forward(X)

def test_invalid_activation_string():
    with pytest.raises(ValueError):
        DenseLayer(2, 2, activation="not_a_function")

def test_custom_activation_function():
    # Weird activation: square the input then subtract 0.5
    def weird_activation(x):
        return np.square(x) - 0.5

    layer = DenseLayer(2, 2, activation=weird_activation)
    X = np.array([[1.0, -2.0]])

    # Compute forward output
    out = layer.forward(X)

    # Manually compute the linear part
    linear_out = X @ layer.W + layer.b

    # Apply weird_activation manually
    expected = weird_activation(linear_out)

    # Compare
    assert np.allclose(out, expected, rtol=1e-5, atol=1e-5)




#tests for multilayer perceptron 

def test_mlp_forward_and_predict():
    mlp = MultilayerPerceptron([(2, 3, "relu"), (3, 2, "softmax")])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = mlp.forward(X)
    assert out.shape == (2, 2)
    preds = mlp.predict(X)
    assert preds.shape == (2,)
    assert np.all((preds == 0) | (preds == 1))

def test_mlp_forward_list_input():
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")])
    out = mlp.forward([[1, 2]])
    assert out.shape == (1, 2)

def test_mlp_forward_empty_input():
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")])
    with pytest.raises(ValueError):
        mlp.forward([])

def test_mlp_forward_nan_input():
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")])
    X = np.array([[np.nan, 1]])
    with pytest.raises(ValueError):
        mlp.forward(X)

def test_mlp_fit_batch_mode_close_enough():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])  # XOR labels
    mlp = MultilayerPerceptron([(2, 3, "relu"), (3, 2, "softmax")], learning_rate=0.1)
    mlp.fit(X, y, epochs=50, verbose=False, close_enough=1e-6)
    preds = mlp.predict(X)
    assert preds.shape == (4,)

def test_mlp_fit_stochastic_mode_moves_weights():
    X = np.array([[0,0],[1,1]])
    y = np.array([[1,0],[0,1]])
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")], learning_rate=0.1)
    initial_out = mlp.forward(X)
    mlp.fit(X, y, epochs=5, stochastic=True, verbose=False)
    new_out = mlp.forward(X)
    assert not np.allclose(initial_out, new_out)

def test_mlp_invalid_activations_length():
    with pytest.raises(ValueError):
        MultilayerPerceptron([DenseLayer(2,2,"sigmoid")], activations=["sigmoid","relu"])

def test_mlp_MSE_functionality():
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")])
    X = np.array([[0, 0], [1, 1]])
    y = np.array([[0], [1]])

    # Call the new MSE method
    mse_value = mlp.MSE(X, y)

    # It should be a float
    assert isinstance(mse_value, float)

    # And it should match the standalone mean_squared_error function
    expected = mean_squared_error(mlp, X, y)
    assert np.isclose(mse_value, expected, rtol=1e-8, atol=1e-8)


def test_mlp_init_with_dense_objects():
    layers = [DenseLayer(2, 2, activation="sigmoid"), DenseLayer(2, 1, activation="sigmoid")]
    mlp = MultilayerPerceptron(layers)
    X = np.array([[1,0]])
    out = mlp.forward(X)
    assert out.shape == (1,1)

def test_mlp_fit_verbose_runs():
    X = np.array([[0,0],[1,1]])
    y = np.array([[1,0],[0,1]])
    mlp = MultilayerPerceptron([(2, 2, "sigmoid")], learning_rate=0.1)
    # Just ensure verbose doesn't break
    mlp.fit(X, y, epochs=2, verbose=True)

