import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import MSE, mean_squared_error
from ml.supervised_learning.multilayer_perceptron import DenseLayer, MultilayerPerceptron
from ml.utils.activations import APPROVED_ACTIVATIONS, relu, sigmoid

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
#confirms that works with custom input
def test_dense_forward_custom_relu():
    layer = DenseLayer(2, 3, activation=APPROVED_ACTIVATIONS["relu"])
    X = np.array([[1.0, -1.0]])
    out = layer.forward(X)
    assert out.shape == (1, 3)
    assert np.all(out >= 0)


def test_mlp_forward_and_predict_with_strings():
    # 2 -> 2 hidden -> 2 output, relu + softmax
    mlp = MultilayerPerceptron(
        layers=[(2, 2, "relu"), (2, 2, "sigmoid")],
        learning_rate=0.1,
        loss_fn="mse"
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = mlp.forward(X)
    assert out.shape == (2, 2)

    preds = mlp.predict(X)
    assert preds.shape == (2,)
    assert np.all((preds == 0) | (preds == 1))


def test_mlp_forward_and_predict_with_instances():

    mlp = MultilayerPerceptron(
        layers=[DenseLayer(2, 2, relu), DenseLayer(2, 2, sigmoid)],
        learning_rate=0.1,
        loss_fn=MSE  # direct Loss instance
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = mlp.forward(X)
    assert out.shape == (2, 2)

    preds = mlp.predict(X)
    assert preds.shape == (2,)
    assert np.all((preds == 0) | (preds == 1))


def test_mlp_fit_deterministic_manual_weights():
    # Simple dataset: 2 inputs, 1 output
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])  # XOR labels (not one-hot for simplicity)

    # Build a tiny MLP: 2 -> 2 hidden -> 1 output
    mlp = MultilayerPerceptron(
        layers=[(2, 2, "tanh"), (2, 1, "sigmoid")],
        learning_rate=0.1,
        loss_fn="mse"
    )

    # Manually set weights and biases to deterministic values
    for layer in mlp.layers:
        layer.W[:] = 0.5   # every weight = 0.5
        layer.b[:] = 0.0   # every bias = 0.0

    # --- Run 1 epoch deterministically ---
    y_pred = mlp.forward(X)
    initial_loss = mlp.loss_fn.mean_loss(y, y_pred)

    mlp.fit(X, y, epochs=1, verbose=False, stochastic=False)
    y_pred_after_1 = mlp.forward(X)
    loss_after_1 = mlp.loss_fn.mean_loss(y, y_pred_after_1)

    # Explicit check: loss should decrease deterministically
    assert loss_after_1 < initial_loss

    # --- Run 2 epochs deterministically ---
    mlp.fit(X, y, epochs=1, verbose=False, stochastic=False)
    y_pred_after_2 = mlp.forward(X)
    loss_after_2 = mlp.loss_fn.mean_loss(y, y_pred_after_2)

    # Check monotonic improvement
    assert loss_after_2 < loss_after_1


def test_mlp_fit_stochastic_moves_weights():
    X = np.array([[0,0],[1,1]])
    y = np.array([[1,0],[0,1]])

    mlp = MultilayerPerceptron(
        layers=[(2, 3, "relu"), (3, 2, "sigmoid")],
        learning_rate=0.1,
        loss_fn="mse"
    )

    # Capture initial weights
    initial_weights = [layer.W.copy() for layer in mlp.layers]

    # Run stochastic training
    mlp.fit(X, y, epochs=5, verbose=False, stochastic=True)

    # Ensure weights have changed
    for init, layer in zip(initial_weights, mlp.layers):
        assert not np.allclose(init, layer.W)

def test_mlp_invalid_loss_string():
    with pytest.raises(ValueError):
        MultilayerPerceptron(
            layers=[(2, 2, "relu")],
            loss_fn="not_a_loss"
        )

def test_mlp_invalid_activation_string():
    with pytest.raises(ValueError):
        MultilayerPerceptron(
            layers=[(2, 2, "not_an_activation")],
            loss_fn="mse"
        )

def test_mlp_empty_layers():
    with pytest.raises(ValueError):
        MultilayerPerceptron(layers=[], loss_fn="mse")


def test_mlp_constructor_with_empty_layer_raises():
    # Trying to build a layer with 0 outputs should fail immediately
    with pytest.raises(ValueError):
        MultilayerPerceptron(
            layers=[(2, 0, "relu"), (0, 1, "sigmoid")],
            learning_rate=0.1,
            loss_fn="mse"
        )

