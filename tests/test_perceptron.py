import numpy as np
from ml.supervised_learning.perceptron import Perceptron
import pytest

# --- Construction tests ---
# Valid construction
p = Perceptron(n_features=3)
assert p.weights.shape == (3,)
assert isinstance(p.bias, float)

# Invalid n_features
try:
    Perceptron(n_features=0)
    assert False, "Expected ValueError for n_features <= 0"
except ValueError:
    pass

try:
    Perceptron(n_features=-5)
    assert False, "Expected ValueError for negative n_features"
except ValueError:
    pass

try:
    Perceptron(n_features="two")
    assert False, "Expected ValueError for non-int n_features"
except ValueError:
    pass

# --- Label encoding/decoding ---
p = Perceptron(n_features=2)
y = np.array(["cat", "dog", "cat"])
enc = p._encode_labels(y)
assert set(enc) <= {-1, 1}
p.classes_ = np.array(["cat", "dog"])
dec = p._decode_labels(np.array([-1, 1]))
assert dec.tolist() == ["cat", "dog"]

# Trigger error: more than 2 classes
try:
    p._encode_labels(np.array(["a", "b", "c"]))
    assert False, "Expected ValueError for >2 classes"
except ValueError:
    pass

# Trigger error: decode before training
p2 = Perceptron(n_features=2)
try:
    p2._decode_labels(np.array([-1, 1]))
    assert False, "Expected RuntimeError for decode without training"
except RuntimeError:
    pass

# --- Activation ---
p = Perceptron(n_features=2)
x = np.array([1.0, 2.0])
assert isinstance(p._activation(x), float)

# --- Predict ---
X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1])
p = Perceptron(n_features=2)
p.train(X, y, epochs=5, seed=0)
preds = p.predict(X)
assert set(preds.tolist()) <= {0, 1}

# Trigger error: wrong shape
try:
    p.predict(np.array([1.0, 2.0]))  # missing sample axis
    assert False, "Expected ValueError for wrong shape"
except ValueError:
    pass

# --- Score ---
acc = p.score(X, y)
assert 0.0 <= acc <= 1.0

# --- Train ---
# Float data
Xf = np.array([[0.0, 0.0], [1.0, 1.0]])
yf = np.array([0.0, 1.0])
p = Perceptron(n_features=2)
p.train(Xf, yf, lr=0.5, epochs=3, seed=42, verbose=False)

# Int data
Xi = np.array([[0, 0], [1, 1]])
yi = np.array([0, 1])
p = Perceptron(n_features=2)
p.train(Xi, yi, lr=1, epochs=2)

# Double precision data
Xd = np.array([[0, 0], [1, 1]], dtype=np.float64)
yd = np.array([0, 1])
p = Perceptron(n_features=2)
p.train(Xd, yd, lr=0.1, epochs=2)

def test_perceptron_nonstochastic_single_epoch_expected_update():
    """
    In non-stochastic mode, after 1 epoch the weights and bias should equal
    the batch update computed from misclassified samples.
    """
    # Simple dataset: two samples, labels 0 and 1
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])

    p = Perceptron(n_features=2)

    # Initialize RNG and weights exactly as train() will
    p._rng = np.random.RandomState(0)
    init_w = p._rng.normal(loc=0.0, scale=0.01, size=p.w.shape)
    init_b = 0.0
    p.w = init_w.copy()
    p.b = init_b

    # Train for 1 epoch, non-stochastic
    p.train(X, y, lr=1.0, epochs=1, seed=0, stochastic=False, verbose=False)

    # Encode labels manually to {-1, +1}
    y_internal = p._encode_labels(y)

    # Compute expected update from misclassified samples
    scores = np.dot(X, init_w) + init_b
    preds = np.where(scores >= 0.0, 1, -1)
    mis_idx = np.where(preds != y_internal)[0]

    expected_w = init_w.copy()
    expected_b = init_b
    if mis_idx.size > 0:
        expected_w += np.sum(y_internal[mis_idx, None] * X[mis_idx], axis=0)
        expected_b += np.sum(y_internal[mis_idx])

    # Assert weights and bias match expected
    assert np.allclose(p.w, expected_w)
    assert np.isclose(p.b, expected_b)



def test_perceptron_learns_and_function():
    """Perceptron should learn the AND function with two features."""
    # AND truth table
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])

    p = Perceptron(n_features=2)
    p.train(X, y, lr=0.1, epochs=20, seed=42, verbose=False)

    # After training, accuracy should be perfect
    assert p.score(X, y) == pytest.approx(1.0)
    # Weights should not remain all zeros
    assert not np.allclose(p.weights, 0.0)


def test_perceptron_updates_weights():
    """Training should change weights and bias from initialization."""
    X = np.array([[0,0],[1,1]])
    y = np.array([0,1])

    p = Perceptron(n_features=2)
    initial_w = p.weights.copy()
    initial_b = p.bias

    p.train(X, y, lr=0.5, epochs=5, seed=0, verbose=False)

    # Weights and bias should be updated
    assert not np.allclose(p.weights, initial_w)
    assert p.bias != initial_b
    # Model should classify training set correctly
    preds = p.predict(X)
    assert set(preds.tolist()) == {0,1}


# Trigger error: mismatched sample sizes
try:
    p.train(np.array([[0, 0], [1, 1]]), np.array([0]))
    assert False, "Expected ValueError for mismatched sample sizes"
except ValueError:
    pass

# Trigger error: wrong feature dimension
try:
    p.train(np.array([[0, 0, 0]]), np.array([0]))
    assert False, "Expected ValueError for wrong feature dimension"
except ValueError:
    pass

# --- Reset ---
p = Perceptron(n_features=2)
p.train(X, y, epochs=1, seed=0)
old_w = p.weights.copy()
p.reset(seed=1)
assert (p.weights != old_w).any()
assert p.history == []
