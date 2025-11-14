import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.unsupervised.clustering.centroids.intertia import compute_inertia

# Normal cases
# ----------------------

def test_compute_inertia_basic_ints():
    X = np.array([[0, 0], [1, 1]])
    centroids = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 1])
    inertia = compute_inertia(X, centroids, labels)
    assert np.isclose(inertia, 0.0)

def test_compute_inertia_with_floats():
    X = np.array([[0.5, 1.5], [2.5, 3.5]])
    centroids = np.array([[0.5, 1.5], [2.5, 3.5]])
    labels = np.array([0, 1])
    inertia = compute_inertia(X, centroids, labels)
    assert np.isclose(inertia, 0.0)

def test_compute_inertia_with_negatives():
    X = np.array([[-1, -2], [-3, -4]])
    centroids = np.array([[-1, -2], [-3, -4]])
    labels = np.array([0, 1])
    inertia = compute_inertia(X, centroids, labels)
    assert np.isclose(inertia, 0.0)

def test_compute_inertia_with_double_precision():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    labels = np.array([0, 1])
    inertia = compute_inertia(X, centroids, labels)
    assert isinstance(inertia, float)
    assert np.isclose(inertia, 0.0)


# ----------------------
# Edge cases and errors
# ----------------------

def test_compute_inertia_labels_length_mismatch():
    X = np.array([[0, 0], [1, 1]])
    centroids = np.array([[0, 0], [1, 1]])
    labels = np.array([0])  # wrong length
    with pytest.raises(ValueError, match="X and labels must have the same length"):
        compute_inertia(X, centroids, labels)

def test_compute_inertia_empty_X_raises():
    X = np.empty((0, 2))
    centroids = np.array([[0, 0]])
    labels = np.array([])
    with pytest.raises(ValueError):
        compute_inertia(X, centroids, labels)

def test_compute_inertia_empty_centroids_raises():
    X = np.array([[0, 0]])
    centroids = np.empty((0, 2))
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_inertia(X, centroids, labels)

def test_compute_inertia_none_input_raises():
    with pytest.raises(ValueError):
        compute_inertia(None, np.array([[0, 0]]), np.array([0]))

def test_compute_inertia_wrong_shape_X():
    X = np.array([1, 2, 3])  # 1D instead of 2D
    centroids = np.array([[0, 0]])
    labels = np.array([0])
    with pytest.raises(ValueError):
        compute_inertia(X, centroids, labels)

def test_compute_inertia_wrong_shape_labels():
    X = np.array([[0, 0], [1, 1]])
    centroids = np.array([[0, 0], [1, 1]])
    labels = np.array([[0, 1]])  # 2D instead of 1D
    with pytest.raises(ValueError):
        compute_inertia(X, centroids, labels)


# ----------------------
# Distance function errors
# ----------------------

def test_compute_inertia_distance_func_not_callable():
    X = np.array([[0, 0]])
    centroids = np.array([[0, 0]])
    labels = np.array([0])
    with pytest.raises(TypeError):
        compute_inertia(X, centroids, labels, distance_func="not_callable")

def test_compute_inertia_distance_func_returns_non_numeric():
    def bad_distance(x, y):
        return "not a number"
    X = np.array([[0, 0]])
    centroids = np.array([[0, 0]])
    labels = np.array([0])
    with pytest.raises(TypeError, match="distance_func must return a numeric scalar"):
        compute_inertia(X, centroids, labels, distance_func=bad_distance)

def test_compute_inertia_distance_func_returns_mixed_types():
    def weird_distance(x, y):
        # Return string for first call, int for second
        if np.all(x == y):
            return "oops"
        return 1
    X = np.array([[0, 0], [1, 1]])
    centroids = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        compute_inertia(X, centroids, labels, distance_func=weird_distance)


# ----------------------
# Special case: LinfinityDistance
# ----------------------

def test_compute_inertia_with_linfinity_distance():
    def LinfinityDistance(x, y):
        return np.max(np.abs(x - y))
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    centroids = np.array([[0.0, 0.0], [1.0, 2.0]])
    labels = np.array([0, 1])
    inertia = compute_inertia(X, centroids, labels, distance_func=LinfinityDistance)
    assert np.isclose(inertia, 0.0)
