import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.unsupervised.clustering.groups_only.silhouette_score import compute_silhouette_score

# ----------------------
# Normal cases
# ----------------------

def test_silhouette_basic_ints():
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6]])
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert -1.0 <= score <= 1.0

def test_silhouette_with_floats():
    X = np.array([[0.5, 1.5], [1.5, 2.5], [5.5, 6.5], [6.5, 7.5]])
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert -1.0 <= score <= 1.0

def test_silhouette_with_negatives():
    X = np.array([[-1, -1], [-2, -2], [5, 5], [6, 6]])
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert -1.0 <= score <= 1.0

def test_silhouette_with_mixed_ints_and_floats():
    X = np.array([[1, 2.5], [3.0, 4], [5, 6.5], [7.0, 8]])
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_silhouette_with_double_precision():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert isinstance(score, float)

def test_silhouette_with_single_precision():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels)
    assert isinstance(score, float)


# ----------------------
# Edge cases and errors
# ----------------------

def test_silhouette_labels_length_mismatch():
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([0])  # wrong length
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)

def test_silhouette_only_one_cluster_raises():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    labels = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)

def test_silhouette_empty_X_raises():
    X = np.empty((0, 2))
    labels = np.array([])
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)

def test_silhouette_wrong_shape_X():
    X = np.array([1, 2, 3])  # 1D instead of 2D
    labels = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)

def test_silhouette_wrong_shape_labels():
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([[0, 1]])  # 2D instead of 1D
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)

def test_silhouette_with_nan_raises():
    X = np.array([[0.0, np.nan], [1.0, 1.0]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError):
        compute_silhouette_score(X, labels)


# ----------------------
# Distance function error
# ----------------------

def test_silhouette_distance_func_not_callable():
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        compute_silhouette_score(X, labels, distance_func="not_callable")



# ----------------------
# Special case: LinfinityDistance
# ----------------------

def test_silhouette_with_linfinity_distance():
    def LinfinityDistance(x, y):
        return np.max(np.abs(x - y))
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    labels = np.array([0, 0, 1, 1])
    score = compute_silhouette_score(X, labels, distance_func=LinfinityDistance)
    assert -1.0 <= score <= 1.0
