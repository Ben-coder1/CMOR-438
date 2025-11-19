import numpy as np
import pytest
from ml.unsupervised_learning.clustering.dbscan import dbscan
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance, LinfinityDistance

def test_basic_clustering_euclidean():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]])
    labels = dbscan(X, eps=0.5, min_samples=2, distance_func=EuclideanDistance)
    # Expect first two points in same cluster, last as noise
    assert labels.shape == (3,)
    assert set(labels) == {0, -1}

def test_basic_clustering_linfty():
    X = np.array([[0, 0], [0, 0.4], [5, 5]])
    labels = dbscan(X, eps=0.5, min_samples=2, distance_func=LinfinityDistance)
    assert labels.shape == (3,)
    assert set(labels) == {0, -1}

def test_negative_and_float_values():
    X = np.array([[-1.5, -2.3], [-1.6, -2.4], [3.2, 4.1]])
    labels = dbscan(X, eps=0.5, min_samples=2)
    assert labels.shape == (3,)
    assert set(labels) == {0, -1}

def test_duplicate_points():
    X = np.array([[1, 1], [1, 1], [2, 2]])
    labels = dbscan(X, eps=0.5, min_samples=2)
    assert labels.shape == (3,)
    # Two identical points should form a cluster
    assert labels[0] == labels[1]

def test_single_point():
    X = np.array([[0, 0]])
    labels = dbscan(X, eps=1.0, min_samples=1)
    assert labels.shape == (1,)
    # Single point with min_samples=1 should be its own cluster
    assert labels[0] == 0



# --- Error triggers ---

def test_eps_zero_or_negative():
    X = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        dbscan(X, eps=0, min_samples=2)
    with pytest.raises(ValueError):
        dbscan(X, eps=-1, min_samples=2)

def test_min_samples_zero_or_negative():
    X = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        dbscan(X, eps=1.0, min_samples=0)
    with pytest.raises(ValueError):
        dbscan(X, eps=1.0, min_samples=-5)

def test_distance_func_not_callable():
    X = np.array([[0, 0], [1, 1]])
    with pytest.raises(TypeError):
        dbscan(X, eps=1.0, min_samples=2, distance_func="not_a_function")

def test_distance_func_returns_non_numeric():
    def bad_distance(a, b):
        return "string"
    X = np.array([[0, 0], [1, 1]])
    with pytest.raises(TypeError):
        dbscan(X, eps=1.0, min_samples=2, distance_func=bad_distance)

def test_empty_input():
    X = np.empty((0, 2))
    with pytest.raises(ValueError):
        dbscan(X, eps=1.0, min_samples=1)