import numpy as np
import pytest
from ml.unsupervised_learning.clustering.k_means_clustering import kmeans_clustering


def test_basic_clustering_reproducible():
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
    centroids1, labels1 = kmeans_clustering(X, k=2, epsilon=1e-4, seed=42)
    centroids2, labels2 = kmeans_clustering(X, k=2, epsilon=1e-4, seed=42)
    assert np.allclose(centroids1, centroids2)
    assert np.array_equal(labels1, labels2)

def test_negative_and_float_values():
    X = np.array([[-1.0, -2.5], [0.0, 0.0], [3.2, -4.1], [5.5, 2.2]])
    centroids, labels = kmeans_clustering(X, k=2, epsilon=1e-4, seed=123)
    assert centroids.shape == (2, 2)
    assert set(labels) <= {0, 1}

def test_linfinity_distance():
    def linf(a, b): return float(np.max(np.abs(a - b)))
    X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    centroids, labels = kmeans_clustering(X, k=2, epsilon=1e-4, seed=1, distance_func=linf)
    assert centroids.shape == (2, 2)

def test_bad_distance_function_non_callable():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(TypeError):
        kmeans_clustering(X, k=2, epsilon=1e-4, distance_func="not_a_func")

def test_bad_distance_function_non_numeric_return():
    def bad_func(a, b): return "not_a_number"
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(TypeError):
        kmeans_clustering(X, k=2, epsilon=1e-4, distance_func=bad_func)

def test_k_greater_than_samples():
    X = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        kmeans_clustering(X, k=2, epsilon=1e-4, seed=1)

def test_invalid_k_type():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        kmeans_clustering(X, k=1.5, epsilon=1e-4)

def test_invalid_epsilon():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        kmeans_clustering(X, k=2, epsilon=-1.0)

def test_invalid_max_iter():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        kmeans_clustering(X, k=2, max_iter=0)

def test_seed_type_error():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(TypeError):
        kmeans_clustering(X, k=2, epsilon=1e-4, seed="bad_seed")

def test_empty_cluster_reinitialization():
    # Force empty cluster by having duplicate points
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    centroids, labels = kmeans_clustering(X, k=2, epsilon=1e-4, seed=42)
    assert centroids.shape == (2, 2)
    assert set(labels) <= {0, 1}
