
from ml.pre_processing.pca import compute_pca
import numpy as np
import pytest

# ----------------------
# Normal cases
# ----------------------

def test_compute_pca_basic_two_features():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    comps, var, X_proj = compute_pca(X, n_components=1)
    # Components should have unit length
    assert np.allclose(np.linalg.norm(comps[0]), 1.0)
    # Explained variance length matches n_components
    assert var.shape == (1,)
    # Projection shape matches (n_samples, n_components)
    assert X_proj.shape == (3, 1)

def test_compute_pca_default_n_components():
    X = np.array([[1, 0], [0, 1], [1, 1]])
    comps, var, X_proj = compute_pca(X)  # default = n_features
    assert comps.shape == (2, 2)
    assert var.shape == (2,)
    assert X_proj.shape == (3, 2)

def test_compute_pca_with_more_samples_than_features():
    X = np.random.randn(10, 3)
    comps, var, X_proj = compute_pca(X, n_components=2)
    assert comps.shape == (2, 3)
    assert var.shape == (2,)
    assert X_proj.shape == (10, 2)

# Numeric variety tests
# ----------------------

def test_compute_pca_with_negatives():
    X = np.array([[-1, -2], [-3, -4], [-5, -6]])
    comps, var, X_proj = compute_pca(X, n_components=1)
    # Shapes correct
    assert comps.shape == (1, 2)
    assert var.shape == (1,)
    assert X_proj.shape == (3, 1)
    # Components unit length
    assert np.allclose(np.linalg.norm(comps[0]), 1.0)

def test_compute_pca_with_floats():
    X = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    comps, var, X_proj = compute_pca(X, n_components=2)
    assert comps.shape == (2, 2)
    assert var.shape == (2,)
    assert X_proj.shape == (3, 2)
    # Mean of transformed data should be ~0
    assert np.allclose(np.mean(X_proj, axis=0), 0.0)

def test_compute_pca_with_mixed_signs():
    X = np.array([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]])
    comps, var, X_proj = compute_pca(X, n_components=2)
    assert comps.shape == (2, 2)
    assert var.shape == (2,)
    assert X_proj.shape == (3, 2)

def test_compute_pca_with_double_precision():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    comps, var, X_proj = compute_pca(X, n_components=2)
    assert comps.dtype == np.float64
    assert var.dtype == np.float64
    assert X_proj.dtype == np.float64
    # Orthogonality check
    dot = np.dot(comps[0], comps[1])
    assert np.isclose(dot, 0.0)

def test_compute_pca_with_single_precision():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    comps, var, X_proj = compute_pca(X, n_components=2)
    assert comps.dtype == np.float32 or comps.dtype == np.float64
    assert var.shape == (2,)
    assert X_proj.shape == (3, 2)

def test_compute_pca_with_mixed_floats_and_ints():
    # Mixed types: ints and floats in the same array
    X = np.array([[1, 2.5], [3.0, 4], [5, 6.5]])
    comps, var, X_proj = compute_pca(X, n_components=2)
    
    # Shapes should be correct
    assert comps.shape == (2, 2)
    assert var.shape == (2,)
    assert X_proj.shape == (3, 2)
    
    # Components should be unit length
    assert np.allclose(np.linalg.norm(comps[0]), 1.0)
    assert np.allclose(np.linalg.norm(comps[1]), 1.0)
    
    # Projection should preserve number of samples
    assert X_proj.shape[0] == X.shape[0]
    
    # Dtype should be float (NumPy promotes mixed ints/floats to float64)
    assert comps.dtype == np.float64
    assert var.dtype == np.float64
    assert X_proj.dtype == np.float64

# ----------------------
# Edge cases and errors
# ----------------------

def test_compute_pca_empty_input_raises():
    X = np.empty((0, 2))
    with pytest.raises(ValueError):
        compute_pca(X)

def test_compute_pca_empty_features_raises():
    X = np.empty((3, 0))
    with pytest.raises(ValueError):
        compute_pca(X)

def test_compute_pca_not_2d_input_raises():
    X = np.array([1, 2, 3])  # 1D
    with pytest.raises(ValueError):
        compute_pca(X)

def test_compute_pca_non_numeric_input_raises():
    X = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(TypeError):
        compute_pca(X)

def test_compute_pca_with_nan_raises():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError):
        compute_pca(X)

def test_compute_pca_invalid_n_components_zero():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        compute_pca(X, n_components=0)

def test_compute_pca_invalid_n_components_negative():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        compute_pca(X, n_components=-1)

def test_compute_pca_too_many_components_raises():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        compute_pca(X, n_components=3)


# ----------------------
# Special cases
# ----------------------

def test_compute_pca_multiple_equal_eigenvalues():
    # Construct data with equal variance in both features
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    comps, var, X_proj = compute_pca(X, n_components=2)
    # Both eigenvalues should be equal (symmetry)
    assert np.allclose(var[0], var[1])
    # Components should still be orthogonal unit vectors
    dot = np.dot(comps[0], comps[1])
    assert np.isclose(dot, 0.0)
    assert np.allclose(np.linalg.norm(comps[0]), 1.0)
    assert np.allclose(np.linalg.norm(comps[1]), 1.0)
