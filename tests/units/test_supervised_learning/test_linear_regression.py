import numpy as np
import pytest
from ml.supervised_learning.linear_regression import LinearRegression


def test_init_defaults():
    """Test that default parameters are set correctly."""
    model = LinearRegression()
    
    assert model.fit_intercept is True
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._is_fitted is False

def test_init_parameters():
    """Test that parameters passed to __init__ are stored correctly."""
    model = LinearRegression(fit_intercept=False)
    
    assert model.fit_intercept is False
    # State attributes should still be None/False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._is_fitted is False


# --- Input Validation Tests (Fit) ---

# --- Input Validation Tests (Fit) ---

def test_fit_raises_type_error_non_numeric_X():
    """Should raise TypeError if X contains strings."""
    model = LinearRegression()
    X = np.array([["a"], ["b"]])
    y = np.array([1, 2])
    # No match string, just checks for TypeError
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_fit_raises_type_error_non_numeric_y():
    """Should raise TypeError if y contains strings."""
    model = LinearRegression()
    X = np.array([[1], [2]])
    y = np.array(["a", "b"])
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_fit_raises_value_error_nan_X():
    """Should raise ValueError if X contains NaNs."""
    model = LinearRegression()
    X = np.array([[1], [np.nan]])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fit_raises_value_error_empty_X():
    """Should raise ValueError if X is empty."""
    model = LinearRegression()
    X = np.array([])
    y = np.array([])
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fit_raises_value_error_shape_mismatch():
    """Should raise ValueError if n_samples in X and y don't match."""
    model = LinearRegression()
    X = np.array([[1], [2], [3]]) 
    y = np.array([1, 2])          
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fit_raises_value_error_weights_shape_mismatch():
    """Should raise ValueError if sample_weights length != y length."""
    model = LinearRegression()
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    weights = np.array([1, 1, 1]) 
    with pytest.raises(ValueError):
        model.fit(X, y, sample_weight=weights)

# --- Input Validation Tests (Predict) ---

def test_predict_raises_runtime_error_not_fitted():
    """Should raise RuntimeError if predict is called before fit."""
    model = LinearRegression()
    X = np.array([[1]])
    with pytest.raises(RuntimeError):
        model.predict(X)

def test_predict_raises_value_error_nan_input():
    """Should raise ValueError if prediction input has NaNs."""
    model = LinearRegression()
    model.fit(np.array([[1]]), np.array([1])) 
    
    X_bad = np.array([[np.nan]])
    with pytest.raises(ValueError):
        model.predict(X_bad)

# --- Functionality Tests ---

def test_simple_univariate_regression():
    """Toy Problem: y = 2x + 1"""
    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    assert np.isclose(model.coef_[0], 2.0)
    assert np.isclose(model.intercept_, 1.0)
    
    pred = model.predict(np.array([[4]]))
    assert np.isclose(pred[0], 9.0)

def test_multivariate_regression():
    """Toy Problem: y = 1*x1 + 2*x2 + 3"""
    X = np.array([[0, 0], [1, 0], [0, 1]])
    y = np.array([3, 4, 5])
    
    model = LinearRegression()
    model.fit(X, y)
    
    assert np.allclose(model.coef_, [1.0, 2.0])
    assert np.isclose(model.intercept_, 3.0)

def test_no_intercept():
    """Toy Problem: y = x (passing through origin)"""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    assert model.intercept_ == 0.0
    assert np.isclose(model.coef_[0], 1.0)

def test_weighted_least_squares():
    """Outlier at (4, 100) ignored via weight=0."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 100])
    weights = np.array([1, 1, 1, 0]) 
    
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    
    assert np.isclose(model.coef_[0], 1.0, atol=1e-5)

def test_model_mse_integration():
    """Test that model.mean_squared_error works correctly with the reshaping fix."""
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    model = LinearRegression()
    model.fit(X, y)

    assert np.isclose(model.mean_squared_error(X, y), 0.0)


def test_singular_matrix_error():
    """
    Toy Problem: Perfectly collinear features.
    x1 = [1, 2]
    x2 = [2, 4] (x2 is exactly 2*x1)
    
    The matrix X^T X will be singular (non-invertible).
    NumPy's pinv usually handles this, but if we strictly wanted to test
    for failure or stability, we check that it handles it gracefully 
    OR raises the error if we forced `inv` (but here we used pinv).
    
    However, our implementation catches LinAlgError and raises ValueError.
    Note: pinv handles collinearity by finding the min-norm solution, 
    so it might NOT raise an error unless the matrix is fundamentally broken 
    in a way pinv can't handle, or if we switched to `np.linalg.solve`.
    
    If your implementation uses `pinv`, it actually *solves* collinear systems.
    Let's test that it *works* or verify the behavior.
    """
    X = np.array([[1, 2], [2, 4], [3, 6]])
    y = np.array([3, 6, 9])
    
    model = LinearRegression()
    # Should NOT crash because we use pseudoinverse (pinv)
    model.fit(X, y)
    
    # It should find a solution that satisfies y.
    preds = model.predict(X)
    assert np.allclose(preds, y)

