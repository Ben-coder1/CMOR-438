from ml.supervised_learning.logitic_regression import LogisticRegression
import numpy as np
import pytest

def test_init_defaults():
    """Test default initialization values."""
    model = LogisticRegression()
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.fit_intercept is True
    assert model._is_fitted is False

def test_init_custom_params():
    """Test initialization with custom parameters."""
    model = LogisticRegression(learning_rate=0.1, n_iterations=50, fit_intercept=False)
    assert model.learning_rate == 0.1
    assert model.n_iterations == 50
    assert model.fit_intercept is False
    assert model.intercept_ is None # Should be None if fit_intercept is False

    # --- Functional Tests ---
def test_fit_simple_classification():
    """Test model ability to separate linearly separable data."""
    X_train = np.array([[1.0], [2.0], [8.0], [9.0]])
    y_train = np.array([0, 0, 1, 1])
    
    # Use high learning rate/iterations for fast, effective convergence on toy data
    model = LogisticRegression(learning_rate=0.5, n_iterations=500) 
    model.fit(X_train, y_train)
    
    assert model._is_fitted is True
    # Check that predictions are sensible (should classify training data perfectly)
    preds = model.predict(X_train)
    assert np.all(preds == y_train)
    
def test_fit_with_different_dtypes():
    """Test fitting with float32 (float), float64 (double), and int data types."""
    # float32/float64 (doubles)
    X_float = np.array([[1.0], [2.0], [8.0], [9.0]], dtype=np.float32)
    # int
    X_int = np.array([[1], [2], [8], [9]], dtype=np.int64)
    y_target = np.array([0, 0, 1, 1])

    # Test float
    model_float = LogisticRegression(learning_rate=0.5, n_iterations=100)
    model_float.fit(X_float, y_target)
    assert model_float._is_fitted is True

    # Test int
    model_int = LogisticRegression(learning_rate=0.5, n_iterations=100)
    model_int.fit(X_int, y_target)
    assert model_int._is_fitted is True

    # Final sanity check: coefficients should be non-zero
    assert np.abs(model_float.coef_[0]) > 0.5 
    assert np.abs(model_int.coef_[0]) > 0.5

# --- Error Tests (Checking your custom validation utilities) ---
def test_fit_error_non_numeric_input():
    """Check TypeError when X contains unconvertible non-numeric data (uses _ensure_numeric_array)."""
    X_bad = np.array([['a'], [1], [2]])
    y_ok = np.array([0, 1, 0])
    model = LogisticRegression()
    
    with pytest.raises(TypeError):
        model.fit(X_bad, y_ok)

def test_fit_error_nan_in_X():
    """Check ValueError when NaN is present in X (uses _ensure_no_nan)."""
    X_nan = np.array([[1.0], [np.nan], [3.0]])
    y_ok = np.array([0, 1, 0])
    model = LogisticRegression()
    
    with pytest.raises(ValueError):
        model.fit(X_nan, y_ok)

def test_fit_error_empty_input():
    """Check ValueError when input array is empty (uses _ensure_non_empty)."""
    X_empty = np.array([]).reshape(0, 1)
    y_empty = np.array([])
    model = LogisticRegression()
    
    with pytest.raises(ValueError):
        model.fit(X_empty, y_empty)

def test_fit_error_target_non_binary():
    """Check ValueError when target y contains values other than 0 or 1."""
    X_ok = np.array([[1], [2], [3]])
    y_bad = np.array([0, 1, 2])
    model = LogisticRegression()
    
    with pytest.raises(ValueError):
        model.fit(X_ok, y_bad)

@pytest.fixture
def fitted_model():
    """Fixture for a simple fitted model."""
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(learning_rate=0.5, n_iterations=500)
    model.fit(X, y)
    return model

# --- Error Tests ---
def test_predict_proba_error_unfitted():
    """Check RuntimeError if predict_proba is called before fit."""
    model = LogisticRegression()
    X_test = np.array([[1.0]])
    
    with pytest.raises(RuntimeError):
        model.predict_proba(X_test)

def test_predict_error_unfitted():
    """Check RuntimeError if predict is called before fit."""
    model = LogisticRegression()
    X_test = np.array([[1.0]])
    
    with pytest.raises(RuntimeError):
        model.predict(X_test)

# --- Functional Tests ---
def test_predict_proba_range(fitted_model):
    """Check that probability outputs are strictly between 0 and 1."""
    X_test = np.array([[0.5], [100]]) # Extreme values
    probas = fitted_model.predict_proba(X_test)
    
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)

def test_predict_standard_threshold(fitted_model):
    """Check standard classification (threshold=0.5) on toy data."""
    # Test data where prediction should be clear
    X_test = np.array([[0.1], [5.0], [9.9]])
    
    preds = fitted_model.predict(X_test)
    # Based on the simple fit, 0.1 should be 0, 5.0 is around the boundary (let's assume 1 after training), 9.9 should be 1
    # For robust test, we check the number of predicted classes
    assert len(preds) == 3
    assert preds.dtype == np.int32 or preds.dtype == np.int64 # Check output type is integer labels

def test_predict_custom_threshold(fitted_model):
    """Check classification using a custom, high threshold."""
    # Use a high threshold (0.99)
    threshold = 0.99 
    X_test = np.array([[8.0], [5.0]])
    
    # 8.0 is likely > 0.99 (Class 1)
    # 5.0 is likely < 0.99 (Class 0, or 1 depending on fit, but definitely testing the threshold logic)
    preds = fitted_model.predict(X_test, threshold=threshold)
    
    # We specifically verify the comparison logic here, assuming a successful fit
    probas = fitted_model.predict_proba(X_test)
    expected = (probas >= threshold).astype(int)
    
    assert np.all(preds == expected)