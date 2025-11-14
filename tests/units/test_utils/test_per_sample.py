from ml.utils.per_sample import _apply_per_sample
import numpy as np
import pytest
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError

def test_vectorized_function():
    # Vectorized distance: L2 norm row-wise
    def l2(u, v): return np.linalg.norm(u - v, axis=1)
    y_true = np.array([[0, 0], [1, 1]])
    y_pred = np.array([[1, 0], [1, 2]])
    result = _apply_per_sample(l2, y_true, y_pred)
    expected = np.array([1.0, np.sqrt(1.0)])
    np.testing.assert_allclose(result, expected)

def test_scalar_only_function():
    # Scalar function: sum of absolute differences
    def l1(u, v): return np.sum(np.abs(u - v))
    y_true = np.array([[0, 0], [1, 1]])
    y_pred = np.array([[1, 0], [1, 2]])
    result = _apply_per_sample(l1, y_true, y_pred)
    expected = np.array([1.0, 1.0])
    np.testing.assert_allclose(result, expected)

def test_single_array_input():
    # Function that operates on a single array
    def sum_vec(x): return np.sum(x)
    X = np.array([[1, 2], [3, 4]])
    result = _apply_per_sample(sum_vec, X)
    expected = np.array([3, 7])
    np.testing.assert_array_equal(result, expected)

def test_mismatched_lengths_raises():
    def dummy(u, v): return np.sum(u - v)
    a = np.array([[1], [2]])
    b = np.array([[1]])
    with pytest.raises(ValueError):
        _apply_per_sample(dummy, a, b)

def test_non_callable_raises():
    a = np.array([[1], [2]])
    with pytest.raises(TypeError):
        _apply_per_sample("not_a_function", a)

def test_function_returns_none_fallback():
    def bad_func(x): return None
    X = np.array([[1], [2]])
    # Vectorized call returns None → fallback row-wise also returns None
    result = _apply_per_sample(bad_func, X)
    # Should produce array of NaNs (float(None) fails, but dtype=float coerces None to nan)
    assert result.shape == (2,)
    assert np.all(np.isnan(result))

def test_function_raises_exception_fallback():
    def raises(u, v): raise RuntimeError("fail")
    a = np.array([[1], [2]])
    b = np.array([[1], [2]])
    # Should fall back to row-wise, which also raises
    with pytest.raises(RuntimeError):
        _apply_per_sample(raises, a, b)

def test_empty_array_input():
    def identity(x): return x
    X = np.array([]).reshape(0, 1)
    result = _apply_per_sample(identity, X)
    assert result.shape == (0,1)