from ml.utils.per_sample import _apply_per_sample
import numpy as np
import pytest
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError
# ==========================================
# TESTS FOR _apply_per_sample
# ==========================================

def test_apply_per_sample_vectorized_success():
    """Test that a function supporting vectorization is called efficiently."""
    def vec_func(x): 
        return x * 2

    arr = np.array([1, 2, 3])
    # Should execute the vectorized path and return [2, 4, 6]
    res = _apply_per_sample(vec_func, arr)
    
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, [2, 4, 6])

def test_apply_per_sample_fallback_logic():
    """Test that function requiring scalar input triggers the fallback loop."""
    def scalar_only_func(x):
        if x.ndim > 0:
            raise ValueError("I only accept scalars!")
        return x + 10

    arr = np.array([1, 2, 3])
    # The try/except block in _apply_per_sample should catch the ValueError
    # and proceed to the row-wise loop.
    res = _apply_per_sample(scalar_only_func, arr)
    
    np.testing.assert_array_equal(res, [11, 12, 13])

def test_apply_per_sample_input_mismatch():
    """Test error when input arrays have different lengths."""
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2])
    
    with pytest.raises(ValueError):
        _apply_per_sample(lambda x, y: x + y, arr1, arr2)

def test_apply_per_sample_handles_nan_output():
    """Test that NaNs returned by the function are preserved correctly."""
    def func_with_nan(x):
        if x < 0:
            return np.nan
        return x

    arr = np.array([-1, 0, 1])
    res = _apply_per_sample(func_with_nan, arr)
    
    # Should result in [nan, 0., 1.]
    # We check properties rather than exact equality for NaNs
    assert np.isnan(res[0])
    assert res[1] == 0.0
    assert res[2] == 1.0
    assert np.issubdtype(res.dtype, np.floating)

def test_apply_per_sample_string_output():
    """Test that the function can return strings (verifying no forced float dtype)."""
    def str_func(x):
        return str(x)

    arr = np.array([1, 2])
    res = _apply_per_sample(str_func, arr)
    
    assert res[0] == "1"
    assert res[1] == "2"
    assert res.dtype.kind in {'U', 'S', 'O'} # Unicode, String, or Object

def test_apply_per_sample_object_return():
    """Test that returning complex objects results in an object array."""
    class DummyObj:
        def __init__(self, val): self.val = val

    def obj_func(x):
        return DummyObj(x)

    arr = np.array([1, 2])
    res = _apply_per_sample(obj_func, arr)
    
    assert res.dtype == object
    assert res[0].val == 1
    assert res[1].val == 2

def test_apply_per_sample_multiple_args():
    """Test passing multiple array arguments to the function."""
    def add(x, y):
        return x + y

    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    
    res = _apply_per_sample(add, a, b)
    np.testing.assert_array_equal(res, [11, 22, 33])