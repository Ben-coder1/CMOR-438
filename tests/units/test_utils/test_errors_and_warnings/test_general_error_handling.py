import numpy as np
import pytest
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_array_like, _ensure_no_nan, _ensure_numeric_array, _ensure_numeric_scalar, _ensure_positive_numeric,
    _ensure_positive_int, _ensure_nonzero, _ensure_same_shape_1d, _ensure_string, _ensure_callable,
    _ensure_non_empty, _ensure_hashable_labels, _ensure_numeric_labels, _ensure_ndim, _ensure_same_length,
    _ensure_in_range, _check_sample_shapes_match, InputShapeError
)


# Tests for _ensure_positive_numeric
# ----------------------------

def test_positive_numeric_valid():
    assert _ensure_positive_numeric(3.5, "alpha") == 3.5
    assert _ensure_positive_numeric(1, "beta") == 1

def test_positive_numeric_non_numeric():
    with pytest.raises(TypeError):
        _ensure_positive_numeric("not a number", "gamma")

def test_positive_numeric_non_positive():
    with pytest.raises(ValueError):
        _ensure_positive_numeric(0, "delta")
    with pytest.raises(ValueError):
        _ensure_positive_numeric(-5, "epsilon")

# ----------------------------
# Tests for _ensure_numeric_array
# ----------------------------

def test_numeric_array_valid_1d():
    arr = _ensure_numeric_array([1, 2, 3], name="vec", ndim=1)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)

def test_numeric_array_valid_2d():
    arr = _ensure_numeric_array([[1.0, 2.0], [3.0, 4.0]], name="X", ndim=2)
    assert arr.shape == (2, 2)

def test_numeric_array_none():
    with pytest.raises(ValueError):
        _ensure_numeric_array(None, name="bad")

def test_numeric_array_empty():
    with pytest.raises(ValueError):
        _ensure_numeric_array([], name="empty")

def test_numeric_array_non_numeric():
    with pytest.raises(TypeError):
        _ensure_numeric_array(["a", "b"], name="bad")

def test_numeric_array_wrong_ndim():
    with pytest.raises(ValueError):
        _ensure_numeric_array([1, 2, 3], name="vec", ndim=2)

# ----------------------------
# Tests for _ensure_numeric_scalar
# ----------------------------

def test_numeric_scalar_valid():
    assert _ensure_numeric_scalar(42, "answer") == 42
    assert _ensure_numeric_scalar(3.14, "pi") == pytest.approx(3.14)

def test_numeric_scalar_invalid():
    with pytest.raises(TypeError):
        _ensure_numeric_scalar("oops", "bad")

# ----------------------------
# Tests for _ensure_array_like
# ----------------------------

def test_array_like_valid():
    arr = _ensure_array_like([1, 2, 3], "vec")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)

def test_array_like_none():
    with pytest.raises(ValueError):
        _ensure_array_like(None, "vec")

def test_array_like_empty():
    with pytest.raises(ValueError):
        _ensure_array_like([], "vec")

def test_array_like_non_array_like():
    class NotArrayLike:
        def __array__(self):
            raise RuntimeError("bad conversion")
    with pytest.raises(TypeError):
        _ensure_array_like(NotArrayLike(), "bad")

# ----------------------------
# Tests for _check_sample_shapes_match
# ----------------------------

def test_shapes_match_exact():
    y = np.array([[1, 2], [3, 4]])
    preds = np.array([[1, 2], [3, 4]])
    _check_sample_shapes_match(y, preds)  # should not raise

def test_shapes_match_per_sample():
    y = np.array([[1], [2], [3]])
    preds = np.array([[0], [1], [2]])
    _check_sample_shapes_match(y, preds)  # should not raise

def test_shapes_mismatch():
    y = np.array([[1, 2], [3, 4]])
    preds = np.array([[1], [2]])
    with pytest.raises(InputShapeError):
        _check_sample_shapes_match(y, preds)

# Tests for _ensure_no_nan
# ----------------------------

def test_no_nan_valid():
    arr = np.array([1.0, 2.0, 3.0])
    out = _ensure_no_nan(arr, "X")
    assert np.array_equal(out, arr)

def test_no_nan_with_nan():
    arr = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        _ensure_no_nan(arr, "X")

def test_no_nan_non_numeric_dtype():
    arr = np.array(["a", "b"])
    # Should just return without error since dtype is not numeric
    out = _ensure_no_nan(arr, "labels")
    assert np.array_equal(out, arr)

# ----------------------------
# Tests for _ensure_positive_int
# ----------------------------

def test_positive_int_valid():
    assert _ensure_positive_int(5, "k") == 5

def test_positive_int_non_int():
    with pytest.raises(TypeError):
        _ensure_positive_int(3.14, "k")

def test_positive_int_non_positive():
    with pytest.raises(ValueError):
        _ensure_positive_int(0, "k")
    with pytest.raises(ValueError):
        _ensure_positive_int(-2, "k")

# ----------------------------
# Tests for _ensure_nonzero
# ----------------------------

def test_nonzero_scalar_valid():
    assert _ensure_nonzero(5, "val") == 5

def test_nonzero_array_valid():
    arr = np.array([1, 2, 3])
    out = _ensure_nonzero(arr, "vec")
    assert np.array_equal(out, arr)

def test_nonzero_scalar_zero():
    with pytest.raises(ValueError):
        _ensure_nonzero(0, "val")

def test_nonzero_array_with_zero():
    with pytest.raises(ValueError):
        _ensure_nonzero(np.array([1, 0, 2]), "vec")

# ----------------------------
# Tests for _ensure_same_shape_1d
# ----------------------------

def test_same_shape_1d_valid():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    _ensure_same_shape_1d(a, "a", b, "b")  # should not raise

def test_same_shape_not_1d():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    with pytest.raises(ValueError):
        _ensure_same_shape_1d(a, "a", b, "b")

def test_same_shape_mismatch():
    a = np.array([1, 2, 3])
    b = np.array([1, 2])
    with pytest.raises(InputShapeError):
        _ensure_same_shape_1d(a, "a", b, "b")

# ----------------------------
# Tests for _ensure_string
# ----------------------------

def test_string_valid():
    assert _ensure_string("hello", "greeting") == "hello"

def test_string_none():
    with pytest.raises(ValueError):
        _ensure_string(None, "greeting")

def test_string_non_string():
    with pytest.raises(TypeError):
        _ensure_string(123, "greeting")

# ----------------------------
# Tests for _ensure_callable
# ----------------------------

def test_callable_valid():
    func = lambda x: x
    assert _ensure_callable(func, "func") is func

def test_callable_none():
    with pytest.raises(ValueError):
        _ensure_callable(None, "func")

def test_callable_non_callable():
    with pytest.raises(TypeError):
        _ensure_callable(42, "func")

# ----------------------------
# Tests for _ensure_non_empty
# ----------------------------

def test_non_empty_valid():
    arr = _ensure_non_empty([1, 2, 3], "vec")
    assert np.array_equal(arr, np.array([1, 2, 3]))

def test_non_empty_empty():
    with pytest.raises(ValueError):
        _ensure_non_empty([], "vec")

# ----------------------------
# Tests for _ensure_hashable_labels
# ----------------------------

def test_hashable_labels_valid():
    arr = _ensure_hashable_labels(["a", "b"], "labels")
    assert np.array_equal(arr, np.array(["a", "b"]))

def test_hashable_labels_empty():
    with pytest.raises(ValueError):
        _ensure_hashable_labels([], "labels")

def test_hashable_labels_unhashable():
    unhashable = [{"a": 1}, {"b": 2}]  # dicts are not hashable
    with pytest.raises(TypeError):
        _ensure_hashable_labels(unhashable, "labels")

# ----------------------------
# Tests for _ensure_numeric_labels
# ----------------------------

def test_numeric_labels_valid():
    arr = _ensure_numeric_labels([1, 2, 3], "labels")
    assert np.array_equal(arr, np.array([1.0, 2.0, 3.0]))

def test_numeric_labels_non_numeric():
    with pytest.raises(TypeError):
        _ensure_numeric_labels(["a", "b"], "labels")

def test_numeric_labels_empty():
    with pytest.raises(ValueError):
        _ensure_numeric_labels([], "labels")

# ----------------------------
# Tests for _ensure_ndim
# ----------------------------

def test_ndim_valid():
    arr = _ensure_ndim([[1, 2], [3, 4]], "X", 2)
    assert arr.shape == (2, 2)

def test_ndim_invalid():
    with pytest.raises(InputShapeError):
        _ensure_ndim([1, 2, 3], "vec", 2)


# Tests for _ensure_same_length
# ----------------------------

def test_same_length_valid():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    # Should not raise
    _ensure_same_length(a, "X", b, "y")

def test_same_length_mismatch():
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    with pytest.raises(InputShapeError) as excinfo:
        _ensure_same_length(a, "X", b, "y")
    assert "X and y must have the same length" in str(excinfo.value)

# ----------------------------
# Tests for _ensure_in_range
# ----------------------------

def test_in_range_valid_inclusive():
    assert _ensure_in_range(0.5, "ratio", 0, 1) == 0.5
    assert _ensure_in_range(0, "ratio", 0, 1) == 0
    assert _ensure_in_range(1, "ratio", 0, 1) == 1

def test_in_range_valid_exclusive():
    assert _ensure_in_range(0.5, "ratio", 0, 1, inclusive=False) == 0.5

def test_in_range_below_min_inclusive():
    with pytest.raises(ValueError):
        _ensure_in_range(-0.1, "ratio", 0, 1)

def test_in_range_below_min_exclusive():
    with pytest.raises(ValueError):
        _ensure_in_range(0, "ratio", 0, 1, inclusive=False)

def test_in_range_above_max_inclusive():
    with pytest.raises(ValueError):
        _ensure_in_range(1.5, "ratio", 0, 1)

def test_in_range_above_max_exclusive():
    with pytest.raises(ValueError):
        _ensure_in_range(1, "ratio", 0, 1, inclusive=False)

def test_in_range_non_numeric():
    with pytest.raises(TypeError):
        _ensure_in_range("bad", "ratio", 0, 1)
