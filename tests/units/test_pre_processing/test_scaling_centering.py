import numpy as np
import pytest
from ml.pre_processing.scaling_centering import (
    normalize_by_max_abs,
    normalize_by_average_abs,
    normalize_vectors_by_average_abs,
    normalize_vectors_by_max_abs,
    center_data,
    center_vectors,
    center_and_normalize,
    standardize_data,
    standardize_vectors
)
from ml.pre_processing.test_split import train_test_split
# --- normalize_by_max_abs ---

def test_normalize_typical_array():
    data = np.array([2, -4, 1])
    result = normalize_by_max_abs(data)
    expected = np.array([0.5, -1.0, 0.25])
    np.testing.assert_allclose(result, expected)

def test_normalize_negative_max():
    data = np.array([-3, -1, -2])
    result = normalize_by_max_abs(data)
    expected = np.array([-1.0, -1/3, -2/3])
    np.testing.assert_allclose(result, expected)

def test_normalize_single_element():
    data = np.array([5])
    result = normalize_by_max_abs(data)
    np.testing.assert_allclose(result, [1.0])

def test_normalize_all_same_value():
    data = np.array([3, 3, 3])
    result = normalize_by_max_abs(data)
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

def test_empty_array_raises():
    with pytest.raises(ValueError):
        normalize_by_max_abs(np.array([]))

def test_zero_max_raises():
    with pytest.raises(ValueError):
        normalize_by_max_abs(np.array([0, 0, 0]))

def test_non_numeric_raises():
    with pytest.raises(TypeError):
        normalize_by_max_abs(np.array([1, "a", 3], dtype=object))


# --- normalize_by_average_abs ---

def test_normalize_average_typical_array():
    data = np.array([2, -4, 1])
    result = normalize_by_average_abs(data)
    avg = np.mean(np.abs(data))
    expected = data / avg
    np.testing.assert_allclose(result, expected)

def test_normalize_average_negative():
    data = np.array([-3, -1, -2])
    result = normalize_by_average_abs(data)
    avg = np.mean(np.abs(data))
    expected = data / avg
    np.testing.assert_allclose(result, expected)

def test_average_empty_array_raises():
    with pytest.raises(ValueError):
        normalize_by_average_abs(np.array([]))

def test_average_zero_array_raises():
    with pytest.raises(ValueError):
        normalize_by_average_abs(np.array([0, 0, 0]))



# --- normalize_vectors_by_max_abs ---

def test_max_abs_normalization_correctness():
    data = np.array([[1, -2], [3, 4]])
    result = normalize_vectors_by_max_abs(data)
    expected = np.array([[1/3, -2/4], [3/3, 4/4]])
    np.testing.assert_allclose(result, expected)

def test_max_abs_normalization_zero_column():
    data = np.array([[0, 1], [0, 2]])
    with pytest.raises(ValueError):
        normalize_vectors_by_max_abs(data)


# --- normalize_vectors_by_average_abs ---

def test_average_abs_normalization_correctness():
    data = np.array([[1, -2], [3, 4]])
    avg0 = np.mean(np.abs(data[:, 0]))
    avg1 = np.mean(np.abs(data[:, 1]))
    expected = np.array([[1/avg0, -2/avg1], [3/avg0, 4/avg1]])
    result = normalize_vectors_by_average_abs(data)
    np.testing.assert_allclose(result, expected)

def test_average_abs_normalization_zero_column():
    data = np.array([[0, 1], [0, 2]])
    with pytest.raises(ValueError):
        normalize_vectors_by_average_abs(data)



# Tests for center_data
# ----------------------

def test_center_data_basic_ints():
    arr = np.array([1, 2, 3])
    result = center_data(arr)
    expected = np.array([-1., 0., 1.])
    assert np.allclose(result, expected)
    assert np.isclose(np.mean(result), 0.0)

def test_center_data_with_floats():
    arr = np.array([1.5, 2.5, 3.5])
    result = center_data(arr)
    assert np.isclose(np.mean(result), 0.0)

def test_center_data_with_negatives():
    arr = np.array([-5, -3, -1])
    result = center_data(arr)
    assert np.isclose(np.mean(result), 0.0)

def test_center_data_empty_array_raises():
    arr = np.array([])
    with pytest.raises(ValueError):
        center_data(arr)


# ----------------------
# Tests for center_vectors
# ----------------------

def test_center_vectors_basic():
    arr = np.array([[1, 2], [3, 4]])
    result = center_vectors(arr)
    expected = np.array([[-1., -1.], [1., 1.]])
    assert np.allclose(result, expected)
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])

def test_center_vectors_with_floats_and_negatives():
    arr = np.array([[1.5, -2.5], [3.5, -4.5]])
    result = center_vectors(arr)
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])

def test_center_vectors_empty_matrix_raises():
    arr = np.empty((0, 2))
    with pytest.raises(ValueError):
        center_vectors(arr)


# ----------------------
# Tests for center_and_normalize
# ----------------------

def test_center_and_normalize_1d_max_abs():
    arr = np.array([1, -2, 3])
    result = center_and_normalize(arr, method="max_abs")
    # After centering: [-1, -4, 1] → max abs = 4 → normalized
    assert np.isclose(np.mean(result), 0.0)
    assert np.max(np.abs(result)) == 1.0

def test_center_and_normalize_1d_average_abs():
    arr = np.array([1, -2, 3])
    result = center_and_normalize(arr, method="average_abs")
    assert np.isclose(np.mean(result), 0.0)

def test_center_and_normalize_2d_max_abs():
    arr = np.array([[1, -2], [3, 4]])
    result = center_and_normalize(arr, method="max_abs")
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])
    assert np.max(np.abs(result)) == 1.0

def test_center_and_normalize_2d_average_abs():
    arr = np.array([[1, -2], [3, 4]])
    result = center_and_normalize(arr, method="average_abs")
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])

def test_center_and_normalize_invalid_method_raises():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        center_and_normalize(arr, method="invalid")

def test_center_and_normalize_invalid_ndim_raises():
    arr = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        center_and_normalize(arr)


# ----------------------
# Tests for standardize_data
# ----------------------

def test_standardize_data_basic():
    arr = np.array([10, 20, 30])
    result = standardize_data(arr)
    assert np.isclose(np.mean(result), 0.0)
    assert np.isclose(np.std(result), 1.0)

def test_standardize_data_with_floats_and_negatives():
    arr = np.array([-1.5, 0.0, 1.5])
    result = standardize_data(arr)
    assert np.isclose(np.mean(result), 0.0)
    assert np.isclose(np.std(result), 1.0)

def test_standardize_data_constant_array_raises():
    arr = np.array([5, 5, 5])
    with pytest.raises(ValueError):
        standardize_data(arr)


# ----------------------
# Tests for standardize_vectors
# ----------------------

def test_standardize_vectors_basic():
    arr = np.array([[1, 2], [3, 4]])
    result = standardize_vectors(arr)
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(result, axis=0), [1.0, 1.0])

def test_standardize_vectors_with_floats_and_negatives():
    arr = np.array([[1.5, -2.5], [3.5, -4.5]])
    result = standardize_vectors(arr)
    assert np.allclose(np.mean(result, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(result, axis=0), [1.0, 1.0])

def test_standardize_vectors_constant_columns_raises():
    arr = np.array([[5, 5], [5, 5]])
    with pytest.raises(ValueError):
        standardize_vectors(arr)


# NaN handling
# ----------------------

def test_center_data_with_nan():
    arr = np.array([1.0, np.nan, 3.0])
    # _ensure_numeric_array should allow numeric, but mean will be nan
    result = center_data(arr)
    # The result should propagate NaN
    assert np.isnan(result).any()

def test_center_vectors_with_nan():
    arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
    result = center_vectors(arr)
    assert np.isnan(result).any()

def test_center_and_normalize_with_nan():
    arr = np.array([1.0, np.nan, 3.0])
    # Depending on implementation, normalization will propagate NaN
    result = center_and_normalize(arr, method="max_abs")
    assert np.isnan(result).any()

def test_standardize_data_with_nan_raises():
    arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        standardize_data(arr)

def test_standardize_vectors_with_nan_raises():
    arr = np.array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(ValueError):
        standardize_vectors(arr)


# ----------------------
# Zero standard deviation / divisor
# ----------------------

def test_center_and_normalize_zero_divisor_raises():
    arr = np.array([5, 5, 5])  # after centering → [0,0,0]
    with pytest.raises(ValueError):
        center_and_normalize(arr, method="max_abs")

def test_standardize_data_zero_sd_raises():
    arr = np.array([5, 5, 5])
    with pytest.raises(ValueError):
        standardize_data(arr)

def test_standardize_vectors_zero_sd_raises():
    arr = np.array([[5, 5], [5, 5]])
    with pytest.raises(ValueError):
        standardize_vectors(arr)
