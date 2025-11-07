import numpy as np
import pytest
from ml.pre_processing.pre_process import (
    normalize_by_max_abs,
    normalize_by_average_abs,
    train_test_split,
    normalize_vectors_by_average_abs,
    normalize_vectors_by_max_abs,
)
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


# --- train_test_split ---

def test_default_split_ratio():
    data = np.arange(10)
    train, test = train_test_split(data, seed=42)
    assert len(train) == 7
    assert len(test) == 3
    np.testing.assert_array_equal(np.sort(np.array(train + test)), data)

def test_custom_split_ratio():
    data = np.arange(20)
    train, test = train_test_split(data, train_ratio=0.25, seed=0)
    assert len(train) == 5
    assert len(test) == 15
    np.testing.assert_array_equal(np.sort(np.array(train + test)), data)


# --- normalize_vectors_by_max_abs ---

#This needs a lot more tests..... Need more tests here
#
#
#
#
#
#
#
#
#
#

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
# --- END OF TESTS ---
