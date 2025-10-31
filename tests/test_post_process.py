import numpy as np
import pytest
from ml.post_processing.post_process import majorityLabel, averageLabel

# --- averageLabel tests ---

def test_average_label_basic():
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([10.0, 20.0])
    np.testing.assert_allclose(averageLabel(arr1), 2.5)
    np.testing.assert_allclose(averageLabel(arr2), 15.0)

def test_average_label_mixed_types():
    arr = np.array([1, 2.5, 3])
    expected = (1 + 2.5 + 3) / 3
    np.testing.assert_allclose(averageLabel(arr), expected)

def test_average_label_large_vector():
    arr = np.arange(1000)  # 0 to 999
    expected = np.mean(arr)
    np.testing.assert_allclose(averageLabel(arr), expected)

def test_average_label_none_input():
    with pytest.raises(ValueError):
        averageLabel(None)

def test_average_label_non_array_input():
    with pytest.raises(TypeError):
        averageLabel("not an array")

def test_average_label_empty_array():
    with pytest.raises(ValueError):
        averageLabel(np.array([]))

def test_average_label_non_numeric_elements():
    with pytest.raises(TypeError):
        averageLabel(np.array([1, "a", 3], dtype=object))


# --- majorityLabel tests ---

def test_majority_label_strings():
    arr = np.array(['cat', 'dog', 'cat', 'bird'])
    assert majorityLabel(arr) == 'cat'

def test_majority_label_integers():
    arr = np.array([1, 2, 2, 3, 1, 2])
    assert majorityLabel(arr) == 2

def test_majority_label_floats():
    arr = np.array([1.1, 2.2, 1.1, 3.3])
    assert majorityLabel(arr) == 1.1

def test_majority_label_mixed_numeric():
    arr = np.array([1, 1.0, 2, 1])
    assert majorityLabel(arr) == 1  # 1 and 1.0 treated as equal

def test_majority_label_mixed_types():
    arr = np.array(['a', 'b', 'a', 1, 1], dtype=object)
    assert majorityLabel(arr) == 'a'

def test_majority_label_tie_breaking():
    arr = np.array(['x', 'y', 'x', 'y'])
    assert majorityLabel(arr) == 'x'

def test_majority_label_large_input():
    arr = np.array(['a'] * 500 + ['b'] * 499)
    assert majorityLabel(arr) == 'a'

def test_majority_label_none_input():
    with pytest.raises(ValueError):
        majorityLabel(None)

def test_majority_label_non_array_input():
    with pytest.raises(TypeError):
        majorityLabel("not an array")

def test_majority_label_empty_array():
    with pytest.raises(ValueError):
        majorityLabel(np.array([]))

def test_majority_label_unhashable_elements():
    with pytest.raises(TypeError):
        majorityLabel(np.array([[1], [1], [2]], dtype=object))
