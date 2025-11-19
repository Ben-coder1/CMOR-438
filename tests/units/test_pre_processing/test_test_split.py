import numpy as np
from ml.pre_processing.test_split import train_test_split, train_test_split_arrays
import pytest

# Normal seeded cases
# ----------------------

def test_train_test_split_list_basic():
    data = [1, 2, 3, 4, 5]
    train, test = train_test_split(data, train_ratio=0.6, seed=42)
    assert len(train) == 3
    assert len(test) == 2
    # Ensure all elements are present, no duplicates
    assert sorted(train + test) == sorted(data)

def test_train_test_split_tuple_input():
    data = (10, 20, 30, 40)
    train, test = train_test_split(data, train_ratio=0.5, seed=1)
    assert isinstance(train, list)
    assert isinstance(test, list)
    assert len(train) == 2
    assert len(test) == 2

def test_train_test_split_numpy_array_input():
    data = np.array([1, 2, 3, 4, 5, 6])
    train, test = train_test_split(data, train_ratio=0.5, seed=123)
    assert len(train) == 3
    assert len(test) == 3
    assert sorted(train + test) == sorted(data.tolist())

def test_train_test_split_reproducibility_with_seed():
    data = list(range(10))
    train1, test1 = train_test_split(data, train_ratio=0.7, seed=99)
    train2, test2 = train_test_split(data, train_ratio=0.7, seed=99)
    assert train1 == train2
    assert test1 == test2


#normal cases

# Tests without seed (non-deterministic shuffle)
# ----------------------

def test_train_test_split_no_seed_list():
    data = [1, 2, 3, 4, 5]
    train, test = train_test_split(data, train_ratio=0.6)
    # Lengths should match ratio
    assert len(train) == int(len(data) * 0.6)
    assert len(test) == len(data) - len(train)
    # All elements preserved
    assert sorted(train + test) == sorted(data)

def test_train_test_split_no_seed_tuple():
    data = (10, 20, 30, 40)
    train, test = train_test_split(data, train_ratio=0.5)
    assert len(train) == int(len(data) * 0.5)
    assert len(test) == len(data) - len(train)
    assert sorted(train + test) == sorted(data)

def test_train_test_split_no_seed_numpy_array():
    data = np.array([1, 2, 3, 4, 5, 6])
    train, test = train_test_split(data, train_ratio=0.5)
    assert len(train) == int(len(data) * 0.5)
    assert len(test) == len(data) - len(train)
    assert sorted(train + test) == sorted(data.tolist())

def test_train_test_split_no_seed_default_ratio():
    data = list(range(10))
    train, test = train_test_split(data)  # default ratio 0.7
    assert len(train) == int(len(data) * 0.7)
    assert len(test) == len(data) - len(train)
    assert sorted(train + test) == sorted(data)


# ----------------------
# Edge cases
# ----------------------

def test_train_test_split_small_dataset_one_element():
    data = [42]
    train, test = train_test_split(data, train_ratio=0.5, seed=0)
    # With one element, train gets 0, test gets 1
    assert len(train) == 0 or len(train) == 1
    assert len(train) + len(test) == 1

def test_train_test_split_train_ratio_default():
    data = list(range(10))
    train, test = train_test_split(data, seed=0)  # default ratio 0.7
    assert len(train) == 7
    assert len(test) == 3

def test_train_test_split_train_ratio_close_to_zero():
    data = list(range(5))
    train, test = train_test_split(data, train_ratio=0.01, seed=0)
    assert len(train) == 0  # int(5*0.01) = 0
    assert len(test) == 5

def test_train_test_split_train_ratio_close_to_one():
    data = list(range(5))
    train, test = train_test_split(data, train_ratio=0.99, seed=0)
    assert len(train) == 4  # int(5*0.99) = 4
    assert len(test) == 1


# ----------------------
# Error cases
# ----------------------

def test_train_test_split_empty_data_raises():
    with pytest.raises(ValueError):
        train_test_split([], train_ratio=0.5)

def test_train_test_split_ratio_zero_raises():
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        train_test_split(data, train_ratio=0)

def test_train_test_split_ratio_one_raises():
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        train_test_split(data, train_ratio=1)

def test_train_test_split_ratio_negative_raises():
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        train_test_split(data, train_ratio=-0.1)

def test_train_test_split_ratio_above_one_raises():
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        train_test_split(data, train_ratio=1.1)

def test_train_test_split_data_none_raises():
    with pytest.raises(TypeError):
        train_test_split(None, train_ratio=0.5)


def test_basic_split_shapes_and_alignment():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 1, 0, 1, 0])
    X_train, y_train, X_test, y_test = train_test_split_arrays(X, y, train_ratio=0.6, seed=42)

    # Check shapes
    assert X_train.shape == (3, 1)
    assert y_train.shape == (3,)
    assert X_test.shape == (2, 1)
    assert y_test.shape == (2,)

    # Alignment: indices must match
    for xi, yi in zip(X_train, y_train):
        assert (xi[0] % 2 == 0 and yi == 1) or (xi[0] % 2 == 1 and yi == 0)

def test_reproducibility_with_seed():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    split1 = train_test_split_arrays(X, y, train_ratio=0.5, seed=123)
    split2 = train_test_split_arrays(X, y, train_ratio=0.5, seed=123)
    # Same seed → identical splits
    assert all(np.array_equal(a, b) for a, b in zip(split1, split2))

def test_different_seed_changes_split():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    split1 = train_test_split_arrays(X, y, train_ratio=0.5, seed=1)
    split2 = train_test_split_arrays(X, y, train_ratio=0.5, seed=2)
    # Different seeds → not identical
    assert not all(np.array_equal(a, b) for a, b in zip(split1, split2))

def test_boundary_train_ratio_low():
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)
    # Very small train_ratio
    X_train, y_train, X_test, y_test = train_test_split_arrays(X, y, train_ratio=0.01, seed=0)
    assert len(X_train) == 0 or len(X_train) == 1  # floor of split
    assert len(X_train) + len(X_test) == 5
    # Alignment check
    assert np.all(X_test.flatten() == y_test)

def test_boundary_train_ratio_high():
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)
    # Very high train_ratio
    X_train, y_train, X_test, y_test = train_test_split_arrays(X, y, train_ratio=0.99, seed=0)
    assert len(X_test) == 0 or len(X_test) == 1
    assert len(X_train) + len(X_test) == 5
    # Alignment check
    assert np.all(X_train.flatten() == y_train)

def test_invalid_train_ratio_zero_or_one():
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)
    with pytest.raises(ValueError):
        train_test_split_arrays(X, y, train_ratio=0)
    with pytest.raises(ValueError):
        train_test_split_arrays(X, y, train_ratio=1)

def test_misaligned_lengths():
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(4)
    with pytest.raises(ValueError):
        train_test_split_arrays(X, y, train_ratio=0.5)

def test_empty_inputs():
    X = np.array([]).reshape(0, 1)
    y = np.array([])
    with pytest.raises(ValueError):
        train_test_split_arrays(X, y, train_ratio=0.5)

def test_non_numpy_inputs_are_converted():
    X = [[1], [2], [3]]
    y = [0, 1, 0]
    X_train, y_train, X_test, y_test = train_test_split_arrays(X, y, train_ratio=0.67, seed=42)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
