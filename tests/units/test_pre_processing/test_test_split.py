import numpy as np
from ml.pre_processing.test_split import train_test_split
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
