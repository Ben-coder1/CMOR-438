from ml.pre_process import normalize_by_max_abs, normalize_by_average_abs, train_test_split, normalize_vectors_by_average_abs, normalize_vectors_by_max_abs

# Correct outputs
def test_normalize_typical_list():
    data = [2, -4, 1]
    result = normalize_by_max_abs(data)
    expected = [0.5, -1.0, 0.25]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

def test_normalize_negative_max():
    data = [-3, -1, -2]
    result = normalize_by_max_abs(data)
    expected = [-1.0, -1/3, -2/3]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

def test_normalize_single_element():
    data = [5]
    result = normalize_by_max_abs(data)
    assert abs(result[0] - 1.0) < 1e-6

def test_normalize_all_same_value():
    data = [3, 3, 3]
    result = normalize_by_max_abs(data)
    assert all(abs(x - 1.0) < 1e-6 for x in result)

#Error cases
def test_empty_list_raises():
    try:
        normalize_by_max_abs([])
        assert False, "Should raise ValueError for empty list"
    except ValueError:
        pass

def test_zero_max_raises():
    try:
        normalize_by_max_abs([0, 0, 0])
        assert False, "Should raise ValueError for zero max"
    except ValueError:
        pass

def test_non_numeric_raises():
    try:
        normalize_by_max_abs([1, "a", 3])
        assert False, "Should raise TypeError for non-numeric values"
    except TypeError:
        pass

def test_non_list_input_raises():
    try:
        normalize_by_max_abs("not a list")
        assert False, "Should raise TypeError for non-list input"
    except TypeError:
        pass

def test_normalize_negative_max_only():
    data = [1, -5, 2]
    result = normalize_by_max_abs(data)
    expected = [1/5, -1.0, 2/5]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6





def test_normalize_typical_list():
    data = [2, -4, 1]
    result = normalize_by_average_abs(data)
    avg = (abs(2) + abs(-4) + abs(1)) / 3  # = 7/3
    expected = [2/avg, -4/avg, 1/avg]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

def test_normalize_negative_average():
    data = [-3, -1, -2]
    result = normalize_by_average_abs(data)
    avg = (3 + 1 + 2) / 3  # = 2.0
    expected = [-3/avg, -1/avg, -2/avg]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

def test_normalize_single_element():
    data = [5]
    result = normalize_by_average_abs(data)
    assert abs(result[0] - 1.0) < 1e-6

def test_normalize_all_same_value():
    data = [3, 3, 3]
    result = normalize_by_average_abs(data)
    assert all(abs(x - 1.0) < 1e-6 for x in result)

def test_normalize_negative_max_only():
    data = [1, -5, 2]
    result = normalize_by_average_abs(data)
    avg = (1 + 5 + 2) / 3  # = 8/3
    expected = [1/(8/3), -5/(8/3), 2/(8/3)]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

#Error cases
def test_empty_list_raises():
    try:
        normalize_by_average_abs([])
        assert False, "Should raise ValueError for empty list"
    except ValueError:
        pass

def test_zero_average_raises():
    try:
        normalize_by_average_abs([0, 0, 0])
        assert False, "Should raise ValueError for zero average"
    except ValueError:
        pass

def test_non_numeric_raises():
    try:
        normalize_by_average_abs([1, "a", 3])
        assert False, "Should raise TypeError for non-numeric values"
    except TypeError:
        pass

def test_non_list_input_raises():
    try:
        normalize_by_average_abs("not a list")
        assert False, "Should raise TypeError for non-list input"
    except TypeError:
        pass

def test_default_split_ratio():
    data = list(range(10))
    train, test = train_test_split(data)
    assert len(train) == 7
    assert len(test) == 3
    assert sorted(train + test) == data

def test_custom_split_ratio():
    data = list(range(20))
    train, test = train_test_split(data, train_ratio=0.25)
    assert len(train) == 5
    assert len(test) == 15
    assert sorted(train + test) == data

def test_split_with_seed():
    data = list(range(10))
    train1, test1 = train_test_split(data, seed=42)
    train2, test2 = train_test_split(data, seed=42)
    assert train1 == train2
    assert test1 == test2

def test_split_single_element():
    data = [99]
    train, test = train_test_split(data, train_ratio=0.5)
    assert len(train) == 0
    assert len(test) == 1

def test_split_near_zero_ratio():
    data = list(range(5))
    train, test = train_test_split(data, train_ratio=0.01)
    assert len(train) == 0
    assert len(test) == 5

def test_split_near_one_ratio():
    data = list(range(5))
    train, test = train_test_split(data, train_ratio=0.99)
    assert len(train) == 4
    assert len(test) == 1

#Error cases
def test_empty_list_raises():
    try:
        train_test_split([])
        assert False, "Should raise ValueError for empty list"
    except ValueError:
        pass

def test_invalid_ratio_low():
    try:
        train_test_split([1, 2, 3], train_ratio=0.0)
        assert False, "Should raise ValueError for ratio <= 0"
    except ValueError:
        pass

def test_invalid_ratio_high():
    try:
        train_test_split([1, 2, 3], train_ratio=1.0)
        assert False, "Should raise ValueError for ratio >= 1"
    except ValueError:
        pass

def test_non_list_input_raises():
    try:
        train_test_split("not a list")
        assert False, "Should raise TypeError for non-list input"
    except TypeError:
        pass


def test_max_abs_normalization_correctness():
    data = [[1, -2], [3, 4]]
    result = normalize_vectors_by_max_abs(data)
    assert result == [[1/3, -2/4], [3/3, 4/4]], "Should normalize each column by its max abs value"

def test_max_abs_normalization_empty_input():
    try:
        normalize_vectors_by_max_abs([])
        assert False, "Should raise ValueError on empty input"
    except ValueError as e:
        assert "must be a non-empty list" in str(e)

def test_max_abs_normalization_non_list_input():
    try:
        normalize_vectors_by_max_abs("not a list")
        assert False, "Should raise ValueError on non-list input"
    except ValueError as e:
        assert "must be a non-empty list" in str(e)

def test_max_abs_normalization_non_list_elements():
    try:
        normalize_vectors_by_max_abs([1, 2, 3])
        assert False, "Should raise TypeError when elements are not lists"
    except TypeError as e:
        assert "Each element must be a list" in str(e)

def test_max_abs_normalization_non_numeric_elements():
    try:
        normalize_vectors_by_max_abs([[1, 2], ["a", 3]])
        assert False, "Should raise TypeError on non-numeric elements"
    except TypeError as e:
        assert "All elements must be numeric" in str(e)

def test_max_abs_normalization_unequal_lengths():
    try:
        normalize_vectors_by_max_abs([[1, 2], [3]])
        assert False, "Should raise ValueError on unequal vector lengths"
    except ValueError as e:
        assert "must have the same length" in str(e)

def test_max_abs_normalization_zero_column():
    try:
        normalize_vectors_by_max_abs([[0, 1], [0, 2]])
        assert False, "Should raise ValueError when max abs is zero in any column"
    except ValueError as e:
        assert "must not be zero" in str(e)

def test_average_abs_normalization_correctness():
    data = [[1, -2], [3, 4]]
    avg0 = (abs(1) + abs(3)) / 2
    avg1 = (abs(-2) + abs(4)) / 2
    expected = [[1/avg0, -2/avg1], [3/avg0, 4/avg1]]
    result = normalize_vectors_by_average_abs(data)
    for r, e in zip(result, expected):
        assert all(abs(a - b) < 1e-6 for a, b in zip(r, e)), "Should normalize each column by its average abs value"

def test_average_abs_normalization_empty_input():
    try:
        normalize_vectors_by_average_abs([])
        assert False, "Should raise ValueError on empty input"
    except ValueError as e:
        assert "must be a non-empty list" in str(e)

def test_average_abs_normalization_non_list_input():
    try:
        normalize_vectors_by_average_abs("not a list")
        assert False, "Should raise ValueError on non-list input"
    except ValueError as e:
        assert "must be a non-empty list" in str(e)

def test_average_abs_normalization_non_list_elements():
    try:
        normalize_vectors_by_average_abs([1, 2, 3])
        assert False, "Should raise TypeError when elements are not lists"
    except TypeError as e:
        assert "Each element must be a list" in str(e)

def test_average_abs_normalization_non_numeric_elements():
    try:
        normalize_vectors_by_average_abs([[1, 2], ["a", 3]])
        assert False, "Should raise TypeError on non-numeric elements"
    except TypeError as e:
        assert "All elements must be numeric" in str(e)

def test_average_abs_normalization_unequal_lengths():
    try:
        normalize_vectors_by_average_abs([[1, 2], [3]])
        assert False, "Should raise ValueError on unequal vector lengths"
    except ValueError as e:
        assert "must have the same length" in str(e)

def test_average_abs_normalization_zero_column():
    try:
        normalize_vectors_by_average_abs([[0, 1], [0, 2]])
        assert False, "Should raise ValueError when average abs is zero in any column"
    except ValueError as e:
        assert "must not be zero" in str(e)


