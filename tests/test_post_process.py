from ml.post_process import majorityLabel, averageLabel



def test_average_label_basic():
    assert averageLabel([1, 2, 3, 4]) == 2.5
    assert averageLabel([10.0, 20.0]) == 15.0

def test_average_label_mixed_types():
    assert averageLabel([1, 2.5, 3]) == (1 + 2.5 + 3) / 3

def test_average_label_large_vector():
    vec = list(range(1000))  # 0 to 999
    expected = sum(vec) / len(vec)
    assert averageLabel(vec) == expected

def test_average_label_none_input():
    try:
        averageLabel(None)
        assert False, "Expected TypeError for None input"
    except TypeError:
        pass

def test_average_label_non_list_input():
    try:
        averageLabel("not a list")
        assert False, "Expected TypeError for non-list input"
    except TypeError:
        pass

def test_average_label_empty_list():
    try:
        averageLabel([])
        assert False, "Expected ValueError for empty list"
    except ValueError:
        pass

def test_average_label_non_numeric_elements():
    try:
        averageLabel([1, 'a', 3])
        assert False, "Expected TypeError for non-numeric elements"
    except TypeError:
        pass


#testing for majority label

def test_majority_label_strings():
    assert majorityLabel(['cat', 'dog', 'cat', 'bird']) == 'cat'

def test_majority_label_integers():
    assert majorityLabel([1, 2, 2, 3, 1, 2]) == 2

def test_majority_label_floats():
    assert majorityLabel([1.1, 2.2, 1.1, 3.3]) == 1.1

def test_majority_label_mixed_numeric():
    assert majorityLabel([1, 1.0, 2, 1]) == 1  # 1 and 1.0 are treated as equal

def test_majority_label_mixed_types():
    assert majorityLabel(['a', 'b', 'a', 1, 1]) == 'a'  # 'a' appears first among top counts

def test_majority_label_tie_breaking():
    assert majorityLabel(['x', 'y', 'x', 'y']) == 'x'  # 'x' appears first

def test_majority_label_large_input():
    labels = ['a'] * 500 + ['b'] * 499
    assert majorityLabel(labels) == 'a'

def test_majority_label_none_input():
    try:
        majorityLabel(None)
        assert False, "Expected TypeError for None input"
    except TypeError:
        pass

def test_majority_label_non_list_input():
    try:
        majorityLabel("not a list")
        assert False, "Expected TypeError for non-list input"
    except TypeError:
        pass

def test_majority_label_empty_list():
    try:
        majorityLabel([])
        assert False, "Expected ValueError for empty list"
    except ValueError:
        pass

def test_majority_label_unhashable_elements():
    try:
        majorityLabel([[1], [1], [2]])
        assert False, "Expected TypeError for unhashable elements"
    except TypeError:
        pass
