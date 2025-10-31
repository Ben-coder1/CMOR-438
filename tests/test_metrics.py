import math
import numpy as np
import pytest


from ml import LnDistanceConstructor, LinfinityDistance, EuclideanDistance, ascii_word_dist, taxicab_distance



def to_arr(x):
    return np.asarray(x)

# ---- Ln (L^p) distance tests ----

def test_ln_distance_p1_list_and_array():
    L1 = LnDistanceConstructor(1)
    assert L1([1, 2], [4, 6]) == 7
    assert L1(np.array([1, 2]), np.array([4, 6])) == 7

def test_ln_distance_float_p():
    L1_5 = LnDistanceConstructor(1.5)
    result = L1_5([1, 2], [4, 6])
    expected = (abs(1 - 4)**1.5 + abs(2 - 6)**1.5)**(1/1.5)
    assert abs(result - expected) < 1e-9

def test_ln_distance_invalid_p():
    with pytest.raises(ValueError):
        LnDistanceConstructor(0.9)

def test_ln_distance_zero_vector():
    L2 = LnDistanceConstructor(2)
    assert L2([0, 0], [0, 0]) == 0.0
    assert L2(np.array([0, 0]), np.array([0, 0])) == 0.0

def test_ln_distance_mismatched_lengths():
    L2 = LnDistanceConstructor(2)
    with pytest.raises(ValueError):
        L2([1, 2], [1])

def test_ln_distance_with_tuples_and_arrays():
    ln = LnDistanceConstructor(1)
    x = (1, 2, 3)
    y = (4, 5, 6)
    assert ln(x, y) == ln([1, 2, 3], [4, 5, 6])
    assert ln(np.array(x), np.array(y)) == ln(x, y)
    assert ln(x, y) == ln(y, x)

def test_ln_distance_mixed_tuple_list_and_numpy():
    ln = LnDistanceConstructor(3)
    x = (1, 2, 3)
    y = [4, 5, 6]
    assert ln(x, y) == ln([1, 2, 3], [4, 5, 6])
    assert ln(np.array(y), x) == ln(x, np.array(y))

def test_ln_distance_empty_vector():
    L2 = LnDistanceConstructor(2)
    with pytest.raises(ValueError):
        L2([], [])

def test_ln_distance_none_input():
    L2 = LnDistanceConstructor(2)
    with pytest.raises(ValueError):
        L2(None, [1, 2])

def test_ln_distance_non_numeric():
    L2 = LnDistanceConstructor(2)
    with pytest.raises(TypeError):
        L2([1, 'a'], [2, 3])
    # object-dtype numpy array containing non-numeric should also raise
    with pytest.raises(TypeError):
        L2(np.array([1, 'a'], dtype=object), np.array([2, 3], dtype=object))

def test_ln_distance_negative_values_and_symmetry():
    L2 = LnDistanceConstructor(2)
    assert L2([-1, -2], [1, 2]) == L2([1, 2], [-1, -2])
    assert L2([-3, 0], [-3, 0]) == 0.0
    assert L2([-5, 5], [5, -5]) == L2([5, -5], [-5, 5])

def test_ln_distance_symmetry_loop():
    L2 = LnDistanceConstructor(2)
    test_pairs = [
        ([1, 2], [4, 6]),
        ([0, 0], [10, 10]),
        ([-3, 7], [7, -3]),
        ([100, -100], [-100, 100]),
        ([1.5, -2.5], [-1.5, 2.5])
    ]
    for vec1, vec2 in test_pairs:
        d1 = L2(vec1, vec2)
        d2 = L2(vec2, vec1)
        assert pytest.approx(d1, rel=1e-9, abs=1e-12) == d2

def test_ln_distance_large_vector():
    L2 = LnDistanceConstructor(2)
    vec1 = list(range(1000))
    vec2 = list(range(1000, 2000))
    # Expected: sqrt(sum of squares of 1000 differences, each = 1000)
    expected = math.sqrt(1000 * (1000**2))
    result = L2(vec1, vec2)
    assert abs(result - expected) < 1e-6

# ---- L-infinity distance tests ----

def test_linfinity_basic():
    assert LinfinityDistance([1, 2], [4, 6]) == 4
    assert LinfinityDistance(np.array([1,2]), np.array([4,6])) == 4

def test_linfinity_zero_distance_and_tuples():
    assert LinfinityDistance([0, 0], [0, 0]) == 0.0
    assert LinfinityDistance([5, -3], [5, -3]) == 0.0
    x = (1,2,3); y = (4,5,6)
    assert LinfinityDistance(x, y) == LinfinityDistance([1,2,3], [4,5,6])
    assert LinfinityDistance(np.array(x), np.array(y)) == LinfinityDistance(x, y)

def test_linfinity_negative_values_and_symmetry():
    assert LinfinityDistance([-1, -2], [1, 2]) == 4
    assert LinfinityDistance([-5, 5], [5, -5]) == 10

def test_linfinity_symmetry_loop():
    test_pairs = [
        ([1, 2], [4, 6]),
        ([0, 0], [10, 10]),
        ([-3, 7], [7, -3]),
        ([100, -100], [-100, 100]),
        ([1.5, -2.5], [-1.5, 2.5])
    ]
    for vec1, vec2 in test_pairs:
        d1 = LinfinityDistance(vec1, vec2)
        d2 = LinfinityDistance(vec2, vec1)
        assert d1 == d2

def test_linfinity_large_vector():
    vec1 = list(range(1000))
    vec2 = [x + 1000 for x in vec1]
    assert LinfinityDistance(vec1, vec2) == 1000

def test_linfinity_mismatched_lengths_empty_none_non_numeric():
    with pytest.raises(ValueError):
        LinfinityDistance([1, 2], [1])
    with pytest.raises(ValueError):
        LinfinityDistance([], [])
    with pytest.raises(ValueError):
        LinfinityDistance(None, [1, 2])
    with pytest.raises(TypeError):
        LinfinityDistance([1, 'a'], [2, 3])
    with pytest.raises(TypeError):
        LinfinityDistance(np.array([1, 'a'], dtype=object), np.array([2, 3], dtype=object))




#euclidean test distances

def test_euclidean_basic():
    a = np.array([1, 2])
    b = np.array([4, 6])
    result = EuclideanDistance(a, b)
    expected = np.sqrt((3**2 + 4**2))
    assert result == expected

def test_euclidean_zero_distance():
    a = np.array([0, 0])
    b = np.array([0, 0])
    assert EuclideanDistance(a, b) == 0.0

    a = np.array([5, -3])
    b = np.array([5, -3])
    assert EuclideanDistance(a, b) == 0.0

def test_euclidean_negative_values():
    a = np.array([-1, -2])
    b = np.array([1, 2])
    result = EuclideanDistance(a, b)
    expected = np.sqrt(2**2 + 4**2)
    assert abs(result - expected) < 1e-6

def test_euclidean_symmetry_loop():
    test_pairs = [
        (np.array([1, 2]), np.array([4, 6])),
        (np.array([0, 0]), np.array([10, 10])),
        (np.array([-3, 7]), np.array([7, -3])),
        (np.array([100, -100]), np.array([-100, 100])),
        (np.array([1.5, -2.5]), np.array([-1.5, 2.5]))
    ]
    for vec1, vec2 in test_pairs:
        d1 = EuclideanDistance(vec1, vec2)
        d2 = EuclideanDistance(vec2, vec1)
        assert abs(d1 - d2) < 1e-6, f"Symmetry failed for {vec1} and {vec2}"

def test_euclidean_large_vector():
    vec1 = np.arange(1000)
    vec2 = vec1 + 1000
    expected = np.sqrt(1000 * (1000**2))
    result = EuclideanDistance(vec1, vec2)
    assert abs(result - expected) < 1e-6

def test_euclidean_with_tuples_and_lists():
    x = (1, 2, 3)
    y = [4, 5, 6]
    # Should work regardless of input type
    assert EuclideanDistance(x, y) == EuclideanDistance(np.array(x), np.array(y))
    assert EuclideanDistance(x, y) == EuclideanDistance(y, x)

def test_euclidean_zero_vector():
    x = np.zeros(3)
    y = np.zeros(3)
    assert EuclideanDistance(x, y) == 0.0

def test_euclidean_mismatched_lengths():
    with pytest.raises(ValueError):
        EuclideanDistance(np.array([1, 2]), np.array([1]))

def test_euclidean_empty_vector():
    with pytest.raises(ValueError):
        EuclideanDistance(np.array([]), np.array([]))

def test_euclidean_none_input():
    with pytest.raises(ValueError):
        EuclideanDistance(None, np.array([1, 2]))

def test_euclidean_non_numeric():
    with pytest.raises(TypeError):
        EuclideanDistance(np.array([1, 'a'], dtype=object), np.array([2, 3], dtype=object))




#word distance tests


def test_basic_distance():
    assert ascii_word_dist("abc", "def") == abs(97 - 100) + abs(98 - 101) + abs(99 - 102)

def test_same_string():
    assert ascii_word_dist("hello", "hello") == 0

def test_empty_strings():
    assert ascii_word_dist("", "") == 0
    assert ascii_word_dist("a", "") == ord("a")
    assert ascii_word_dist("", "a") == ord("a")

def test_padding_behavior():
    # "abc" vs "a"
    # ASCII: [97, 98, 99] vs [97, 0, 0]
    expected = abs(97 - 97) + abs(98 - 0) + abs(99 - 0)
    assert ascii_word_dist("abc", "a") == expected
    assert ascii_word_dist("a", "abc") == expected  # symmetry

def test_symmetry():
    assert ascii_word_dist("abc", "def") == ascii_word_dist("def", "abc")
    assert ascii_word_dist("abc", "") == ascii_word_dist("", "abc")

def test_non_string_input():
    try:
        ascii_word_dist("abc", 123)
        assert False, "Expected TypeError for non-string input"
    except TypeError:
        pass

    try:
        ascii_word_dist(None, "abc")
        assert False, "Expected TypeError for None input"
    except TypeError:
        pass

def test_unicode_characters():
    # Should still work for valid Unicode characters
    assert ascii_word_dist("a", "ñ") == abs(ord("a") - ord("ñ"))



#taxicab tests
def assert_raises(exc_type, callable_obj, *args, **kwargs):
    try:
        callable_obj(*args, **kwargs)
    except Exception as e:
        assert isinstance(e, exc_type), f"Expected {exc_type}, got {type(e)}"
    else:
        raise AssertionError(f"Expected {exc_type} to be raised but no exception was raised")

# ---- Taxicab (Manhattan) distance tests ----

def test_taxicab_basic():
    a = np.array([1, 2])
    b = np.array([4, 6])
    assert taxicab_distance(a, b) == 7
    # also works with lists
    assert taxicab_distance([1, 2], [4, 6]) == 7

def test_raises_on_none_inputs():
    assert_raises(ValueError, taxicab_distance, None, np.array([1, 2, 3]))
    assert_raises(ValueError, taxicab_distance, np.array([1, 2, 3]), None)

def test_raises_on_empty_vectors():
    assert_raises(ValueError, taxicab_distance, np.array([]), np.array([1]))
    assert_raises(ValueError, taxicab_distance, np.array([1]), np.array([]))

def test_raises_on_length_mismatch():
    assert_raises(ValueError, taxicab_distance, np.array([1, 2]), np.array([1, 2, 3]))

def test_raises_on_non_numeric_elements():
    # object dtype arrays with non-numeric entries
    assert_raises(TypeError, taxicab_distance,
                  np.array([1, 2, "three"], dtype=object),
                  np.array([1, 2, 3], dtype=object))
    assert_raises(TypeError, taxicab_distance,
                  np.array([1, object()], dtype=object),
                  np.array([1, 2], dtype=object))

def test_basic_positive_distance():
    assert taxicab_distance(np.array([1, 2]), np.array([4, 6])) == 7

def test_positives():
    assert taxicab_distance(np.array([0, 1, 2]), np.array([1, 2, 3])) == 3

def test_negatives():
    assert taxicab_distance(np.array([-1, -2, -3]), np.array([-4, -5, -6])) == 9

def test_mixed_signs():
    assert taxicab_distance(np.array([-1, 2, -3]), np.array([4, -2, 3])) == 15

def test_with_floats():
    result = taxicab_distance(np.array([0.5, 1.25, -2.75]),
                              np.array([1.5, -0.25, -1.75]))
    assert abs(result - 3.5) <= 1e-12

def test_large_numbers():
    a = np.array([10**18, -(10**18), 10**18])
    b = np.array([-(10**18), 10**18, 0])
    expected = 5 * 10**18
    assert taxicab_distance(a, b) == expected

def test_large_vector_correctness():
    n = 50_000
    v1 = np.arange(n)
    v2 = np.arange(n - 1, -1, -1)
    expected = np.sum(np.abs(v1 - v2))
    assert taxicab_distance(v1, v2) == expected

def test_identity_zero_distance():
    v = np.array([0, -1.5, 2, 3.25])
    assert taxicab_distance(v, v) == 0
