from ml.metrics import LnDistanceCostructor, LinfinityDistance, EuclideanDistance, ascii_word_dist, taxicab_distance




#testing sweet for Ln distance constructor
def test_ln_distance_p1():
    L1 = LnDistanceCostructor(1)
    assert L1([1, 2], [4, 6]) == 7

def test_ln_distance_float_p():
    L1_5 = LnDistanceCostructor(1.5)
    result = L1_5([1, 2], [4, 6])
    expected = (abs(1 - 4)**1.5 + abs(2 - 6)**1.5)**(1/1.5)
    assert abs(result - expected) < 1e-6

def test_ln_distance_invalid_p():
    try:
        LnDistanceCostructor(0.9)
        assert False, "Expected ValueError for p < 1"
    except ValueError:
        pass

def test_ln_distance_zero_vector():
    L2 = LnDistanceCostructor(2)
    assert L2([0, 0], [0, 0]) == 0.0

def test_ln_distance_mismatched_lengths():
    try:
        L2 = LnDistanceCostructor(2)
        L2([1, 2], [1])
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass
def test_ln_distance_with_tuples():
    ln = LnDistanceCostructor(1)
    x = (1, 2, 3)
    y = (4, 5, 6)
    assert ln(x, y) == ln([1, 2, 3], [4, 5, 6])
    assert ln(x, y) == ln(y, x)

def test_ln_distance_mixed_tuple_list():
    ln = LnDistanceCostructor(3)
    x = (1, 2, 3)
    y = [4, 5, 6]
    assert ln(x, y) == ln([1, 2, 3], [4, 5, 6])
    assert ln(y, x) == ln(x, y)

def test_ln_distance_zero_vector():
    ln = LnDistanceCostructor(2)
    x = (0, 0, 0)
    y = (0, 0, 0)
    assert ln(x, y) == 0


def test_ln_distance_empty_vector():
    try:
        L2 = LnDistanceCostructor(2)
        L2([], [])
        assert False, "Expected ValueError for empty vectors"
    except ValueError:
        pass

def test_ln_distance_none_input():
    try:
        L2 = LnDistanceCostructor(2)
        L2(None, [1, 2])
        assert False, "Expected ValueError for None input"
    except ValueError:
        pass

def test_ln_distance_non_numeric():
    try:
        L2 = LnDistanceCostructor(2)
        L2([1, 'a'], [2, 3])
        assert False, "Expected TypeError for non-numeric input"
    except TypeError:
        pass
def test_ln_distance_negative_values():
    L2 = LnDistanceCostructor(2)
    assert L2([-1, -2], [1, 2]) == L2([1, 2], [-1, -2])
    assert L2([-3, 0], [-3, 0]) == 0.0
    assert L2([-5, 5], [5, -5]) == L2([5, -5], [-5, 5])

def test_ln_distance_symmetry_loop():
    L2 = LnDistanceCostructor(2)
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
        assert abs(d1 - d2) < 1e-6, f"Symmetry failed for {vec1} and {vec2}"


def test_ln_distance_large_vector():
    L2 = LnDistanceCostructor(2)
    vec1 = list(range(1000))
    vec2 = list(range(1000, 2000))
    # Expected: sqrt(sum of squares of 1000 differences, each = 1000)
    expected = (1000 * (1000**2))**0.5
    result = L2(vec1, vec2)
    assert abs(result - expected) < 1e-6


#testing suite for Linfinity distance

def test_linfinity_basic():
    assert LinfinityDistance([1, 2], [4, 6]) == 4

def test_linfinity_zero_distance():
    assert LinfinityDistance([0, 0], [0, 0]) == 0.0
    assert LinfinityDistance([5, -3], [5, -3]) == 0.0

def test_linfinity_negative_values():
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
        assert d1 == d2, f"Symmetry failed for {vec1} and {vec2}"

def test_linfinity_large_vector():
    vec1 = list(range(1000))
    vec2 = [x + 1000 for x in vec1]
    assert LinfinityDistance(vec1, vec2) == 1000
    
def test_linfinity_with_tuples():
    x = (1, 2, 3)
    y = (4, 5, 6)
    assert LinfinityDistance(x, y) == LinfinityDistance([1, 2, 3], [4, 5, 6])
    assert LinfinityDistance(x, y) == LinfinityDistance(y, x)
    assert LinfinityDistance(x, [4, 5, 6]) == LinfinityDistance([1, 2, 3], y)

def test_linfinity_zero_vector():
    x = (0, 0, 0)
    y = (0, 0, 0)
    assert LinfinityDistance(x, y) == 0


def test_linfinity_mismatched_lengths():
    try:
        LinfinityDistance([1, 2], [1])
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass

def test_linfinity_empty_vector():
    try:
        LinfinityDistance([], [])
        assert False, "Expected ValueError for empty vectors"
    except ValueError:
        pass

def test_linfinity_none_input():
    try:
        LinfinityDistance(None, [1, 2])
        assert False, "Expected ValueError for None input"
    except ValueError:
        pass

def test_linfinity_non_numeric():
    try:
        LinfinityDistance([1, 'a'], [2, 3])
        assert False, "Expected TypeError for non-numeric input"
    except TypeError:
        pass


#euclidean test distances

def test_euclidean_basic():
    assert EuclideanDistance([1, 2], [4, 6]) == ((3**2 + 4**2)**0.5)

def test_euclidean_zero_distance():
    assert EuclideanDistance([0, 0], [0, 0]) == 0.0
    assert EuclideanDistance([5, -3], [5, -3]) == 0.0

def test_euclidean_negative_values():
    result = EuclideanDistance([-1, -2], [1, 2])
    expected = ((2**2 + 4**2)**0.5)
    assert abs(result - expected) < 1e-6

def test_euclidean_symmetry_loop():
    test_pairs = [
        ([1, 2], [4, 6]),
        ([0, 0], [10, 10]),
        ([-3, 7], [7, -3]),
        ([100, -100], [-100, 100]),
        ([1.5, -2.5], [-1.5, 2.5])
    ]
    for vec1, vec2 in test_pairs:
        d1 = EuclideanDistance(vec1, vec2)
        d2 = EuclideanDistance(vec2, vec1)
        assert abs(d1 - d2) < 1e-6, f"Symmetry failed for {vec1} and {vec2}"

def test_euclidean_large_vector():
    vec1 = list(range(1000))
    vec2 = [x + 1000 for x in vec1]
    expected = (1000 * (1000**2))**0.5
    result = EuclideanDistance(vec1, vec2)
    assert abs(result - expected) < 1e-6

def test_euclidean_with_tuples():
    x = (1, 2, 3)
    y = (4, 5, 6)
    assert EuclideanDistance(x, y) == EuclideanDistance([1, 2, 3], [4, 5, 6])
    assert EuclideanDistance(x, y) == EuclideanDistance(y, x)
    assert EuclideanDistance(x, [4, 5, 6]) == EuclideanDistance([1, 2, 3], y)

def test_euclidean_zero_vector():
    x = (0, 0, 0)
    y = (0, 0, 0)
    assert EuclideanDistance(x, y) == 0


def test_euclidean_mismatched_lengths():
    try:
        EuclideanDistance([1, 2], [1])
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass

def test_euclidean_empty_vector():
    try:
        EuclideanDistance([], [])
        assert False, "Expected ValueError for empty vectors"
    except ValueError:
        pass

def test_euclidean_none_input():
    try:
        EuclideanDistance(None, [1, 2])
        assert False, "Expected ValueError for None input"
    except ValueError:
        pass

def test_euclidean_non_numeric():
    try:
        EuclideanDistance([1, 'a'], [2, 3])
        assert False, "Expected TypeError for non-numeric input"
    except TypeError:
        pass


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

def test_taxicab_basic():
    assert taxicab_distance([1, 2], [4, 6]) == 7

def assert_raises(exc_type, callable_obj, *args, **kwargs):
    try:
        callable_obj(*args, **kwargs)
    except Exception as e:
        assert isinstance(e, exc_type), f"Expected {exc_type}, got {type(e)}"
    else:
        raise AssertionError(f"Expected {exc_type} to be raised but no exception was raised")

def test_raises_on_none_inputs():
    assert_raises(ValueError, taxicab_distance, None, [1, 2, 3])
    assert_raises(ValueError, taxicab_distance, [1, 2, 3], None)

def test_raises_on_empty_vectors():
    assert_raises(ValueError, taxicab_distance, [], [1])
    assert_raises(ValueError, taxicab_distance, [1], [])

def test_raises_on_length_mismatch():
    assert_raises(ValueError, taxicab_distance, [1, 2], [1, 2, 3])

def test_raises_on_non_numeric_elements():
    assert_raises(TypeError, taxicab_distance, [1, 2, "three"], [1, 2, 3])
    assert_raises(TypeError, taxicab_distance, [1, object()], [1, 2])

def test_basic_positive_distance():
    assert taxicab_distance([1, 2], [4, 6]) == 7

def test_positives():
    assert taxicab_distance([0, 1, 2], [1, 2, 3]) == 3

def test_negatives():
    assert taxicab_distance([-1, -2, -3], [-4, -5, -6]) == 9

def test_mixed_signs():
    assert taxicab_distance([-1, 2, -3], [4, -2, 3]) == 15

def test_with_floats():
    result = taxicab_distance([0.5, 1.25, -2.75], [1.5, -0.25, -1.75])
    assert abs(result - 3.5) <= 1e-12

def test_large_numbers():
    a = [10**18, -(10**18), 10**18]
    b = [-(10**18), 10**18, 0]
    expected = 5 * 10**18
    assert taxicab_distance(a, b) == expected

def test_large_vector_correctness():
    n = 50_000  # large but conservative for typical CI runners; increase if desired
    v1 = list(range(n))
    v2 = list(range(n - 1, -1, -1))
    expected = sum(abs(a - b) for a, b in zip(v1, v2))
    assert taxicab_distance(v1, v2) == expected

def test_identity_zero_distance():
    v = [0, -1.5, 2, 3.25]
    assert taxicab_distance(v, v) == 0
