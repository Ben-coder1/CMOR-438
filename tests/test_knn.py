from ml.metrics import EuclideanDistance, ascii_word_dist
from ml.knn import KNN
from ml.pre_process import normalize_by_max_abs, normalize_by_average_abs, train_test_split, normalize_vectors_by_max_abs, normalize_vectors_by_average_abs


def test_basic_neighbors():
    X = [[0, 0], [1, 1], [2, 2]]
    y = ['a', 'b', 'c']
    knn = KNN()
    neighbors = knn.find_neighbors([1, 1], X, y, K=2, dist=EuclideanDistance)
    assert neighbors == [([1, 1], 'b', 0.0), ([0, 0], 'a', EuclideanDistance([1, 1], [0, 0]))]

def test_identical_points_including_target():
    X = [[1, 1], [1, 1], [2, 2]]
    y = ['a', 'b', 'c']
    knn = KNN()
    neighbors = knn.find_neighbors([1, 1], X, y, K=2, dist=EuclideanDistance)
    assert neighbors[0][2] == 0.0
    assert neighbors[1][2] == 0.0

def test_k_too_large():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    try:
        knn.find_neighbors([0], X, y, K=3, dist=EuclideanDistance)
        assert False, "Expected ValueError for K too large"
    except ValueError:
        pass

def test_neighbors_with_class_data_euclidean():
    X = [[1, 1], [2, 2], [3, 3]]
    y = ["a", "b", "c"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([2.1, 2.1], knn.X, knn.y, K=2, dist=EuclideanDistance)
    assert len(neighbors) == 2
    assert neighbors[0][1] == "b"

def test_neighbors_with_class_data_ascii():
    X = ["cat", "bat", "apple"]
    y = ["mammal", "mammal", "fruit"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors("cap", knn.X, knn.y, K=2, dist=ascii_word_dist)
    assert len(neighbors) == 2
    assert all(label in y for _, label, _ in neighbors)

def test_neighbors_target_equals_training_point():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([1, 2], knn.X, knn.y, K=1, dist=EuclideanDistance)
    assert neighbors[0][0] == [1, 2]
    assert neighbors[0][1] == "a"

def test_neighbors_k_equals_dataset_size():
    X = [[1, 2], [3, 4], [5, 6]]
    y = ["a", "b", "c"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([2, 3], knn.X, knn.y, K=3, dist=EuclideanDistance)
    assert len(neighbors) == 3
    assert neighbors[0][2] <= neighbors[1][2] <= neighbors[2][2]


def test_k_zero():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    try:
        knn.find_neighbors([0], X, y, K=0, dist=EuclideanDistance)
        assert False, "Expected TypeError for K=0"
    except TypeError:
        pass

def test_y_wrong_length():
    X = [[0], [1]]
    y = ['a']
    knn = KNN()
    try:
        knn.find_neighbors([0], X, y, K=1, dist=EuclideanDistance)
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass

def test_y_none_allowed():
    X = [[0], [1]]
    knn = KNN()
    neighbors = knn.find_neighbors([0], X, y=None, K=1, dist=EuclideanDistance)
    assert neighbors[0][1] is None

def test_y_mixed_types():
    X = [[0], [1], [2]]
    y = ['a', 1, 2.5]
    knn = KNN()
    neighbors = knn.find_neighbors([1], X, y, K=2, dist=EuclideanDistance)
    assert isinstance(neighbors[0][1], (str, int, float))
    assert isinstance(neighbors[1][1], (str, int, float))

def test_target_as_tuple():
    X = [[0], [1], [2]]
    y = ['a', 'b', 'c']
    knn = KNN()
    neighbors = knn.find_neighbors((1,), X, y, K=2, dist=EuclideanDistance)
    assert neighbors[0][2] == 0.0

def test_target_as_dict_with_custom_distance():
    def dict_dist(a, b):
        return abs(a['x'] - b['x'])

    X = [{'x': 1}, {'x': 2}, {'x': 3}]
    y = ['a', 'b', 'c']
    knn = KNN()
    neighbors = knn.find_neighbors({'x': 2}, X, y, K=2, dist=dict_dist)
    assert neighbors[0][1] == 'b'
    assert neighbors[0][2] == 0

def test_knn_find_neighbors_with_ascii_word_dist():
    X = ["cat", "dog", "bat", "apple"]
    y = ["mammal", "mammal", "mammal", "fruit"]
    target = "cap"

    knn = KNN()
    neighbors = knn.find_neighbors(target, X, y, K=2, dist=ascii_word_dist)

    assert len(neighbors) == 2
    assert all(label in y for _, label, _ in neighbors)
    assert neighbors[0][2] <= neighbors[1][2]  # distances are sorted

#this distance is only for testing, so did not put in the metrics class
def flexible_ascii_dist(a, b):
    a_str = "".join(map(str, a)) if isinstance(a, (list, tuple)) else str(a)
    b_str = "".join(map(str, b)) if isinstance(b, (list, tuple)) else str(b)
    max_len = max(len(a_str), len(b_str))
    a_str = a_str.ljust(max_len, "\0")
    b_str = b_str.ljust(max_len, "\0")
    return sum(abs(ord(x) - ord(y)) for x, y in zip(a_str, b_str))

def test_knn_with_multitype_distance():
    X = ["cat", ("d", "o", "g"), ["b", "a", "t"], "apple"]
    y = ["mammal", "mammal", "mammal", "fruit"]
    target = ["c", "a", "p"]

    knn = KNN()
    neighbors = knn.find_neighbors(target, X, y, K=2, dist=flexible_ascii_dist)

    assert len(neighbors) == 2
    assert all(label in y for _, label, _ in neighbors)
    assert neighbors[0][2] <= neighbors[1][2]


def test_invalid_k_type():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    try:
        knn.find_neighbors([0], X, y, K='two', dist=EuclideanDistance)
        assert False, "Expected TypeError for non-integer K"
    except TypeError:
        pass

def test_invalid_distance_function():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    try:
        knn.find_neighbors([0], X, y, K=1, dist="not_callable")
        assert False, "Expected TypeError for non-callable distance function"
    except TypeError:
        pass

def test_missing_target():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    try:
        knn.find_neighbors(None, X, y, K=1, dist=EuclideanDistance)
        assert False, "Expected ValueError for missing target"
    except ValueError:
        pass





#Testing suite for knn predict 

def test_knn_predict_with_ascii_word_dist():
    X = ["cat", "bat", "apple"]
    y = ["mammal", "mammal", "fruit"]
    target = "cap"

    knn = KNN(X, y)
    pred = knn.predict(target, K=2, dist=ascii_word_dist)
    assert pred == "mammal"


def test_predict_classification():
    X = [[1, 1], [2, 2], [3, 3]]
    y = ["a", "a", "b"]
    knn = KNN(X, y)
    assert knn.predict([1.5, 1.5], classify=True, K=2) == "a"

def test_predict_regression_basic():
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0.0, 1.0, 2.0]
    knn = KNN(X, y)
    pred = knn.predict([1.5, 1.5], classify=False, K=2, dist=EuclideanDistance)
    expected = (1.0 + 2.0) / 2
    assert abs(pred - expected) < 1e-6


def test_predict_regression_with_larger_tied_neighbors():
    X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    knn = KNN(X, y)
    target = [3, 3]
    K = 2

    # Distances from target:
    # [3,3] → 0.0
    # [2,2] → √2 ≈ 1.41
    # [4,4] → √2 ≈ 1.41
    # So neighbors are [3,3] and either [2,2] or [4,4]

    pred = knn.predict(target, classify=False, K=2)

    # Accept either average of [3.0, 2.0] or [3.0, 4.0]
    valid_outputs = [(3.0 + 2.0) / 2, (3.0 + 4.0) / 2]
    assert any(abs(pred - val) < 1e-6 for val in valid_outputs), f"Unexpected prediction: {pred}"


def test_predict_with_ascii_distance():
    X = ["cat", "bat", "apple"]
    y = ["mammal", "mammal", "fruit"]
    knn = KNN(X, y)
    assert knn.predict("cap", K=2, dist=ascii_word_dist) == "mammal"

def test_predict_with_multitype_distance():
    X = ["cat", ("b", "a", "t"), ["a", "p", "p", "l", "e"]]
    y = ["mammal", "mammal", "fruit"]
    knn = KNN(X, y)
    assert knn.predict(["c", "a", "p"], K=2, dist=flexible_ascii_dist) == "mammal"


def test_predict_target_equals_training_point():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    assert knn.predict([1, 2], K=1) == "a"

def test_predict_k_equals_1():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    assert knn.predict([2, 3], K=1) in y

def test_predict_k_equals_dataset_size():
    X = [[1, 2], [3, 4], [5, 6]]
    y = ["a", "a", "b"]
    knn = KNN(X, y)
    assert knn.predict([2, 3], K=3) == "a"

def test_predict_k_too_large():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    try:
        knn.predict([2, 3], K=5)
        assert False, "Should raise ValueError for K > len(X)"
    except ValueError:
        pass

def test_predict_k_zero():
    X = [[1, 2]]
    y = ["a"]
    knn = KNN(X, y)
    try:
        knn.predict([1, 2], K=0)
        assert False, "Should raise TypeError for K=0"
    except TypeError:
        pass


def test_predict_missing_data():
    knn = KNN()
    try:
        knn.predict([1, 2])
        assert False, "Should raise ValueError for missing training data"
    except ValueError:
        pass

def test_predict_mismatched_lengths():
    X = [[1, 2], [3, 4]]
    y = ["a"]
    knn = KNN()
    try:
        knn.predict([1, 2], X=X, y=y)
        assert False, "Should raise ValueError for mismatched lengths"
    except ValueError:
        pass

def test_predict_non_numeric_regression_labels():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    try:
        knn.predict([2, 3], classify=False)
        assert False, "Should raise TypeError for non-numeric regression labels"
    except TypeError:
        pass

def test_predict_invalid_classify_flag():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    try:
        knn.predict([2, 3], classify="yes")
        assert False, "Should raise TypeError for non-boolean classify"
    except TypeError:
        pass

def test_predict_non_callable_distance():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    try:
        knn.predict([2, 3], dist="euclidean")
        assert False, "Should raise TypeError for non-callable dist"
    except TypeError:
        pass








def test_error_single_training_point():
    knn = KNN([[1, 2]], ["a"])
    err = knn.error([[1, 2]], ["a"], K=1)
    assert err == 0.0, "Should be zero error with identical single training/test point"

def test_error_large_training_set():
    X_train = [[i, i + 1] for i in range(1000)]
    y_train = [i % 2 for i in range(1000)]
    X_test = [[0, 1], [999, 1000]]
    y_test = [0, 1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=5)
    assert 0.0 <= err <= 1.0, "Error should be bounded and finite"

def test_error_mixed_type_labels_classification():
    X_train = [[0], [1], [2]]
    y_train = ["yes", 1, True]
    X_test = [[0], [1], [2]]
    y_test = ["yes", 1, True]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=1)
    assert err == 0.0, "Should classify correctly with mixed-type labels"

def test_error_with_ascii_word_dist():
    X_train = ["abc", "def", "ghi"]
    y_train = ["x", "y", "z"]
    X_test = ["abc", "ghi"]
    y_test = ["x", "z"]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=1, dist=ascii_word_dist)
    assert err == 0.0, "ascii_word_dist should yield correct classification output"








def test_error_binary_classification():
    X_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    X_test = [[1], [2]]
    y_test = [0, 1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=3)
    assert err == 0.0, "Should classify correctly with majority voting"





def test_error_float_regression():
    X_train = [[0], [1], [2]]
    y_train = [1.0, 2.0, 3.0]
    X_test = [[1.5], [0.5]]
    y_test = [2.5, 1.5]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0.0) < 1e-6, "Should compute mean absolute error correctly"


def test_error_multiple_neighbors():
    X_train = [[0], [1], [2], [3], [4]]
    y_train = [0, 0, 1, 1, 1]
    X_test = [[2]]
    y_test = [1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=5)
    assert err == 0.0, "Should classify correctly with larger K"

def test_error_regression_with_negative_values():
    X_train = [[-1], [0], [1]]
    y_train = [-2.0, 0.0, 2.0]
    X_test = [[-0.5], [0.5]]
    y_test = [-1.0, 1.0]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0) < 1e-6, "Should handle negative regression values correctly"

def test_error_regression_mean_65():
    X_train = [[0], [10]]
    y_train = [0.0, 10.0]
    X_test = [[100], [200]]
    y_test = [70.0, 70.0]  # Each error = |70 - 5| = 65
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 65.0) < 1e-6, "Should produce mean regression error of 65.0"




def test_error_mismatched_test_lengths():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a", "b"])
        assert False, "Should raise ValueError for mismatched test lengths"
    except ValueError:
        pass

def test_error_missing_training_data():
    knn = KNN()
    try:
        knn.error([[1, 2]], ["a"])
        assert False, "Should raise ValueError for missing training data"
    except ValueError:
        pass

def test_error_classify_not_boolean():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a"], classify="yes")
        assert False, "Should raise TypeError for non-boolean classify"
    except TypeError:
        pass

def test_error_dist_not_callable():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a"], dist="not a function")
        assert False, "Should raise TypeError for non-callable dist"
    except TypeError:
        pass

def test_error_K_not_integer():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a"], K=2.5)
        assert False, "Should raise TypeError for non-integer K"
    except TypeError:
        pass

def test_error_K_too_large():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a"], K=5)
        assert False, "Should raise ValueError for K too large"
    except ValueError:
        pass

def test_error_X_train_not_list():
    knn = KNN()
    try:
        knn.error([[1, 2]], ["a"], X_train="bad", y_train=["a"])
        assert False, "Should raise TypeError for non-list X_train"
    except TypeError:
        pass

def test_error_X_y_train_length_mismatch():
    knn = KNN()
    try:
        knn.error([[1, 2]], ["a"], X_train=[[1, 2]], y_train=["a", "b"])
        assert False, "Should raise ValueError for mismatched training lengths"
    except ValueError:
        pass

def test_error_target_is_none():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([None], ["a"])
        assert False, "Should raise ValueError for None target"
    except ValueError:
        pass

def test_error_regression_with_non_numeric_labels():
    knn = KNN([[1, 2]], ["a"])
    try:
        knn.error([[1, 2]], ["a"], classify=False)
        assert False, "Should raise TypeError for non-numeric labels in regression"
    except TypeError:
        pass




#Integration tests


def test_knn_classification_basic():
    X = [[0], [1], [2], [3]]
    y = ["a", "a", "b", "b"]
    knn = KNN(X, y)
    pred = knn.predict([1.5], K=3)
    assert pred == "a", "Should classify based on majority label"

def test_knn_regression_basic():
    X = [[0], [1], [2]]
    y = [1.0, 2.0, 3.0]
    knn = KNN(X, y)
    pred = knn.predict([1.5], K=2, classify=False)
    assert abs(pred - 2.5) < 1e-6, "Should regress to average of nearest labels"

def test_knn_error_classification():
    X_train = [[0], [1], [2], [3]]
    y_train = ["a", "a", "b", "b"]
    X_test = [[1], [2]]
    y_test = ["a", "b"]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=3)
    assert err == 0.0, "Should compute zero classification error"

def test_knn_error_regression():
    X_train = [[0], [1], [2]]
    y_train = [1.0, 2.0, 3.0]
    X_test = [[1.5], [0.5]]
    y_test = [2.5, 1.5]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0.0) < 1e-6, "Should compute zero regression error"

def test_knn_string_classification_with_ascii():
    X = ["cat", "dog", "fish"]
    y = ["pet", "pet", "wild"]
    knn = KNN(X, y)
    pred = knn.predict("cat", K=1, dist=ascii_word_dist)
    assert pred == "pet", "Should classify string input using ascii_word_dist"

def test_knn_error_with_mixed_labels():
    X = [[0], [1], [2]]
    y = ["yes", 1, True]
    knn = KNN(X, y)
    err = knn.error([[0], [1], [2]], ["yes", 1, True], K=1)
    assert err == 0.0, "Should classify correctly with mixed-type labels"

def test_knn_error_large_dataset():
    X = [[i] for i in range(1000)]
    y = [i % 2 for i in range(1000)]
    X_test = [[0], [999]]
    y_test = [0, 1]
    knn = KNN(X, y)
    err = knn.error(X_test, y_test, K=5)
    assert 0.0 <= err <= 1.0, "Should compute bounded error on large dataset"

def test_knn_predict_with_explicit_training_override():
    knn = KNN()
    X_train = [[0], [1]]
    y_train = ["a", "b"]
    pred = knn.predict([0.5], K=2, X=X_train, y=y_train)
    assert pred in ["a", "b"], "Should predict using provided training data"

def test_knn_error_with_explicit_training_override():
    knn = KNN()
    X_train = [[0], [1]]
    y_train = [0.0, 2.0]
    X_test = [[0.5]]
    y_test = [1.0]
    err = knn.error(X_test, y_test, K=2, classify=False, X_train=X_train, y_train=y_train)
    assert abs(err - 0.0) < 1e-6, "Should compute correct error with explicit training override"


#because train split is stochastic, these tests have ranged errors.
def test_knn_pipeline_max_abs_numeric_classification():
    data = [([10], 0), ([20], 0), ([30], 1), ([40], 1)]
    X = [x for x, _ in data]
    y = [y for _, y in data]
    X_norm = normalize_vectors_by_max_abs(X)
    normalized = [(x, y) for x, y in zip(X_norm, y)]
    train, test = train_test_split(normalized, train_ratio=0.5)
    X_train, y_train = [x for x, _ in train], [y for _, y in train]
    X_test, y_test = [x for x, _ in test], [y for _, y in test]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2)
    assert 0.0 <= err <= 1.0, "Should classify correctly with K=2 after componentwise max-abs normalization"

def test_knn_pipeline_average_abs_numeric_regression():
    data = [([-5], -10.0), ([-2], -4.0), ([2], 4.0), ([5], 10.0)]
    X = [x for x, _ in data]
    y = [y for _, y in data]
    X_norm = normalize_vectors_by_average_abs(X)
    normalized = [(x, y) for x, y in zip(X_norm, y)]
    train, test = train_test_split(normalized, train_ratio=0.5)
    X_train, y_train = [x for x, _ in train], [y for _, y in train]
    X_test, y_test = [x for x, _ in test], [y for _, y in test]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert isinstance(err, float) and err >= 0.0, "Should produce valid non-negative regression error"

def test_knn_pipeline_max_abs_numeric_regression():
    data = [([1], 2.0), ([2], 4.0), ([3], 6.0), ([4], 8.0)]
    X = [x for x, _ in data]
    y = [y for _, y in data]
    X_norm = normalize_vectors_by_max_abs(X)
    normalized = [(x, y) for x, y in zip(X_norm, y)]
    train, test = train_test_split(normalized, train_ratio=0.5)
    X_train, y_train = [x for x, _ in train], [y for _, y in train]
    X_test, y_test = [x for x, _ in test], [y for _, y in test]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert 0.0 <= err <= 8.0, "Should regress correctly with K=2 after componentwise max-abs normalization"

def test_knn_pipeline_average_abs_numeric_classification():
    data = [([-3], 0), ([-1], 0), ([1], 1), ([3], 1)]
    X = [x for x, _ in data]
    y = [y for _, y in data]
    X_norm = normalize_vectors_by_average_abs(X)
    normalized = [(x, y) for x, y in zip(X_norm, y)]
    train, test = train_test_split(normalized, train_ratio=0.5)
    X_train, y_train = [x for x, _ in train], [y for _, y in train]
    X_test, y_test = [x for x, _ in test], [y for _, y in test]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2)
    assert 0.0 <= err <= 1.0, "Should classify correctly with K=2 after componentwise average-abs normalization"
