import pytest
import numpy as np
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance, ascii_word_dist
from ml.pre_processing.test_split import train_test_split
from ml.supervised_learning.knn import KNN
from ml.pre_processing.scaling_centering import normalize_by_max_abs, normalize_by_average_abs, normalize_vectors_by_max_abs, normalize_vectors_by_average_abs

def test_basic_neighbors():
    X = [[0, 0], [1, 1], [2, 2]]
    y = ['a', 'b', 'c']
    knn = KNN()
    neighbors = knn.find_neighbors([1, 1], X, y, K=2, dist=EuclideanDistance)
    assert neighbors[0][1] == 'b'
    assert neighbors[0][2] == 0.0


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
    with pytest.raises(ValueError):
        knn.find_neighbors([0], X, y, K=3, dist=EuclideanDistance)


def test_neighbors_with_class_data_euclidean():
    X = [[1, 1], [2, 2], [3, 3]]
    y = ["a", "b", "c"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([2.1, 2.1], knn.X, knn.y, K=2, dist=EuclideanDistance)
    assert len(neighbors) == 2
    assert neighbors[0][1] == "b"


def test_neighbors_with_class_data_ascii():
    # Non-numeric X should raise under the new KNN contract
    X = ["cat", "bat", "apple"]
    y = ["mammal", "mammal", "fruit"]
    knn = KNN()
    with pytest.raises((TypeError, ValueError)):
        knn.find_neighbors("cap", X, y, K=2, dist=ascii_word_dist)


def test_neighbors_target_equals_training_point():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([1, 2], knn.X, knn.y, K=1, dist=EuclideanDistance)
    assert neighbors[0][1] == "a"
    assert neighbors[0][2] == 0.0


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
    with pytest.raises(ValueError):
        knn.find_neighbors([0], X, y, K=0, dist=EuclideanDistance)


def test_y_wrong_length():
    X = [[0], [1]]
    y = ['a']
    knn = KNN()
    with pytest.raises(ValueError):
        knn.find_neighbors([0], X, y, K=1, dist=EuclideanDistance)


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
    # Non-numeric X should raise under the new KNN contract
    def dict_dist(a, b):
        return abs(a['x'] - b['x'])

    X = [{'x': 1}, {'x': 2}, {'x': 3}]
    y = ['a', 'b', 'c']
    knn = KNN()
    with pytest.raises((TypeError, ValueError)):
        knn.find_neighbors({'x': 2}, X, y, K=2, dist=dict_dist)


def test_knn_find_neighbors_with_ascii_word_dist():
    # Non-numeric X should raise under the new KNN contract
    X = ["cat", "dog", "bat", "apple"]
    y = ["mammal", "mammal", "mammal", "fruit"]
    target = "cap"
    knn = KNN()
    with pytest.raises((TypeError, ValueError)):
        knn.find_neighbors(target, X, y, K=2, dist=ascii_word_dist)




def test_invalid_k_type():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    with pytest.raises(TypeError):
        knn.find_neighbors([0], X, y, K='two', dist=EuclideanDistance)


def test_invalid_distance_function():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    with pytest.raises(TypeError):
        knn.find_neighbors([0], X, y, K=1, dist="not_callable")


def test_missing_target():
    X = [[0], [1]]
    y = ['a', 'b']
    knn = KNN()
    with pytest.raises(ValueError):
        knn.find_neighbors(None, X, y, K=1, dist=EuclideanDistance)


# --- Testing suite for knn predict ---




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
    pred = knn.predict(target, classify=False, K=2)
    valid_outputs = [(3.0 + 2.0) / 2, (3.0 + 4.0) / 2]
    assert any(abs(pred - val) < 1e-6 for val in valid_outputs)








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
    with pytest.raises(ValueError):
        knn.predict([2, 3], K=5)


def test_predict_k_zero():
    X = [[1, 2]]
    y = ["a"]
    knn = KNN(X, y)
    with pytest.raises(ValueError):
        knn.predict([1, 2], K=0)


def test_predict_missing_data():
    knn = KNN()
    with pytest.raises(ValueError):
        knn.predict([1, 2])


def test_predict_mismatched_lengths():
    X = [[1, 2], [3, 4]]
    y = ["a"]
    knn = KNN()
    with pytest.raises(ValueError):
        knn.predict([1, 2], X=X, y=y)


def test_predict_non_numeric_regression_labels():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    with pytest.raises(TypeError):
        knn.predict([2, 3], classify=False)


def test_predict_invalid_classify_flag():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    with pytest.raises(TypeError):
        knn.predict([2, 3], classify="yes")


def test_predict_non_callable_distance():
    X = [[1, 2], [3, 4]]
    y = ["a", "b"]
    knn = KNN(X, y)
    with pytest.raises(TypeError):
        knn.predict([2, 3], dist="euclidean")

# --- Testing suite for knn error ---


def test_error_single_training_point():
    knn = KNN([[1, 2]], ["a"])
    err = knn.error([[1, 2]], ["a"], K=1)
    assert err == 0.0


def test_error_large_training_set():
    X_train = [[i, i + 1] for i in range(1000)]
    y_train = [i % 2 for i in range(1000)]
    X_test = [[0, 1], [999, 1000]]
    y_test = [0, 1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=5)
    assert 0.0 <= err <= 1.0


def test_error_mixed_type_labels_classification():
    X_train = [[0], [1], [2]]
    y_train = ["yes", 1, True]
    X_test = [[0], [1], [2]]
    y_test = ["yes", 1, True]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=1)
    assert err == 0.0


def test_error_with_ascii_word_dist():
    # Non-numeric X should raise under the new KNN contract
    X_train = ["abc", "def", "ghi"]
    y_train = ["x", "y", "z"]
    X_test = ["abc", "ghi"]
    y_test = ["x", "z"]
    
    with pytest.raises((TypeError, ValueError)):
        knn = KNN(X_train, y_train)
        knn.error(X_test, y_test, K=1, dist=ascii_word_dist)


def test_error_binary_classification():
    X_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    X_test = [[1], [2]]
    y_test = [0, 1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=3)
    assert err == 0.0


def test_error_float_regression():
    X_train = [[0], [1], [2]]
    y_train = [1.0, 2.0, 3.0]
    X_test = [[1.5], [0.5]]
    y_test = [2.5, 1.5]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0.0) < 1e-6


def test_error_multiple_neighbors():
    X_train = [[0], [1], [2], [3], [4]]
    y_train = [0, 0, 1, 1, 1]
    X_test = [[2]]
    y_test = [1]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=5)
    assert err == 0.0


def test_error_regression_with_negative_values():
    X_train = [[-1], [0], [1]]
    y_train = [-2.0, 0.0, 2.0]
    X_test = [[-0.5], [0.5]]
    y_test = [-1.0, 1.0]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0) < 1e-6


def test_error_regression_mean_65():
    X_train = [[0], [10]]
    y_train = [0.0, 10.0]
    X_test = [[100], [200]]
    y_test = [70.0, 70.0]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 65.0) < 1e-6


def test_error_mismatched_test_lengths():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(ValueError):
        knn.error([[1, 2]], ["a", "b"])


def test_error_missing_training_data():
    knn = KNN()
    with pytest.raises(ValueError):
        knn.error([[1, 2]], ["a"])


def test_error_classify_not_boolean():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(TypeError):
        knn.error([[1, 2]], ["a"], classify="yes")


def test_error_dist_not_callable():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(TypeError):
        knn.error([[1, 2]], ["a"], dist="not a function")


def test_error_K_not_integer():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(TypeError):
        knn.error([[1, 2]], ["a"], K=2.5)


def test_error_K_too_large():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(ValueError):
        knn.error([[1, 2]], ["a"], K=5)


def test_error_X_train_not_list():
    knn = KNN()
    with pytest.raises(TypeError):
        knn.error([[1, 2]], ["a"], X_train="bad", y_train=["a"])


def test_error_X_y_train_length_mismatch():
    knn = KNN()
    with pytest.raises(ValueError):
        knn.error([[1, 2]], ["a"], X_train=[[1, 2]], y_train=["a", "b"])


def test_error_target_is_none():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(TypeError):
        knn.error([None], ["a"])


def test_error_regression_with_non_numeric_labels():
    knn = KNN([[1, 2]], ["a"])
    with pytest.raises(TypeError):
        knn.error([[1, 2]], ["a"], classify=False)


# --- Integration tests ---


def test_knn_classification_basic():
    X = [[0], [1], [2], [3]]
    y = ["a", "a", "b", "b"]
    knn = KNN(X, y)
    pred = knn.predict([1.5], K=3)
    assert pred == "a"


def test_knn_regression_basic():
    X = [[0], [1], [2]]
    y = [1.0, 2.0, 3.0]
    knn = KNN(X, y)
    pred = knn.predict([1.5], K=2, classify=False)
    assert abs(pred - 2.5) < 1e-6


def test_knn_error_classification():
    X_train = [[0], [1], [2], [3]]
    y_train = ["a", "a", "b", "b"]
    X_test = [[1], [2]]
    y_test = ["a", "b"]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=3)
    assert err == 0.0


def test_knn_error_regression():
    X_train = [[0], [1], [2]]
    y_train = [1.0, 2.0, 3.0]
    X_test = [[1.5], [0.5]]
    y_test = [2.5, 1.5]
    knn = KNN(X_train, y_train)
    err = knn.error(X_test, y_test, K=2, classify=False)
    assert abs(err - 0.0) < 1e-6


def test_knn_string_classification_with_ascii():
    # Non-numeric X should raise under the new KNN contract
    X = ["cat", "dog", "fish"]
    y = ["pet", "pet", "wild"]
    
    with pytest.raises((TypeError, ValueError)):
        knn = KNN(X, y)
        knn.predict("cat", K=1, dist=ascii_word_dist)


def test_knn_error_with_mixed_labels():
    X = [[0], [1], [2]]
    y = ["yes", 1, True]
    knn = KNN(X, y)
    err = knn.error([[0], [1], [2]], ["yes", 1, True], K=1)
    assert err == 0.0


def test_knn_error_large_dataset():
    X = [[i] for i in range(1000)]
    y = [i % 2 for i in range(1000)]
    X_test = [[0], [999]]
    y_test = [0, 1]
    knn = KNN(X, y)
    err = knn.error(X_test, y_test, K=5)
    assert 0.0 <= err <= 1.0, "Should compute bounded error on large dataset"




# --- train_test_split boundary tests ---

def test_train_ratio_zero_raises():
    with pytest.raises(ValueError):
        train_test_split(np.arange(5), train_ratio=0.0)

def test_train_ratio_one_raises():
    with pytest.raises(ValueError):
        train_test_split(np.arange(5), train_ratio=1.0)

def test_train_ratio_near_zero_and_one():
    data = np.arange(10)
    train, test = train_test_split(data, train_ratio=0.1, seed=42)
    assert len(train) == 1 and len(test) == 9
    train, test = train_test_split(data, train_ratio=0.9, seed=42)
    assert len(train) == 9 and len(test) == 1


# --- non-numeric X enforcement ---

def test_non_numeric_X_in_init_raises():
    with pytest.raises((TypeError, ValueError)):
        KNN(["cat", "dog"], ["a", "b"])


# --- regression with integer labels ---

def test_regression_with_integer_labels():
    X = [[0], [1], [2]]
    y = [1, 2, 3]  # ints, still numeric
    knn = KNN(X, y)
    pred = knn.predict([1.5], classify=False, K=2)
    assert abs(pred - 2.5) < 1e-6


# --- custom distance override sanity check ---

def test_custom_distance_override():
    # trivial distance: always 0
    def zero_dist(a, b): return 0.0
    X = [[0], [1]]
    y = ["a", "b"]
    knn = KNN(X, y)
    neighbors = knn.find_neighbors([999], X, y, K=2, dist=zero_dist)
    # Both neighbors should be returned with distance 0
    assert all(d == 0.0 for _, _, d in neighbors)




