import numpy as np
import pytest
from ml.distances.evaluation_metrics import mean_squared_error, classification_accuracy
from ml.utils._errors_and_warnings.error_handling import ModelInterfaceError, InputShapeError
from ml.distances.metrics import EuclideanDistance, taxicab_distance, LinfinityDistance, ascii_word_dist

#very important, make sure have tests for all functions.


# Regression model: always predicts y = 2*x + 1
class LinearToyModel:
    def predict(self, X):
        X = np.asarray(X)
        return 2 * X.ravel() + 1

# Classification model: predicts class 1 if sum of features > 0, else 0
class ThresholdToyClassifier:
    def predict(self, X):
        X = np.asarray(X)
        # handle 1D or 2D input
        if X.ndim == 1:
            return np.where(X > 0, 1, 0)
        else:
            return np.where(np.sum(X, axis=1) > 0, 1, 0)
        
class BadModel:
    pass


def test_mse_perfect_fit():
    model = LinearToyModel()
    X = np.array([0, 1, 2, 3])
    y = 2 * X + 1  # exactly matches model
    mse = mean_squared_error(model, X, y)
    assert np.isclose(mse, 0.0), f"Expected 0, got {mse}"

def test_mse_with_error():
    model = LinearToyModel()
    X = np.array([0, 1, 2])
    y = np.array([1, 4, 7])  # true relation is y=2x+1, so last point is off
    mse = mean_squared_error(model, X, y)
    # predictions: [1, 3, 5], errors: [0,1,2], squared: [0,1,4], mean=5/3
    assert np.isclose(mse, 5/3), f"Expected 5/3, got {mse}"

def test_mse_single_sample():
    model = LinearToyModel()
    X = np.array([2])
    y = np.array([5])
    mse = mean_squared_error(model, X, y)
    assert np.isclose(mse, 0.0)

def test_mse_euclidean_distance_ints():
    model = LinearToyModel()
    X = np.array([0, 1, 2])
    y = np.array([1, 3, 5])  # perfect fit
    mse = mean_squared_error(model, X, y, distance=EuclideanDistance)
    assert np.isclose(mse, 0.0)

def test_mse_euclidean_distance_floats():
    model = LinearToyModel()
    X = np.array([0.0, 1.5, 2.5])
    y = 2 * X + 1
    mse = mean_squared_error(model, X, y, distance=EuclideanDistance)
    assert np.isclose(mse, 0.0)

def test_mse_taxicab_distance():
    model = LinearToyModel()
    X = np.array([0, 1, 2])
    y = np.array([2, 4, 6])  # off by [1,1,1]
    mse = mean_squared_error(model, X, y, distance=taxicab_distance)
    # Each distance = 1, squared = 1, mean = 1
    assert np.isclose(mse, 1.0)

def test_mse_linf_distance():
    model = LinearToyModel()
    X = np.array([0, 1])
    y = np.array([1, 10])  # preds = [1,3], diffs = [0,7]
    mse = mean_squared_error(model, X, y, distance=LinfinityDistance)
    # distances = [0,7], squared = [0,49], mean = 24.5
    assert np.isclose(mse, 24.5)



def test_mse_single_sample_float():
    model = LinearToyModel()
    X = np.array([2.0])
    y = np.array([5.0])
    mse = mean_squared_error(model, X, y, distance=EuclideanDistance)
    assert np.isclose(mse, 0.0)


def test_mse_model_without_predict():
    with pytest.raises(ModelInterfaceError):
        mean_squared_error(BadModel(), [1,2,3], [1,2,3])

def test_mse_empty_inputs():
    model = LinearToyModel()
    with pytest.raises(ValueError):
        mean_squared_error(model, [], [])

def test_mse_mismatched_lengths():
    model = LinearToyModel()
    X = np.array([0,1,2])
    y = np.array([1,2])  # shorter
    with pytest.raises(InputShapeError):
        mean_squared_error(model, X, y)

def test_mse_custom_distance_l1():
    model = LinearToyModel()
    X = np.array([0,1,2])
    y = np.array([1,3,5])
    # predictions are [1,3,5], so L1 distance = 0 each sample
    def l1(vec1, vec2): return float(np.sum(np.abs(np.asarray(vec1)-np.asarray(vec2))))
    mse = mean_squared_error(model, X, y, distance=l1)
    assert np.isclose(mse, 0.0)

def test_mse_custom_distance_bad_return():
    model = LinearToyModel()
    X = np.array([0,1])
    y = np.array([1,3])
    def bad_distance(a,b): return "not a number"
    with pytest.raises(ValueError):
        mean_squared_error(model, X, y, distance=bad_distance)




def test_classification_perfect():
    model = ThresholdToyClassifier()
    X = np.array([[-1], [2], [0], [3]])
    y = np.array([0, 1, 0, 1])  # matches rule exactly
    acc = classification_accuracy(model, X, y)
    assert np.isclose(acc, 1.0), f"Expected 1.0, got {acc}"

def test_classification_partial():
    model = ThresholdToyClassifier()
    X = np.array([[-2], [-1], [1], [2]])
    y = np.array([0, 1, 0, 1])  # second and third are wrong
    acc = classification_accuracy(model, X, y)
    # 2 correct out of 4
    assert np.isclose(acc, 0.5), f"Expected 0.5, got {acc}"

def test_classification_int_labels():
    model = ThresholdToyClassifier()
    X = np.array([[-1], [2], [0]])
    y = np.array([0, 1, 0])
    acc = classification_accuracy(model, X, y)
    assert np.isclose(acc, 1.0)

def test_classification_float_labels():
    # classifier outputs ints, but y are floats
    model = ThresholdToyClassifier()
    X = np.array([[-1], [2]])
    y = np.array([0.0, 1.0])  # float labels
    acc = classification_accuracy(model, X, y)
    assert np.isclose(acc, 1.0)

def test_classification_all_wrong_strings():
    class StringClassifier:
        def predict(self, X):
            return np.array(["dog", "dog"])
    model = StringClassifier()
    X = ["ignored"] * 2
    y = np.array(["cat", "cat"])
    acc = classification_accuracy(model, X, y)
    assert np.isclose(acc, 0.0)


def test_classification_string_labels():
    class StringClassifier:
        def predict(self, X):
            return np.array(["cat", "dog", "cat"])
    model = StringClassifier()
    X = ["ignored"] * 3
    y = np.array(["cat", "dog", "mouse"])
    acc = classification_accuracy(model, X, y)
    # 2 correct out of 3
    assert np.isclose(acc, 2/3)


def test_classification_all_wrong():
    model = ThresholdToyClassifier()
    X = np.array([[1],[2],[3]])
    y = np.array([0,0,0])  # all opposite
    acc = classification_accuracy(model, X, y)
    assert np.isclose(acc, 0.0)

def test_classification_model_without_predict():
    with pytest.raises(ModelInterfaceError):
        classification_accuracy(BadModel(), [1,2,3], [0,1,0])

def test_classification_empty_inputs():
    model = ThresholdToyClassifier()
    with pytest.raises(ValueError):
        classification_accuracy(model, [], [])

def test_classification_mismatched_lengths():
    model = ThresholdToyClassifier()
    X = np.array([[1],[2],[3]])
    y = np.array([0,1])  # shorter
    with pytest.raises(InputShapeError):
        classification_accuracy(model, X, y)

def test_classification_incompatible_shapes():
    model = ThresholdToyClassifier()
    X = np.array([[1],[2]])
    y = np.array([[0,1],[1,0]])  # shape mismatch with preds (n_samples,2) vs (n_samples,)
    with pytest.raises(TypeError):
        classification_accuracy(model, X, y)

def test_classification_non_array_inputs():
    model = ThresholdToyClassifier()
    X = [[-1],[2],[0]]
    y = [0,1,0]
    acc = classification_accuracy(model, X, y)
    assert 0.0 <= acc <= 1.0
