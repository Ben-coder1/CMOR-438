import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.supervised.performance import classification_accuracy
from ml.utils._errors_and_warnings._general_error_handling import ModelInterfaceError, InputShapeError


#very important, make sure have tests for all functions.



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

