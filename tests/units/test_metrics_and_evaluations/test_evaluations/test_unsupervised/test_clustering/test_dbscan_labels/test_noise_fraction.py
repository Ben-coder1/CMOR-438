import numpy as np
import pytest
from ml.metrics_and_evaluations.evaluation.unsupervised.clustering.dbscan_labels.noise_fraction import compute_noise_fraction

# ----------------------
# Normal cases
# ----------------------

def test_noise_fraction_basic():
    labels = np.array([0, 0, 1, 1, -1, -1])
    result = compute_noise_fraction(labels)
    assert np.isclose(result, 2/6)

def test_noise_fraction_no_noise():
    labels = np.array([0, 1, 1, 2])
    result = compute_noise_fraction(labels)
    assert np.isclose(result, 0.0)

def test_noise_fraction_all_noise():
    labels = np.array([-1, -1, -1])
    result = compute_noise_fraction(labels)
    assert np.isclose(result, 1.0)

def test_noise_fraction_with_negatives_and_clusters():
    labels = np.array([-1, 0, -1, 2, 3])
    result = compute_noise_fraction(labels)
    assert np.isclose(result, 2/5)

def test_noise_fraction_with_floats_and_ints():
    labels = np.array([-1.0, 0, 1, -1])
    result = compute_noise_fraction(labels)
    assert np.isclose(result, 2/4)


# ----------------------
# Edge cases and errors
# ----------------------

def test_noise_fraction_empty_array_raises():
    labels = np.array([])
    with pytest.raises(ValueError):
        compute_noise_fraction(labels)

def test_noise_fraction_none_input_raises():
    with pytest.raises(ValueError):
        compute_noise_fraction(None)

def test_noise_fraction_wrong_shape_raises():
    labels = np.array([[0, -1], [1, 2]])  # 2D instead of 1D
    with pytest.raises(ValueError):
        compute_noise_fraction(labels)

def test_noise_fraction_with_nan_raises():
    labels = np.array([0, -1, np.nan])
    with pytest.raises(ValueError):
        compute_noise_fraction(labels)

def test_noise_fraction_non_numeric_input_raises():
    labels = np.array(["a", "b", "c"])
    with pytest.raises(TypeError):
        compute_noise_fraction(labels)
