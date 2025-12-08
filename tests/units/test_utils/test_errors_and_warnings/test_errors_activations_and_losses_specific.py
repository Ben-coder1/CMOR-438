import pytest
import warnings

from ml.utils._errors_and_warnings.activations_and_losses_specific import (
    _validate_activation, _validate_loss_fn
)
from ml.utils.activations import Activation, APPROVED_ACTIVATIONS
from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import APPROVED_LOSSES, APPROVED_LOSSES, Loss

# ----------------------------
# Tests for _validate_activation
# ----------------------------

def test_validate_activation_with_string_valid():
    act = APPROVED_ACTIVATIONS["relu"]
    approved = {"relu": act}
    result = _validate_activation("relu", approved)
    assert result is act

def test_validate_activation_with_string_invalid():
    approved = {"relu": APPROVED_ACTIVATIONS["relu"]}
    with pytest.raises(ValueError) as excinfo:
        _validate_activation("sigmoid", approved)
    assert "Unsupported activation name" in str(excinfo.value)

def test_validate_activation_with_activation_instance():
    act = APPROVED_ACTIVATIONS["tanh"]
    result = _validate_activation(act)
    assert result is act

def test_validate_activation_invalid_type():
    with pytest.raises(ValueError) as excinfo:
        _validate_activation(123)
    assert "Activation must be a string or an Activation instance" in str(excinfo.value)


# ----------------------------
# Tests for _validate_loss_fn
# ----------------------------

def test_validate_loss_fn_with_string_valid():
    loss = APPROVED_LOSSES["mse"]
    approved = {"mse": loss}
    result = _validate_loss_fn("mse", approved)
    assert result is loss

def test_validate_loss_fn_with_string_no_registry():
    with pytest.raises(ValueError) as excinfo:
        _validate_loss_fn("mse")
    assert "no approved_losses registry" in str(excinfo.value)

def test_validate_loss_fn_with_string_not_in_registry():
    approved = {"mae": APPROVED_LOSSES["mae"]}
    with pytest.raises(ValueError) as excinfo:
        _validate_loss_fn("mse", approved)
    assert "not in approved losses" in str(excinfo.value)

def test_validate_loss_fn_with_loss_instance_in_registry():
    loss = APPROVED_LOSSES["mse"]
    approved = {"mse": loss}
    result = _validate_loss_fn(loss, approved)
    assert result is loss

def test_validate_loss_fn_with_loss_instance_not_in_registry_warns():
    loss = Loss("bad", "bad", "bad_name")
    approved = {"mse": APPROVED_LOSSES["mse"]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_loss_fn(loss, approved)
        assert result is loss
        assert any("not in approved losses" in str(wi.message) for wi in w)

def test_validate_loss_fn_invalid_type():
    with pytest.raises(TypeError) as excinfo:
        _validate_loss_fn(123)
    assert "Expected a Loss instance or string" in str(excinfo.value)
