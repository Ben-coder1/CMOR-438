import numpy as np
import pytest
from ml.supervised_learning.ensemble_methods import RandomSubspaceEnsemble
from ml.supervised_learning.knn import KNN
from ml.supervised_learning.linear_regression import LinearRegression
from ml.utils._errors_and_warnings._general_error_handling import InvalidSignatureError



# ==========================================
# 1. TEST ARTIFACTS (Toy & Broken Models)
# ==========================================

class ToyThresholdClassifier:
    """Returns 1 if ANY single feature > 0.5, else 0."""
    def fit(self, X, y): pass
    def predict(self, X):
        if X.ndim == 1: 
            return 1 if np.max(X) > 0.5 else 0 
        max_values = np.max(X, axis=1)
        return np.where(max_values > 0.5, 1, 0)

class ToySumClassifier:
    """Returns 1 if sum of features > 1.0, else 0."""
    def fit(self, X, y): pass
    def predict(self, X):
        sums = np.sum(X, axis=1) if X.ndim > 1 else np.sum(X)
        if X.ndim == 1: return 1 if sums > 1.0 else 0
        return np.where(sums > 1.0, 1, 0)

class CrashOnTrain:
    def fit(self, X, y): raise ValueError("I crashed during training!")
    def predict(self, X): pass

class CrashOnPredict:
    def fit(self, X, y): pass
    def predict(self, X): raise ValueError("I crashed during prediction!")

# --- STATEFUL MODELS ---

class MockTrainedModel:
    """
    A mock model that starts unfitted.
    Training sets 'is_fitted' to True.
    Prediction returns a constant array of 1s if training has occurred.
    """
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        # Return 1 for every sample
        if X.ndim == 1: return np.array([1])
        return np.ones(X.shape[0])

class StatefulOffsetModel:
    """
    A model where 'fit' calculates the mean of the training target 'y'
    and stores it as an offset.
    'predict' returns 0 + offset.
    """
    def __init__(self):
        self.offset = 0.0
        self.fitted = False

    def fit(self, X, y):
        self.offset = float(np.mean(y))
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Not fitted")
        # Return the learned offset for every sample
        n_samples = 1 if X.ndim == 1 else X.shape[0]
        return np.full(n_samples, self.offset)

# ==========================================
# 2. FIXTURES
# ==========================================

@pytest.fixture
def data():
    """Basic dataset for functional testing."""
    X_train = np.array([
        [0.1, 0.1, 0.1, 0.1], 
        [0.9, 0.9, 0.9, 0.9], 
        [0.6, 0.1, 0.1, 0.1], 
        [0.1, 0.1, 0.6, 0.1]
    ])
    y_train = np.array([0, 1, 0, 0])
    X_test = np.array([
        [0.2, 0.2, 0.2, 0.2], 
        [0.8, 0.8, 0.8, 0.8]
    ])
    return X_train, y_train, X_test

# ==========================================
# 3. LOGIC & CORRECTNESS TESTS
# ==========================================

def test_init_and_deep_copy(data):
    """Test that models are added and deeply copied (modifying original doesn't affect ensemble)."""
    lr_original = LinearRegression(fit_intercept=True)
    
    ens = RandomSubspaceEnsemble(n_features_in_subset=2, task_type="regression")
    ens.add_model(lr_original, "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=2)
    
    # Mutate the original instance
    lr_original.fit_intercept = False
    
    # Assert ensemble copies retain the original state (True)
    assert ens.models[0]['model'].fit_intercept is True
    assert ens.models[1]['model'].fit_intercept is True

def test_heterogeneous_voting_logic(data):
    """Verify ensemble logic works with two DIFFERENT types of toy models simultaneously."""
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble(n_features_in_subset=4, task_type="classification")
    
    ens.add_model(ToyThresholdClassifier(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    ens.add_model(ToySumClassifier(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    
    ens.fit(X_train, y_train)
    
    # Case 1: All features high -> Both predict 1
    res1 = ens.predict([[0.8, 0.8, 0.8, 0.8]])
    assert res1[0] == 1

    # Case 2: All features low -> Both predict 0
    res2 = ens.predict([[0.1, 0.1, 0.1, 0.1]])
    assert res2[0] == 0

# --- TESTS FOR TRAINING INTEGRITY ---

def test_ensemble_fit_integrity(data):
    """
    Verifies that state update occurs in base models upon training.
    Uses MockTrainedModel which flips 'is_fitted' to True.
    """
    X_train, y_train, _ = data
    n_features = X_train.shape[1]
    
    ens = RandomSubspaceEnsemble(n_features_in_subset=2, task_type='classification')
    
    # Add 3 copies of the stateful model
    ens.add_model(MockTrainedModel(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=3)
    
    # Ensure initial state is unfitted
    assert ens.is_fitted == False
    for m in ens.models:
        assert m['model'].is_fitted == False

    # Execute training
    ens.fit(X_train, y_train)
    
    # 1. Check Ensemble State
    assert ens.is_fitted == True
    assert len(ens.models) == 3

    # 2. Check Base Model State (must be True now)
    for model_entry in ens.models:
        assert model_entry['model'].is_fitted == True
        
        # Check training data storage
        assert model_entry['stored_X_train'].shape[0] == X_train.shape[0]
        # Stored X should have subset of features (2)
        assert model_entry['stored_X_train'].shape[1] == 2
        
        # Check feature indices
        indices = model_entry['feature_indices']
        assert indices.size == 2
        assert np.all(indices < n_features)

def test_stateful_value_propagation(data):
    """
    Tests that a value calculated during training (mean of y) is correctly
    stored in the model and used during prediction.
    """
    X_train = np.zeros((5, 4)) # Dummy X
    # Target y has mean = 10.0
    y_train = np.array([10.0, 10.0, 10.0, 10.0, 10.0]) 
    
    ens = RandomSubspaceEnsemble(n_features_in_subset=2, task_type="regression")
    
    # Add the model that learns the offset
    ens.add_model(StatefulOffsetModel(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=1)
    
    ens.fit(X_train, y_train)
    
    # Predict on dummy data
    preds = ens.predict([[0,0,0,0]])
    
    # The model should return the learned mean (10.0)
    assert preds[0] == 10.0

# ==========================================
# 4. SUBSET LOGIC
# ==========================================

def test_subset_clamping(data):
    """Test that requesting more features than exist clamps to max features."""
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble(n_features_in_subset=100, task_type="classification")
    ens.add_model(ToyThresholdClassifier(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    ens.fit(X_train, y_train)
    assert len(ens.models[0]['feature_indices']) == 4

def test_subset_percentage(data):
    """Test that float inputs < 1.0 are treated as percentages."""
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble(n_features_in_subset=0.25, task_type="classification")
    ens.add_model(ToyThresholdClassifier(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    ens.fit(X_train, y_train)
    assert len(ens.models[0]['feature_indices']) == 1

# ==========================================
# 5. IMPORTED MODELS
# ==========================================

def test_lazy_knn_integration(data):
    """Test execution with the specific Lazy KNN signature."""
    X_train, y_train, X_test = data
    ens = RandomSubspaceEnsemble(n_features_in_subset=2, task_type="classification")
    ens.add_model(
        model_init=KNN(), 
        train_signature=None, 
        predict_signature="predict(target=X_TEST, classify=True, K=1, X=X_TRAIN, y=Y_TRAIN)"
    )
    ens.fit(X_train, y_train)
    preds = ens.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]

def test_linear_regression_integration(data):
    """Test execution with the specific Linear Regression signature."""
    X_train, y_train, X_test = data
    ens = RandomSubspaceEnsemble(n_features_in_subset=2, task_type="regression")
    ens.add_model(
        model_init=LinearRegression(),
        train_signature="fit(X_TRAIN, Y_TRAIN)",
        predict_signature="predict(X_TEST)"
    )
    ens.fit(X_train, y_train)
    preds = ens.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert np.issubdtype(preds.dtype, np.number)

# ==========================================
# 6. PARSING & ERROR TESTS
# ==========================================

def test_predict_before_fit(data):
    _, _, X_test = data
    ens = RandomSubspaceEnsemble()
    with pytest.raises(RuntimeError, match="must be fitted"):
        ens.predict(X_test)

def test_bad_model_training_crash(data):
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble()
    ens.add_model(CrashOnTrain(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    with pytest.raises(RuntimeError, match="I crashed during training"):
        ens.fit(X_train, y_train)

def test_bad_model_prediction_crash(data):
    X_train, y_train, X_test = data
    ens = RandomSubspaceEnsemble()
    ens.add_model(CrashOnPredict(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    ens.fit(X_train, y_train)
    with pytest.raises(RuntimeError, match="I crashed during prediction"):
        ens.predict(X_test)

def test_parsing_mistake_attribute(data):
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble()
    ens.add_model(LinearRegression(), "fit_magic(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
    with pytest.raises(InvalidSignatureError, match="Method not found"):
        ens.fit(X_train, y_train)

def test_parsing_mistake_syntax(data):
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble()
    ens.add_model(LinearRegression(), "fit(X_TRAIN, Y_TRAIN", "predict(X_TEST)")
    with pytest.raises(InvalidSignatureError, match="Syntax error"):
        ens.fit(X_train, y_train)

def test_parsing_mistake_variable_name(data):
    X_train, y_train, _ = data
    ens = RandomSubspaceEnsemble()
    ens.add_model(LinearRegression(), "fit(WRONG_VAR, Y_TRAIN)", "predict(X_TEST)")
    with pytest.raises(InvalidSignatureError, match="Unknown variable"):
        ens.fit(X_train, y_train)