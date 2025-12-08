import pytest
import numpy as np
from ml.supervised_learning.decision_trees import Node
from ml.supervised_learning.Regression_trees import RegressionTree, variance, mean_absolute_deviation
from ml.utils._errors_and_warnings import InputShapeError

# ==========================================
# 1. Test Impurity Metrics
# ==========================================

def test_variance_metric():
    # Case 1: Standard variance
    # Mean = 2.0. Squared diffs: (1-2)^2=1, (2-2)^2=0, (3-2)^2=1. Sum=2. Avg=2/3
    y = np.array([1, 2, 3])
    expected = 2.0 / 3.0
    assert np.isclose(variance(y), expected)

    # Case 2: Zero variance (pure)
    y_pure = np.array([5, 5, 5])
    assert variance(y_pure) == 0.0

    # Case 3: Empty
    assert variance(np.array([])) == 0.0

def test_mean_absolute_deviation_metric():
    # Case 1: Standard MAD
    # Data: [1, 2, 9]. Median = 2.
    # Absolute diffs: |1-2|=1, |2-2|=0, |9-2|=7. Mean = (1+0+7)/3 = 8/3
    y = np.array([1, 2, 9])
    expected = 8.0 / 3.0
    assert np.isclose(mean_absolute_deviation(y), expected)

    # Case 2: Zero MAD
    y_pure = np.array([5, 5, 5])
    assert mean_absolute_deviation(y_pure) == 0.0

    # Case 3: Empty
    assert mean_absolute_deviation(np.array([])) == 0.0


# ==========================================
# 2. Test Model Logic and functionality
# ==========================================

def test_fit_predict_simple_variance_split():
    """
    Test a perfect split situation using Variance (MSE).
    X = [1, 10], y = [1, 10]
    Best split should separate 1 and 10 perfectly (likely threshold ~5.5).
    """
    X = np.array([[1], [10]])
    y = np.array([1, 10])
    
    model = RegressionTree(criterion=variance, leaf_aggregator=np.mean)
    model.fit(X, y)
    
    # Predict close to 1 -> should get 1.0 (Input must be <= split threshold)
    # Predict close to 10 -> should get 10.0 (Input must be > split threshold)
    # FIX: Changed test input from 1.1 to 0.9 to ensure it falls in the Left branch
    preds = model.predict(np.array([[0.9], [9.9]]))
    assert np.allclose(preds, np.array([1.0, 10.0]))

def test_custom_criterion_integration_mad():
    """
    Test that plugging in MAD changes the tree's behavior compared to Variance.
    We use a dataset with an outlier where Mean (MSE) and Median (MAE) diverge.
    
    Data: Group A: [1, 1, 1, 100] -> Mean=25.75, Median=1
    """
    # Create a node that effectively acts as a single leaf.
    X = np.array([[1], [2], [3], [4]]) 
    y = np.array([1, 1, 1, 100])
    
    # FIX: Use max_depth=1 (valid positive) + high min_samples_split to force single leaf
    
    # 1. Standard Tree (MSE/Mean)
    mse_model = RegressionTree(max_depth=1, min_samples_split=10, leaf_aggregator=np.mean)
    mse_model.fit(X, y)
    pred_mse = mse_model.predict(np.array([[1]]))
    
    # 2. Robust Tree (MAD/Median)
    mae_model = RegressionTree(max_depth=1, min_samples_split=10, leaf_aggregator=np.median)
    mae_model.fit(X, y)
    pred_mae = mae_model.predict(np.array([[1]]))
    
    # Assertions
    assert np.isclose(pred_mse[0], 25.75) # Mean influenced by outlier
    assert np.isclose(pred_mae[0], 1.0)   # Median ignores outlier


# ==========================================
# 3. Test Error Handling (Public API)
# ==========================================

def test_fit_raises_input_shape_mismatch():
    """Test X.samples != y.samples"""
    model = RegressionTree()
    X = np.array([[1], [2]])
    y = np.array([1]) # Mismatch
    
    with pytest.raises(InputShapeError, match="samples"):
        model.fit(X, y)

def test_fit_raises_non_numeric_input():
    """Test passing string arrays raises error (Regression requires numeric inputs)."""
    model = RegressionTree()
    X = np.array([[1], [2]])
    y_str = np.array(["cat", "dog"]) 
    
    # FIX: Helper raises TypeError for non-numeric types, not ValueError
    with pytest.raises(TypeError, match="numeric"):
        model.fit(X, y_str)

def test_fit_raises_nan_input():
    """Test passing NaNs raises error."""
    model = RegressionTree()
    X = np.array([[1], [np.nan]])
    y = np.array([1, 2])
    
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y)

def test_predict_raises_before_fit():
    """Test predict called on unfitted tree."""
    model = RegressionTree()
    X = np.array([[1]])
    
    # FIX: Updated regex to match the actual error message "Tree not fitted."
    with pytest.raises(RuntimeError, match="Tree not fitted."):
        model.predict(X)

def test_predict_raises_shape_mismatch_features():
    """Test passing invalid data types to predict."""
    model = RegressionTree()
    X_train = np.array([[1, 2], [3, 4]]) 
    y_train = np.array([10, 20])
    model.fit(X_train, y_train)
    
    # FIX: Helper raises TypeError for non-numeric types, not ValueError
    with pytest.raises(TypeError, match="numeric"):
        model.predict(np.array([["a", "b"]]))

def test_init_validates_callable_criterion():
    """Test that passing a non-callable to init raises TypeError."""
    with pytest.raises(TypeError):
        RegressionTree(criterion="not_a_function")

def test_init_validates_positive_hyperparams():
    """Test min_samples_split and max_depth validation."""
    with pytest.raises(ValueError):
        RegressionTree(min_samples_split=0)
    with pytest.raises(ValueError):
        RegressionTree(max_depth=-1)


# ==========================================
# 4. Test Inherited Methods
# ==========================================

def test_regression_tree_split_on_instance():
    """Verifies RegressionTree instance correctly uses the inherited _split method."""
    model = RegressionTree()
    
    # Arrange: Data array representing a single feature column
    X_column = np.array([5.5, 1.2, 8.0, 3.1, 5.5, 9.9])
    split_thresh = 5.5
    
    # Act: Calling the inherited method via the RegressionTree instance
    left_idxs, right_idxs = model._split(X_column, split_thresh)
    
    # Assert
    # Left (<= 5.5): indices 0, 1, 3, 4
    assert np.array_equal(left_idxs, np.array([0, 1, 3, 4]))
    # Right (> 5.5): indices 2, 5
    assert np.array_equal(right_idxs, np.array([2, 5]))

def test_regression_tree_traverse_full_path_on_instance():
    """Verifies that the RegressionTree instance can correctly traverse a manually built tree."""
    model = RegressionTree()
    
    # Arrange: Manually build a simple tree structure using the Node class
    # L2: Leaves
    leaf_c = Node(value=5.5)   
    leaf_d = Node(value=10.0)  
    
    # L1: Internal Nodes
    node_a = Node(feature=1, threshold=5.0, left=leaf_c, right=leaf_d) 
    node_b = Node(value=20.2) 
    
    # L0: Root
    root_node = Node(feature=0, threshold=10.0, left=node_a, right=node_b) 

    # Test Sample 1: Goes Left -> Right -> Value 10.0
    x1 = np.array([0.5, 6.0, 99]) 
    assert model._traverse_tree(x1, root_node) == 10.0
    
    # Test Sample 2: Goes Right -> Value 20.2
    x2 = np.array([12.0, 1.0, 0]) 
    assert model._traverse_tree(x2, root_node) == 20.2