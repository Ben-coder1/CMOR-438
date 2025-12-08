import pytest
import numpy as np
from ml.supervised_learning.decision_trees import BaseDecisionTree, DecisionTree, entropy, gini, Node
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError


#node tests

def test_node_initialization_as_internal_node():
    """
    Tests that a newly created internal node has a feature, threshold, 
    and children, but no final prediction value.
    """
    # Arrange: Create dummy child nodes
    left_child = Node(value=5)
    right_child = Node(value=10)
    
    # Act: Create an internal node (a split point)
    internal_node = Node(feature=2, threshold=5.5, left=left_child, right=right_child)
    
    # Assert
    assert internal_node.feature == 2
    assert internal_node.threshold == 5.5
    assert internal_node.left is left_child
    assert internal_node.right is right_child
    assert internal_node.value is None

def test_node_initialization_as_leaf_node():
    """
    Tests that a newly created leaf node has a prediction value and 
    no split-related attributes.
    """
    # Act: Create a leaf node with a classification value (0)
    leaf_node_class = Node(value=0)
    # Act: Create a leaf node with a regression value (12.5)
    leaf_node_reg = Node(value=12.5)

    # Assert Classification Leaf
    assert leaf_node_class.feature is None
    assert leaf_node_class.threshold is None
    assert leaf_node_class.left is None
    assert leaf_node_class.right is None
    assert leaf_node_class.value == 0

    # Assert Regression Leaf
    assert leaf_node_reg.value == 12.5

def test_is_leaf_method_for_internal_node():
    """
    Tests that is_leaf() returns False for a node designed for splitting.
    """
    # Arrange: Internal node (value is None)
    internal_node = Node(feature=0, threshold=1.0)
    
    # Assert
    assert internal_node.is_leaf() is False

def test_is_leaf_method_for_leaf_node():
    """
    Tests that is_leaf() returns True for a terminal node (value is set).
    """
    # Arrange: Leaf node (value is set to 5)
    leaf_node = Node(value=5)
    
    # Assert
    assert leaf_node.is_leaf() is True

def test_is_leaf_method_for_zero_value():
    """
    Tests that is_leaf() works correctly even if the prediction is 0 or False.
    """
    # Arrange: Leaf node with prediction 0
    leaf_node = Node(value=0)
    
    # Assert: 0 is not None, so it is a leaf.
    assert leaf_node.is_leaf() is True



# --- Mock Implementation tests for BaseDecisionTree---

class MockTree(BaseDecisionTree):
    """
    A minimal temporary class inheriting BaseDecisionTree to facilitate direct
    testing of the shared instance methods (_split, _traverse_tree).
    """
    pass

# --- Test Functions ---

def test_split_basic_functionality():
    """Tests the _split method for correct indexing."""
    mock_tree = MockTree()
    X_column = np.array([10, 5, 20, 15, 5])
    split_thresh = 10
    
    left_idxs, right_idxs = mock_tree._split(X_column, split_thresh)
    
    # Expected: Indices 0 (10), 1 (5), 4 (5) go left (<= 10)
    assert np.array_equal(left_idxs, np.array([0, 1, 4]))
    # Expected: Indices 2 (20), 3 (15) go right (> 10)
    assert np.array_equal(right_idxs, np.array([2, 3]))

def test_split_edge_case_no_right_split():
    """Tests when all samples satisfy the split (no right children)."""
    mock_tree = MockTree()
    X_column = np.array([1, 2, 3])
    split_thresh = 10
    
    left_idxs, right_idxs = mock_tree._split(X_column, split_thresh)
    
    assert len(left_idxs) == 3
    assert len(right_idxs) == 0

def test_traverse_tree_simple_path():
    """Tests traversal in a two-leaf tree."""
    mock_tree = MockTree()
    
    left_leaf = Node(value=0)
    right_leaf = Node(value=1)
    root_node = Node(feature=1, threshold=0.5, left=left_leaf, right=right_leaf)
    
    # Sample 1: Feature 1 = 0.4 (Goes Left: 0.4 <= 0.5)
    x1 = np.array([99, 0.4]) 
    # Sample 2: Feature 1 = 0.6 (Goes Right: 0.6 > 0.5)
    x2 = np.array([10, 0.6])

    assert mock_tree._traverse_tree(x1, root_node) == 0
    assert mock_tree._traverse_tree(x2, root_node) == 1

def test_traverse_tree_deep_path():
    """Tests traversal in a multi-level tree."""
    mock_tree = MockTree()
    
    # Tree Structure (reusing Node):
    # L1: Root (F0 @ 1.0)
    # L2:   Left (F1 @ 5.0), Right Leaf (Value 20)
    # L3:     Left Leaf (Value 10), Right Leaf (Value 15)
    
    # L3
    leaf_c = Node(value=10)
    leaf_d = Node(value=15)
    # L2
    node_b = Node(value=20)
    node_a = Node(feature=1, threshold=5.0, left=leaf_c, right=leaf_d)
    # L1
    root_node = Node(feature=0, threshold=1.0, left=node_a, right=node_b)

    # Path 1: F0=0.5 (Left) -> F1=6.0 (Right) -> Value 15
    x1 = np.array([0.5, 6.0]) 
    assert mock_tree._traverse_tree(x1, root_node) == 15
    
    # Path 2: F0=2.0 (Right) -> Value 20
    x2 = np.array([2.0, 1.0]) 
    assert mock_tree._traverse_tree(x2, root_node) == 20




@pytest.fixture
def int_dataset():
    """Simple deterministic integer dataset (AND-like logic)."""
    # [0,0]->0, [1,0]->0, [0,1]->0, [1,1]->1
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

@pytest.fixture
def float_dataset():
    """Simple deterministic float dataset (Threshold logic)."""
    # Class 0 if x <= 5.0, Class 1 if x > 5.0
    X = np.array([[1.5], [2.5], [7.5], [8.5]])
    y = np.array([0, 0, 1, 1])
    return X, y

# --- 1. Impurity Metric Tests (Criteria) ---

def test_entropy_logic():
    """Test entropy calculation on known distributions."""
    # Pure node: only one class
    y_pure = np.array([1, 1, 1])
    assert entropy(y_pure) == 0.0

    # 50/50 split: - (0.5 log2 0.5 + 0.5 log2 0.5) = - (-0.5 - 0.5) = 1.0
    y_split = np.array([0, 0, 1, 1])
    assert np.isclose(entropy(y_split), 1.0)

    # 3 classes, equal distribution
    y_multi = np.array([0, 1, 2])
    # - 3 * (1/3 * log2(1/3)) = log2(3) ≈ 1.585
    expected = -np.sum([(1/3) * np.log2(1/3)] * 3)
    assert np.isclose(entropy(y_multi), expected)

def test_gini_logic():
    """Test gini calculation on known distributions."""
    # Pure node: 1 - sum(1^2) = 0
    y_pure = np.array([1, 1, 1])
    assert gini(y_pure) == 0.0

    # 50/50 split: 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5
    y_split = np.array([0, 0, 1, 1])
    assert np.isclose(gini(y_split), 0.5)

    # 4 classes, equal distribution: 1 - 4*(0.25^2) = 1 - 0.25 = 0.75
    y_multi = np.array([0, 1, 2, 3])
    assert np.isclose(gini(y_multi), 0.75)

# --- 2. Internal Helper Method Tests ---

def test_split_indices(int_dataset):
    """Test the low-level _split method."""
    dt = DecisionTree()
    X = np.array([10, 20, 10, 30])
    
    # Split at 15. Expected: indices 0,2 on left (10), 1,3 on right (20, 30)
    left, right = dt._split(X, 15)
    
    np.testing.assert_array_equal(sorted(left), [0, 2])
    np.testing.assert_array_equal(sorted(right), [1, 3])

def test_most_common_label():
    """Test finding the mode of the target array."""
    dt = DecisionTree()
    
    # Simple majority
    y = np.array([0, 1, 1, 1, 0])
    assert dt._most_common_label(y) == 1
    
    # Tie breaking (usually lowest value or first encounter depending on numpy implementation, 
    # np.unique usually sorts values so it picks the smaller class ID in a tie)
    y_tie = np.array([0, 0, 1, 1])
    assert dt._most_common_label(y_tie) in [0, 1] 

    # Empty
    assert dt._most_common_label(np.array([])) is None

def test_information_gain_calculation():
    """Test the manual calculation of information gain."""
    dt = DecisionTree(criterion='entropy')
    
    # Parent: 50/50 split (Entropy 1.0)
    # y = [0, 0, 1, 1]
    y = np.array([0, 0, 1, 1])
    X_col = np.array([1, 1, 2, 2]) # Perfect splitter at 1.5
    
    parent_imp = entropy(y) # 1.0
    
    # If we split at 1.5:
    # Left (<=1.5): [0, 0] -> Entropy 0
    # Right (>1.5): [1, 1] -> Entropy 0
    # Weighted avg child impurity = 0
    # IG = 1.0 - 0 = 1.0
    ig = dt._information_gain(y, X_col, 1.5, parent_imp)
    assert np.isclose(ig, 1.0)

    # Bad split (puts everything in one side)
    ig_bad = dt._information_gain(y, X_col, 0.5, parent_imp)
    # Child Left: empty, Child Right: [0,0,1,1] (Entropy 1.0)
    # Implementation usually returns 0 gain for empty splits
    assert ig_bad == 0.0

def test_best_split_finder():
    """Test that _best_split finds the obvious optimal split."""
    dt = DecisionTree(criterion='entropy')
    
    # X construction:
    # Feature 0: [1, 1, 0, 0] -> No clear correlation with y=[0, 1, 0, 1]
    # Feature 1: [10, 20, 10, 20] -> Perfectly correlates with y
    
    # Samples:
    # 1. Feat0=1, Feat1=10 -> Class 0
    # 2. Feat0=1, Feat1=20 -> Class 1
    # 3. Feat0=0, Feat1=10 -> Class 0
    # 4. Feat0=0, Feat1=20 -> Class 1
    
    X = np.array([
        [1, 10], 
        [1, 20],
        [0, 10],
        [0, 20]
    ])
    y = np.array([0, 1, 0, 1])
    
    # Feature 0 splits:
    # If split < 0.5: Left(y=[0,1]), Right(y=[0,1]) -> Gain 0
    
    # Feature 1 splits:
    # If split < 15: Left(y=[0,0]), Right(y=[1,1]) -> Gain 1.0 (Perfect)

    feat_idx, thresh = dt._best_split(X, y, n_features=2)
    
    assert feat_idx == 1 
    assert thresh is not None

# --- 3. Initialization Tests ---

def test_init_valid_defaults():
    dt = DecisionTree()
    assert dt.min_samples_split == 2
    assert dt.max_depth == 100
    assert dt.root is None

def test_init_invalid_numeric_params():
    with pytest.raises(ValueError):
        DecisionTree(min_samples_split=0)
    with pytest.raises(ValueError):
        DecisionTree(max_depth=-5)

def test_init_invalid_numeric_types():
    with pytest.raises(TypeError):
        DecisionTree(min_samples_split="two")

def test_init_criterion_validation():
    dt_ent = DecisionTree(criterion='entropy')
    assert callable(dt_ent.criterion)
    
    with pytest.raises(ValueError):
        DecisionTree(criterion='magic')
    with pytest.raises(TypeError):
        DecisionTree(criterion=123)

# --- 4. Public API Logic Tests ---

def test_fit_predict_integers_entropy(int_dataset):
    X, y = int_dataset
    dt = DecisionTree(criterion='entropy')
    dt.fit(X, y)
    preds = dt.predict(X)
    np.testing.assert_array_equal(preds, y)

def test_fit_predict_floats_gini(float_dataset):
    X, y = float_dataset
    dt = DecisionTree(criterion='gini')
    dt.fit(X, y)
    preds = dt.predict(X)
    np.testing.assert_array_equal(preds, y)

def test_custom_criterion_function(int_dataset):
    X, y = int_dataset
    def custom_gini(y_subset):
        _, counts = np.unique(y_subset, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs**2)

    dt = DecisionTree(criterion=custom_gini)
    dt.fit(X, y)
    preds = dt.predict(X)
    np.testing.assert_array_equal(preds, y)

# --- 5. Error Handling Tests ---

def test_fit_shape_mismatch(int_dataset):
    X, _ = int_dataset
    y_bad = np.array([0, 1])
    dt = DecisionTree()
    with pytest.raises(InputShapeError):
        dt.fit(X, y_bad)

def test_fit_non_numeric_X():
    dt = DecisionTree()
    X_str = np.array([["a", "b"], ["c", "d"]])
    y = np.array([0, 1])
    with pytest.raises(TypeError):
        dt.fit(X_str, y)

def test_fit_nan_values_X():
    dt = DecisionTree()
    X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        dt.fit(X_nan, y)

def test_fit_empty_data():
    dt = DecisionTree()
    X_empty = np.array([])
    y_empty = np.array([])
    with pytest.raises(ValueError):
        dt.fit(X_empty, y_empty)

def test_fit_wrong_dimension_X():
    dt = DecisionTree()
    X_1d = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        dt.fit(X_1d, y)

def test_predict_unfitted(int_dataset):
    X, _ = int_dataset
    dt = DecisionTree()
    with pytest.raises(RuntimeError):
        dt.predict(X)

def test_predict_nan_values(int_dataset):
    X, y = int_dataset
    dt = DecisionTree()
    dt.fit(X, y)
    X_bad = np.array([[np.nan, 0]])
    with pytest.raises(ValueError):
        dt.predict(X_bad)

def test_predict_non_numeric(int_dataset):
    X, y = int_dataset
    dt = DecisionTree()
    dt.fit(X, y)
    X_str = np.array([["a", "b"]])
    with pytest.raises(TypeError):
        dt.predict(X_str)

def test_predict_dimension_mismatch(int_dataset):
    X, y = int_dataset
    dt = DecisionTree()
    dt.fit(X, y)
    X_flat = np.array([0, 0])
    with pytest.raises(ValueError):
        dt.predict(X_flat)