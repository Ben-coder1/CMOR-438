from typing import Callable, Optional, Tuple, Union
import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    InputShapeError, 
    _ensure_array_like, 
    _ensure_callable, 
    _ensure_no_nan, 
    _ensure_non_empty, 
    _ensure_numeric_array, 
    _ensure_positive_numeric, 
    _check_sample_shapes_match
)

# --- 2. Impurity Metrics ---

def entropy(y: np.ndarray) -> float:
    r"""
    Calculate the entropy of a label array.

    Formula: $H(S) = - \sum (p_i * log_2(p_i))$

    Parameters
    ----------
    y : np.ndarray
        The target vector of shape (n_samples,).

    Returns
    -------
    float
        The entropy value. 0.0 indicates a perfectly pure node.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 0, 0])
    >>> float(entropy(y))
    1.0
    >>> y_pure = np.array([1, 1, 1])
    >>> abs(float(entropy(y_pure))) < 1e-10
    True 
    """
    _ensure_non_empty(y, "target_vector_for_entropy")
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    
    # We use a mask to avoid log(0) calculation
    probabilities = probabilities[probabilities > 0]
    
    return -np.sum(probabilities * np.log2(probabilities))

def gini(y: np.ndarray) -> float:
    r"""
    Calculate the Gini impurity of a label array.

    Formula: $G(S) = 1 - \sum (p_i^2)$

    Parameters
    ----------
    y : np.ndarray
        The target vector of shape (n_samples,).

    Returns
    -------
    float
        The Gini impurity value. 0.0 indicates a perfectly pure node.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 0, 0])
    >>> float(gini(y))
    0.5
    >>> y_pure = np.array([1, 1, 1])
    >>> float(gini(y_pure))
    0.0
    """
    _ensure_non_empty(y, "target_vector_for_gini")
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

# --- 3. Graph/Node Structure ---

class Node:
    """
    A structural unit (vertex) in the decision tree graph.
    
    Attributes
    ----------
    feature : int or None
        The index of the feature column used to split this node. 
        None if this is a leaf node.
    threshold : float or None
        The numeric value used to split the feature (<= threshold goes left). 
        None if this is a leaf node.
    left : Node or None
        The left child node (satisfies condition).
    right : Node or None
        The right child node (does not satisfy condition).
    value : any or None
        The predicted class label if this is a leaf node. None otherwise.
    """
    def __init__(self, feature: int = None, threshold: float = None, 
                 left: 'Node' = None, right: 'Node' = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """Check if the node is a leaf (terminal) node."""
        return self.value is not None

class BaseDecisionTree:
    """
    Base class providing universal, structural methods for decision tree
    implementations (Classification and Regression).

    This class contains logic that is identical for all tree types, primarily
    focused on node splitting and tree traversal, allowing specialized child
    classes (like DecisionTree and RegressionTree) to inherit these core functions.
    """
    
    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Splits a single feature column based on a given threshold value.

        This function performs the binary partitioning of samples into two groups
        based on a simple inequality check: $X_{column} \le threshold$ (left)
        or $X_{column} > threshold$ (right).

        Parameters
        ----------
        X_column : np.ndarray
            The column vector of feature values for the current node, shape (n_samples,).
        split_thresh : float
            The numerical threshold used to define the split.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two 1D NumPy arrays:
            - left_idxs: Indices where $X_{column} \le split\_thresh$.
            - right_idxs: Indices where $X_{column} > split\_thresh$.

        Examples
        --------
        >>> X_col = np.array([10, 5, 20, 15, 5])
        >>> threshold = 10
        >>> # Mock instance needed only to call the method
        >>> tree = type('MockTree', (BaseDecisionTree,), {})() 
        >>> left, right = tree._split(X_col, threshold)
        >>> left.tolist()
        [0, 1, 4]
        >>> right.tolist()
        [2, 3]
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x: np.ndarray, node: Node):
        """
        Recursively traverses the decision tree graph for a single input sample (x).

        The function follows the split criteria (feature index and threshold) from the 
        current node until it reaches a leaf node, returning the predicted value 
        stored there.

        Parameters
        ----------
        x : np.ndarray
            A single sample vector of shape (n_features,).
        node : Node
            The current node being visited (starts at the root).

        Returns
        -------
        Union[any, float]
            The predicted value (class label for classification, mean/median for regression)
            stored at the terminal leaf node.
            
        Examples
        --------
        >>> # Setup: Create a simple tree (Root splits on feature 0 at 5.0)
        >>> node_a = Node(value=10) # Left Leaf
        >>> node_b = Node(value=20) # Right Leaf
        >>> root = Node(feature=0, threshold=5.0, left=node_a, right=node_b)
        >>> # Mock instance needed only to call the method
        >>> tree = type('MockTree', (BaseDecisionTree,), {})() 
        
        >>> # Test 1: Go Left (Feature 0 value is 4.0 <= 5.0)
        >>> x1 = np.array([4.0, 1.0]) 
        >>> tree._traverse_tree(x1, root)
        10
        
        >>> # Test 2: Go Right (Feature 0 value is 6.0 > 5.0)
        >>> x2 = np.array([6.0, 0.5]) 
        >>> tree._traverse_tree(x2, root)
        20
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# --- 4. Decision Tree Classifier ---

class DecisionTree(BaseDecisionTree):
    r"""
    A decision tree classifier implemented from scratch using NumPy and a graph/node structure.
    
    This implementation supports binary splitting based on Information Gain (Entropy) 
    or Gini Impurity. It builds the tree recursively by selecting the locally optimal 
    split at every node.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
        Must be a positive integer.
    max_depth : int, default=100
        The maximum depth of the tree. Limits the number of sequential splits to 
        prevent overfitting. Must be a positive integer.
    criterion : {'entropy', 'gini'} or Callable, default='entropy'
        The function to measure the quality of a split.
        - 'entropy': Uses Information Gain ($H(S)$).
        - 'gini': Uses Gini Impurity ($1 - \sum p_i^2$).
        - Callable: A custom function that takes a target array `y` and returns a float.

    Attributes
    ----------
    root : Node or None
        The root node of the decision tree graph after fitting.
    criterion : Callable
        The resolved impurity function used during training.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 1], [3, 2], [3, 3], [5, 5]])
    >>> y = np.array([0, 0, 1, 1, 1])
    >>> dt = DecisionTree(max_depth=3, criterion='entropy')
    >>> _ = dt.fit(X, y)
    >>> dt.predict(np.array([[5, 4]]))
    array([1])
    """
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, 
                 criterion: Union[str, Callable] = 'entropy'):
        self.min_samples_split = _ensure_positive_numeric(min_samples_split, "min_samples_split")
        self.max_depth = _ensure_positive_numeric(max_depth, "max_depth")
        
        # Resolve criterion
        if isinstance(criterion, str):
            if criterion == 'entropy':
                self.criterion = entropy
            elif criterion == 'gini':
                self.criterion = gini
            else:
                raise ValueError(f"Unknown criterion string '{criterion}'. Supported: 'entropy', 'gini'.")
        else:
            self.criterion = _ensure_callable(criterion, "criterion")
            
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Build (train) the decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Must be numeric.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : DecisionTree
            Returns the instance of the fitted model.

        Raises
        ------
        ValueError
            If input arrays contain NaNs or are empty.
        InputShapeError
            If the number of samples in X and y do not match.

        Examples
        --------
        >>> dt = DecisionTree()
        >>> X_train = np.array([[0, 0], [1, 1]])
        >>> y_train = np.array([0, 1])
        >>> _ = dt.fit(X_train, y_train)
        """
        # Validate Inputs using strict helpers
        X_clean = _ensure_numeric_array(X, name="X", ndim=2)
        X_clean = _ensure_no_nan(X_clean, name="X")
        
        y_clean = _ensure_array_like(y, name="y")
        y_clean = _ensure_no_nan(y_clean, name="y")
        
        # Basic consistency check (rows must match)
        if X_clean.shape[0] != y_clean.shape[0]:
            raise InputShapeError(f"X has {X_clean.shape[0]} samples but y has {y_clean.shape[0]}.")

        self.root = self._grow_tree(X_clean, y_clean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            The predicted class labels.

        Raises
        ------
        RuntimeError
            If the tree has not been fitted yet.

        Examples
        --------
        >>> dt = DecisionTree()
        >>> # Using more samples ensures a robust split around 0.5
        >>> X_train = np.array([[0.1], [0.1], [0.9], [0.9]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> _ = dt.fit(X_train, y_train)
        >>> dt.predict(np.array([[0.1], [0.9]]))
        array([0, 1])
        """
        # Validate Inputs
        if self.root is None:
            raise RuntimeError("The tree has not been fitted yet. Call fit() first.")
            
        X_clean = _ensure_numeric_array(X, name="X", ndim=2)
        X_clean = _ensure_no_nan(X_clean, name="X")
        
        predictions = np.array([self._traverse_tree(x, self.root) for x in X_clean])
        
        # Consistency check for output shape using helper
        try:
             # creating a dummy comparison array of same length to satisfy the helper's logic 
             dummy_y = np.zeros(X_clean.shape[0])
             _check_sample_shapes_match(dummy_y, predictions)
        except InputShapeError:
             raise RuntimeError("Internal prediction shape mismatch.")

        return predictions

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the decision tree graph.

        Parameters
        ----------
        X : np.ndarray
            Input features for the current node.
        y : np.ndarray
            Target labels for the current node.
        depth : int
            The current depth of the recursion.

        Returns
        -------
        Node
            The root node of the constructed subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping Criteria: Max depth reached, pure node, or not enough samples
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        feat_idx, threshold = self._best_split(X, y, n_features)

        # If no valid split found (gain is 0 or cannot split), make it a leaf
        if feat_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create split
        left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)
        
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=feat_idx, threshold=threshold, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Determine the best feature and threshold to split the current node.
        Iterates through all features and all unique values to maximize Information Gain.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        
        parent_impurity = self.criterion(y)

        # Loop through all features
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Iterate through all unique values of the feature as thresholds
            for threshold in thresholds:
                # Calculate Information Gain
                gain = self._information_gain(y, X_column, threshold, parent_impurity)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, 
                          threshold: float, parent_impurity: float) -> float:
        r"""
        Calculate Information Gain for a potential split.

        Equation: 
        $IG = Impurity(parent) - \sum \frac{N_{child}}{N_{parent}} \times Impurity(child)$

        Returns
        -------
        float
            The calculated information gain. Returns 0.0 if the split creates empty children.
        """
        # Generate split indices
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        # Calculate weighted average impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        e_l = self.criterion(y[left_idxs])
        e_r = self.criterion(y[right_idxs])
        
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_impurity - child_impurity


    def _most_common_label(self, y: np.ndarray):
        """Returns the most frequent class label in the array `y`."""
        if len(y) == 0:
            return None
        # Using numpy to find mode
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
