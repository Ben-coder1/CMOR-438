
from collections.abc import Callable
from typing import Optional, Tuple
import numpy as np
from ml.supervised_learning.decision_trees import BaseDecisionTree, Node
from ml.utils._errors_and_warnings._general_error_handling import InputShapeError, _ensure_callable, _ensure_positive_numeric, _ensure_numeric_array, _ensure_no_nan

# --- 1. Impurity Metrics ---

def variance(y: np.ndarray) -> float:
    r"""
    Calculate the variance of the target array. 
    
    This is used as the impurity metric (cost function) for Mean Squared Error (MSE) 
    reduction in regression trees, where $Gain = Var(Parent) - \sum w_i Var(Child_i)$.

    Parameters
    ----------
    y : np.ndarray
        The target vector of shape (n_samples,).

    Returns
    -------
    float
        The variance value. Returns 0.0 if the array is empty.
        
    Examples
    --------
    >>> float(variance(np.array([1, 1, 2, 2])))
    0.25
    """
    if len(y) == 0: return 0.0
    return np.var(y)

def mean_absolute_deviation(y: np.ndarray) -> float:
    r"""
    Calculate the Mean Absolute Deviation (MAD) of the target array relative to the median.
    
    This is an alternative impurity metric often used when the goal is to minimize 
    Mean Absolute Error (MAE), as the median minimizes the sum of absolute residuals.

    Parameters
    ----------
    y : np.ndarray
        The target vector of shape (n_samples,).

    Returns
    -------
    float
        The Mean Absolute Deviation value. Returns 0.0 if the array is empty.
        
    Examples
    --------
    >>> float(mean_absolute_deviation(np.array([1, 2, 9])))
    2.6666666666666665
    """
    if len(y) == 0: return 0.0
    median = np.median(y)
    return np.mean(np.abs(y - median))

# --- 2. Regression Tree ---

class RegressionTree(BaseDecisionTree):
    r"""
    A decision tree model designed for regression tasks.
    
    It constructs the tree by recursively selecting splits that maximize the 
    reduction in impurity (gain), using a generalized criterion and predicting 
    the aggregated value (e.g., mean or median) at the leaf nodes.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_depth : int, default=100
        The maximum depth of the tree.
    criterion : Callable, default=variance
        The function used to measure the quality of a split (impurity). 
        It must take a target array `y` and return a single float impurity score.
    leaf_aggregator : Callable, default=np.mean
        The function used to determine the prediction value at a leaf node. 
        It must take a target array `y` and return a single float. (e.g., np.mean 
        for variance criterion, np.median for MAD criterion).
    """
    def __init__(self, 
                 min_samples_split: int = 2, 
                 max_depth: int = 100, 
                 criterion: Callable[[np.ndarray], float] = variance,
                 leaf_aggregator: Callable[[np.ndarray], float] = np.mean):
        self.min_samples_split = _ensure_positive_numeric(min_samples_split, "min_samples_split")
        self.max_depth = _ensure_positive_numeric(max_depth, "max_depth")
        self.criterion = _ensure_callable(criterion, "criterion")
        self.leaf_aggregator = _ensure_callable(leaf_aggregator, "leaf_aggregator")
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionTree':
        """
        Build (train) the regression tree from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples (features).
        y : np.ndarray of shape (n_samples,)
            The target values (continuous values).

        Returns
        -------
        self : RegressionTree
            Returns the instance of the fitted model.

        Raises
        ------
        InputShapeError
            If the number of samples in X and y do not match.
            
        Examples
        --------
        >>> X_train = np.array([[10], [12], [20]])
        >>> y_train = np.array([5, 6, 10])
        >>> model = RegressionTree()
        >>> model.fit(X_train, y_train)
        <ml.supervised_learning.Regression_trees.RegressionTree object at ...>
        """
        X_clean = _ensure_numeric_array(X, name="X", ndim=2)
        X_clean = _ensure_no_nan(X_clean, name="X")
        y_clean = _ensure_numeric_array(y, name="y", ndim=1)
        y_clean = _ensure_no_nan(y_clean, name="y")
        
        if X_clean.shape[0] != y_clean.shape[0]:
            raise InputShapeError(f"X has {X_clean.shape[0]} samples but y has {y_clean.shape[0]}.")

        self.root = self._grow_tree(X_clean, y_clean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            The predicted continuous target values.

        Raises
        ------
        RuntimeError
            If the tree has not been fitted yet.
            
        Examples
        --------
        >>> X_train = np.array([[10], [12], [20]])
        >>> y_train = np.array([5, 6, 10])
        >>> model = RegressionTree(max_depth=1)
        >>> model.fit(X_train, y_train)
        <ml.supervised_learning.Regression_trees.RegressionTree object at ...>
        >>> # Prediction is mean(5, 6, 10) = 10.0 (due to max_depth=1)
        >>> model.predict(np.array([[15]]))
        array([10.])
        """
        if self.root is None:
            raise RuntimeError("Tree not fitted.")
            
        X_clean = _ensure_numeric_array(X, name="X", ndim=2)
        X_clean = _ensure_no_nan(X_clean, name="X")
        
        return np.array([self._traverse_tree(x, self.root) for x in X_clean])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively builds the decision tree graph by finding the optimal split
        at the current node.

        Parameters
        ----------
        X : np.ndarray
            Input features for the current node.
        y : np.ndarray
            Target values for the current node.
        depth : int
            The current depth of the recursion.

        Returns
        -------
        Node
            The root node of the constructed subtree (either a split node or a leaf).
        """
        n_samples, n_features = X.shape
        current_impurity = self.criterion(y)

        if (depth >= self.max_depth or 
            current_impurity < 1e-10 or 
            n_samples < self.min_samples_split):
            
            leaf_value = self.leaf_aggregator(y) if len(y) > 0 else 0.0
            return Node(value=leaf_value)

        feat_idx, threshold = self._best_split(X, y, n_features, current_impurity)

        if feat_idx is None:
            leaf_value = self.leaf_aggregator(y) if len(y) > 0 else 0.0
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)
        
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=feat_idx, threshold=threshold, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int, parent_impurity: float) -> Tuple[Optional[int], Optional[float]]:
        """
        Determines the feature index and threshold value that yields the maximum
        reduction in impurity (information gain).

        Parameters
        ----------
        X : np.ndarray
            Input features for the current node.
        y : np.ndarray
            Target values for the current node.
        n_features : int
            The number of features to iterate through.
        parent_impurity : float
            The impurity score of the current node (parent).

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The index of the best feature and the corresponding best threshold. 
            Returns (None, None) if no split yields positive gain.
            
        Examples
        --------
        # If parent_impurity (variance) = 0.25 (from y=[1, 1, 2, 2]), 
        # a split yielding children [1, 1] (imp=0) and [2, 2] (imp=0) 
        # would maximize gain: Gain = 0.25 - (1/2 * 0 + 1/2 * 0) = 0.25
        """
        best_gain = -float('inf')
        split_idx, split_thresh = None, None
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._calculate_gain(y, X_column, threshold, parent_impurity)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        if best_gain > 0:
            return split_idx, split_thresh
        return None, None

    def _calculate_gain(self, y: np.ndarray, X_column: np.ndarray, 
                        threshold: float, parent_impurity: float) -> float:
        r"""
        Calculates the Reduction in Impurity (Information Gain) for a potential split.

        Equation: $Gain = Impurity(Parent) - \sum \frac{N_{child}}{N_{parent}} \times Impurity(Child)$

        Parameters
        ----------
        y : np.ndarray
            Target values for the current node.
        X_column : np.ndarray
            Feature values for the potential split column.
        threshold : float
            The value defining the split ($X_{column} \le threshold$ goes left).
        parent_impurity : float
            The impurity score of the parent node (calculated using `self.criterion`).

        Returns
        -------
        float
            The calculated information gain (reduction in impurity).
        """
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        imp_l = self.criterion(y[left_idxs])
        imp_r = self.criterion(y[right_idxs])
        
        weighted_child_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r
        return parent_impurity - weighted_child_impurity