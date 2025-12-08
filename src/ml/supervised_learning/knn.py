import numbers
import numpy as np
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance
from ml.post_processing.label_post_processing import majorityLabel, averageLabel
from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan, _ensure_callable, _ensure_positive_int, _ensure_same_length

class KNN:
    """
    k-Nearest Neighbors (KNN) predictor.

    This implementation requires **homogeneous numeric data** for both
    feature vectors (`X`) and labels (`y`). All inputs are converted to
    NumPy arrays with numeric dtype. Mixed-type or ragged inputs (e.g.,
    strings, tuples, lists of unequal length) are not supported.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features), optional
        Training feature vectors. Must be numeric and homogeneous.
    y : array_like of shape (n_samples,), optional
        Training labels. Must be numeric or categorical values that can
        be stored in a NumPy array.

    Attributes
    ----------
    X : np.ndarray
        Stored training feature vectors.
    y : np.ndarray
        Stored training labels.

    Examples
    --------
    >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
    >>> y_train = np.array([0, 1, 1])
    >>> knn = KNN(X_train, y_train)
    >>> knn.predict([0.9, 0.9], classify=True, K=2)
    1
    """

    def __init__(self, X=None, y=None):
        # Validate and store training data if provided
        self.X = _ensure_numeric_array(X, name="X", ndim=2) if X is not None else None
        self.y = np.asarray(y) if y is not None else None

    def find_neighbors(self, target, X, y=None, K=5, dist=None):
        """
        Find the K nearest neighbors to a target vector.

        Parameters
        ----------
        target : array_like of shape (n_features,)
            Target vector.
        X : array_like of shape (n_samples, n_features)
            Training feature vectors.
        y : array_like of shape (n_samples,), optional
            Training labels.
        K : int, default=5
            Number of neighbors to retrieve.
        dist : callable, optional
            Distance function. Defaults to Euclidean distance.

        Returns
        -------
        list of tuples
            Each tuple is (neighbor_vector, neighbor_label, distance).

        Raises
        ------
        ValueError
            If target is None, K is too large, or X and y lengths mismatch.
        TypeError
            If K is not a positive integer, dist is not callable,
            or dist returns non-numeric values.

        Examples
        --------
        >>> X = np.array([[0,0],[1,1],[2,2]])
        >>> y = np.array([0,1,1])
        >>> knn = KNN(X, y)
        >>> neighbors = knn.find_neighbors([0.9,0.9], X, y, K=2)
        >>> len(neighbors)
        2
        """
        if target is None:
            raise ValueError("Target vector must be provided.")

        K = _ensure_positive_int(K, "K")

        if dist is None:
            dist = EuclideanDistance
        else:
            _ensure_callable(dist, "dist")

        # Validate inputs
        target = _ensure_numeric_array(target, name="target", ndim=1)
        X = _ensure_numeric_array(X, name="X", ndim=2)
        _ensure_no_nan(X, "X")

        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError("X and y must be the same length.")
        if K > len(X):
            raise ValueError(f"K={K} is too large for dataset of size {len(X)}.")

        # Compute distances to all training samples
        dists = []
        for x_i in X:
            d = dist(target, x_i)
            if not isinstance(d, numbers.Real):
                raise TypeError(f"dist must return a numeric scalar, got {type(d).__name__}.")
            dists.append(d)
        dists = np.array(dists)

        # Get indices of K nearest neighbors
        idx = np.argsort(dists)[:K]
        labels = y[idx] if y is not None else [None] * K

        return [(X[i], labels[j], dists[i]) for j, i in enumerate(idx)]

    def predict(self, target, classify=True, K=5, X=None, y=None, dist=None):
        """
        Predict a label for the target vector.

        Parameters
        ----------
        target : array_like of shape (n_features,)
            Target vector.
        classify : bool, default=True
            If True, perform classification (majority vote).
            If False, perform regression (average).
        K : int, default=5
            Number of neighbors to use.
        X : array_like, optional
            Training features (if not provided at init).
        y : array_like, optional
            Training labels (if not provided at init).
        dist : callable, optional
            Distance function. Defaults to Euclidean distance.

        Returns
        -------
        label
            Predicted label (categorical or numeric).

        Raises
        ------
        ValueError
            If training data is missing.
        TypeError
            If classify is not boolean or regression labels are non-numeric.

        Examples
        --------
        >>> X = np.array([[0,0],[1,1],[2,2]])
        >>> y = np.array([0,1,1])
        >>> knn = KNN(X, y)
        >>> knn.predict([0.9,0.9], classify=True, K=2)
        1
        """
        if not isinstance(classify, bool):
            raise TypeError("classify must be a boolean.")

        K = _ensure_positive_int(K, "K")

        if dist is None:
            dist = EuclideanDistance
        else:
            _ensure_callable(dist, "dist")

        target = _ensure_numeric_array(target, name="target", ndim=1)
        X = _ensure_numeric_array(X, name="X", ndim=2) if X is not None else self.X
        y = np.asarray(y) if y is not None else self.y

        if X is None or y is None:
            raise ValueError("X and y must be provided either in init or in predict.")
        if not classify and not np.issubdtype(y.dtype, np.number):
            raise TypeError("All labels must be numeric for regression.")

        neighbors = self.find_neighbors(target, X, y, K, dist)
        labels = [label for _, label, _ in neighbors]

        return majorityLabel(labels) if classify else averageLabel(labels)



    def error(self, X_test, y_test, K=5, dist=None, classify=True, X_train=None, y_train=None) -> float:

        """
        Parameters
        ----------
        X_test : array_like of shape (n_samples, n_features)
            Test feature vectors.
        y_test : array_like of shape (n_samples,)
            True labels or target values for test samples.
        K : int, default=1
            Number of neighbors to use.
        classify : bool, default=True
            If True, compute classification error (misclassification rate).
            If False, compute regression error (mean absolute error).
        X_train : array_like of shape (n_train, n_features), optional
            Training feature vectors. If None, uses the training data stored in
            the KNN instance.
        y_train : array_like of shape (n_train,), optional
            Training labels or target values. If None, uses the labels stored in
            the KNN instance.
        dist : callable, optional
            Distance function. If None, defaults to EuclideanDistance.

        Returns
        -------
        float
            Error value:
            - Misclassification rate for classification.
            - Mean absolute error for regression.

        Raises
        ------
        TypeError
            If `X_train` is provided but not array-like, or dist is not callable,
            or dist returns non-numeric values.
        ValueError
            If training data is missing, or if `X_test` and `y_test` lengths differ.

        Examples
        --------
        >>> X_train = np.array([[0,0],[1,1],[2,2]])
        >>> y_train = np.array([0,1,1])
        >>> knn = KNN(X_train, y_train)
        >>> X_test = np.array([[0.9,0.9],[2.1,2.1]])
        >>> y_test = np.array([1,1])
        >>> round(knn.error(X_test, y_test, K=2, classify=True), 2)
        0.0

        Regression error example:
        >>> y_train_reg = np.array([0.0, 1.0, 2.0])
        >>> knn_reg = KNN(X_train, y_train_reg)
        >>> round(knn_reg.error(X_test, np.array([1.0, 2.0]), K=2, classify=False), 2)
        0.5
        """
        # Validate K
        K = _ensure_positive_int(K, "K")

        # Validate training input type if explicitly provided
        if X_train is not None and not isinstance(X_train, (list, tuple, np.ndarray)):
            raise TypeError("X_train must be array-like (list, tuple, or np.ndarray)")

        # Use provided training data or fall back to stored attributes
        X_train = _ensure_numeric_array(X_train, name="X_train", ndim=2) if X_train is not None else self.X
        y_train = np.asarray(y_train) if y_train is not None else self.y

        # Validate test data
        X_test = _ensure_numeric_array(X_test, name="X_test", ndim=2)
        y_test = np.asarray(y_test)

        _ensure_no_nan(X_train, "X_train")
        _ensure_no_nan(X_test, "X_test")
        _ensure_no_nan(y_train, "y_train")

        if X_train is None or y_train is None:
            raise ValueError("Training data must be provided.")

        _ensure_same_length(X_test, "X_test", y_test, "y_test")
        _ensure_same_length(X_train, "X_train", y_train, "y_train")

        if dist is None:
            dist = EuclideanDistance
        else:
            _ensure_callable(dist, "dist")

        # Predict for each test sample
        preds = []
        for x in X_test:
            pred = self.predict(x, classify=classify, K=K, X=X_train, y=y_train, dist=dist)
            preds.append(pred)

        # Compute error
        if classify:
            # Misclassification rate
            errors = np.mean(np.array(preds) != y_test)
        else:
            # Mean absolute error for regression
            errors = np.mean(np.abs(np.array(preds, dtype=float) - y_test))

        return float(errors)

