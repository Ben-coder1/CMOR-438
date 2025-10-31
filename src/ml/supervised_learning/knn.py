import numpy as np
from ml.distances.metrics import EuclideanDistance
from ml.post_processing.post_process import majorityLabel, averageLabel

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
        self.X = np.asarray(X, dtype=float) if X is not None else None
        self.y = np.asarray(y) if y is not None else None

    def find_neighbors(self, target, X, y=None, K=5, dist=None):
        """
        Find the K nearest neighbors to a target vector.

        Parameters
        ----------
        target : array_like of shape (n_features,)
            Input vector to compare against.
        X : array_like of shape (n_samples, n_features)
            Feature vectors. Must be numeric and homogeneous.
        y : array_like of shape (n_samples,), optional
            Labels corresponding to `X`.
        K : int, default=5
            Number of neighbors to return.
        dist : callable, optional
            Distance function. Defaults to Euclidean distance.

        Returns
        -------
        list of tuple
            A list of (x_i, label, distance) for the K closest points.
        """
        if target is None:
            raise ValueError("Target vector must be provided.")
        if not isinstance(K, int) or K <= 0:
            raise TypeError("K must be a positive integer.")
        if dist is None:
            dist = EuclideanDistance
        if not callable(dist):
            raise TypeError("dist must be callable.")

        X = np.asarray(X, dtype=float)
        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError("X and y must be the same length.")
        if K > len(X):
            raise ValueError(f"K={K} is too large for dataset of size {len(X)}.")

        dists = np.array([dist(target, x_i) for x_i in X])
        idx = np.argsort(dists)[:K]
        labels = y[idx] if y is not None else [None] * K
        return [(X[i], labels[j], dists[i]) for j, i in enumerate(idx)]

    def predict(self, target, classify=True, K=5, X=None, y=None, dist=None):
        """
        Predict a label for the target vector.

        Parameters
        ----------
        target : array_like of shape (n_features,)
            Input vector to classify or regress.
        classify : bool, default=True
            If True, perform classification (majority label).
            If False, perform regression (average label).
        K : int, default=5
            Number of neighbors to consider.
        X : array_like, optional
            Feature vectors. Defaults to stored training data.
        y : array_like, optional
            Labels. Defaults to stored training labels.
        dist : callable, optional
            Distance function. Defaults to Euclidean distance.

        Returns
        -------
        label
            Predicted label (int, float, or categorical).
        """
        if not isinstance(classify, bool):
            raise TypeError("classify must be a boolean.")
        if dist is None:
            dist = EuclideanDistance

        X = np.asarray(X, dtype=float) if X is not None else self.X
        y = np.asarray(y) if y is not None else self.y
        if X is None or y is None:
            raise ValueError("X and y must be provided either in init or in predict.")
        if not classify and not np.issubdtype(y.dtype, np.number):
            raise TypeError("All labels must be numeric for regression.")

        neighbors = self.find_neighbors(target, X, y, K, dist)
        labels = [label for _, label, _ in neighbors]
        return majorityLabel(labels) if classify else averageLabel(labels)

    def error(self, X_test, y_test, K=5, dist=None, classify=True, X_train=None, y_train=None):
        """
        Compute prediction error on test data.

        Parameters
        ----------
        X_test : array_like of shape (n_samples, n_features)
            Test feature vectors.
        y_test : array_like of shape (n_samples,)
            True labels for test data.
        K : int, default=5
            Number of neighbors to consider.
        dist : callable, optional
            Distance function. Defaults to Euclidean distance.
        classify : bool, default=True
            If True, compute classification error (misclassification rate).
            If False, compute regression error (mean absolute error).
        X_train : array_like, optional
            Training feature vectors. Defaults to stored training data.
        y_train : array_like, optional
            Training labels. Defaults to stored training labels.

        Returns
        -------
        float
            Mean error over the test set.
        """
        if X_train is not None and not isinstance(X_train, (list, tuple, np.ndarray)):
            raise TypeError("X_train must be array-like (list, tuple, or np.ndarray)")

        X_train = np.asarray(X_train, dtype=float) if X_train is not None else self.X
        y_train = np.asarray(y_train) if y_train is not None else self.y
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test)

        if X_train is None or y_train is None:
            raise ValueError("Training data must be provided.")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must be the same length.")
        if dist is None:
            dist = EuclideanDistance

        preds = [self.predict(x, classify=classify, K=K, X=X_train, y=y_train, dist=dist)
                 for x in X_test]

        if classify:
            errors = np.mean(np.array(preds) != y_test)
        else:
            errors = np.mean(np.abs(np.array(preds, dtype=float) - y_test))

        return errors
