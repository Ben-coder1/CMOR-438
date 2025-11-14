

import numbers
import numpy as np

from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance
from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan, _ensure_callable


def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    distance_func: callable = EuclideanDistance,
) -> float:
    """
    Compute silhouette score for clustering.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Data points.
    labels : array_like of shape (n_samples,)
        Cluster assignments for each sample.
    distance_func : callable, optional
        Distance function. Defaults to EuclideanDistance.

    Returns
    -------
    float
        Silhouette score in [-1, 1].

    Raises
    ------
    ValueError
        If X and labels lengths mismatch, or fewer than 2 clusters.
    TypeError
        If distance_func is not callable or does not return numeric.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> round(compute_silhouette_score(X, labels), 2)
    0.8

    Error case: only one cluster
    >>> labels = np.array([0, 0, 0, 0])
    >>> compute_silhouette_score(X, labels)
    Traceback (most recent call last):
        ...
    ValueError: Silhouette score requires at least 2 clusters.
    """
    X = _ensure_numeric_array(X, name="X", ndim=2)
    labels = _ensure_numeric_array(labels, name="labels", ndim=1)
    _ensure_no_nan(X, "X")

    _ensure_callable(distance_func, "distance_func")

    n_samples = X.shape[0]
    if len(labels) != n_samples:
        raise ValueError("X and labels must have the same length.")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    scores = []
    for i, x in enumerate(X):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == lbl] for lbl in unique_labels if lbl != labels[i]]

        # a(i): mean intra-cluster distance
        a = np.mean([distance_func(x, y) for y in same_cluster if not np.array_equal(x, y)] or [0.0])
        if not isinstance(a, numbers.Real):
            raise TypeError("distance_func must return numeric values.")

        # b(i): min mean distance to other clusters
        b = min(
            np.mean([distance_func(x, y) for y in cluster]) for cluster in other_clusters if len(cluster) > 0
        )
        if not isinstance(b, numbers.Real):
            raise TypeError("distance_func must return numeric values.")

        score = (b - a) / max(a, b)
        scores.append(score)

    return float(np.mean(scores))
