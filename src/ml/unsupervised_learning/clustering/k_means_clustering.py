import numpy as np
import numbers
from typing import Optional, Tuple
from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan, _ensure_callable
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance

#do note that if a poor distance is chosen, taking the mean will not give good clusters. Thus using Euclidean distance is recommended.
def kmeans_clustering(
    X,
    k: int,
    epsilon: Optional[float] = None,
    max_iter: Optional[int] = None,
    seed: Optional[int] = None,
    distance_func: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
    >>> centroids, labels = kmeans_clustering(X, k=2, epsilon=1e-4, seed=42)
    >>> centroids.shape
    (2, 2)
    >>> set(labels) <= {0, 1}
    True

    Custom distance function:
    >>> def manhattan(a, b): return float(np.sum(np.abs(a - b)))
    >>> centroids, labels = kmeans_clustering(X, k=2, epsilon=1e-4, seed=42, distance_func=manhattan)
    >>> centroids.shape
    (2, 2)

    Error case: k larger than samples
    >>> kmeans_clustering(np.array([[0.0, 0.0]]), k=2, epsilon=1e-4, seed=1)
    Traceback (most recent call last):
        ...
    ValueError: k=2 cannot exceed number of samples (1).
    """


    # --- Input validation ---
    X = _ensure_numeric_array(X, name="X", ndim=2)
    _ensure_no_nan(X, name="X")

    n_samples, n_features = X.shape

    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k}.")
    if k > n_samples:
        raise ValueError(f"k={k} cannot exceed number of samples ({n_samples}).")

    if epsilon is None and max_iter is None:
        raise ValueError("Must provide either epsilon > 0 or max_iter.")

    if epsilon is not None:
        if not isinstance(epsilon, numbers.Real) or epsilon <= 0:
            raise ValueError(f"if an epsilon is provideded it must be > 0, got {epsilon}.")

    if max_iter is not None:
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"if given, max_iter must be a positive integer, got {max_iter}.")

    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None.")

    # --- Distance function validation ---
    if distance_func is None:
        distance_func = EuclideanDistance
    else:
        _ensure_callable(distance_func, "distance_func")

    rng = np.random.default_rng(seed)

    # --- Initialize centroids ---
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[indices]

    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter if max_iter is not None else np.iinfo(int).max):
        # --- Assign step ---
        for i, x in enumerate(X):
            distances = []
            for c in centroids:
                d = distance_func(x, c)
                if not isinstance(d, numbers.Real):
                    #I leave errors like these because I specifally like noting that It is indeed from the return values.
                    raise TypeError(
                        f"distance_func must return a numeric scalar, got {type(d).__name__}."
                    )
                distances.append(d)
            labels[i] = int(np.argmin(distances))

        # --- Update step ---
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                # Reinitialize empty cluster to a random point
                new_centroids[j] = X[rng.integers(0, n_samples)]
            else:
                new_centroids[j] = members.mean(axis=0)

        # --- Convergence check ---
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if epsilon is not None and shift < epsilon:
            break

    return centroids, labels
