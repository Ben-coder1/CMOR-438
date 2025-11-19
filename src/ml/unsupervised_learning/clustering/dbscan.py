import numpy as np
import numbers
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_array_like,
    _ensure_ndim,
    _ensure_numeric_array,
    _ensure_no_nan,
    _ensure_positive_int,
    _ensure_callable,
    _ensure_numeric_scalar,
    _ensure_positive_numeric,
)
from ml.metrics_and_evaluations.metrics.metrics import EuclideanDistance

def dbscan(
    X,
    eps: float,
    min_samples: int,
    distance_func: callable = EuclideanDistance,
):
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Input data.
    eps : float
        Neighborhood radius. Must be > 0.
    min_samples : int
        Minimum number of points required to form a dense region. Must be > 0.
    distance_func : callable
        Distance function. Must take two vectors and return a numeric scalar.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster labels for each point. Noise points are labeled -1.

    Raises
    ------
    ValueError
        If eps <= 0 or min_samples <= 0.
    TypeError
        If distance_func is not callable or does not return numeric.
    """

    # --- Input validation ---

    X = _ensure_array_like(X, name="X")
    X = _ensure_ndim(X, name="X", ndim=2)
    X = _ensure_no_nan(X, name="X")

    

    eps = _ensure_positive_numeric(eps, name="eps")
    min_samples = _ensure_positive_int(min_samples, name="min_samples")
    _ensure_callable(distance_func, name="distance_func")

    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)   # Initialize all points as noise
    visited = np.zeros(n_samples, dtype=bool)    # Track visited points
    cluster_id = 0

    def region_query(point_idx):
        """Find all neighbors within eps of point_idx."""
        neighbors = []
        for j in range(n_samples):
            d = distance_func(X[point_idx], X[j])
            if not isinstance(d, numbers.Real):
                raise TypeError("distance_func must return a numeric scalar.")
            if d <= eps:
                neighbors.append(j)
        return neighbors

    # --- Main loop ---
    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1  # noise
        else:
            # Start a new cluster
            labels[i] = cluster_id
            seeds = neighbors.copy()

            # Expand cluster
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = region_query(j)
                    if len(j_neighbors) >= min_samples:
                        # Add unvisited neighbors
                        seeds.extend([n for n in j_neighbors if not visited[n]])
                # Assign cluster label if noise or unassigned
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels
