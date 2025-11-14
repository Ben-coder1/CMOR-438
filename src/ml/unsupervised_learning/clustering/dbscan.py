import numpy as np
import numbers
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array,
    _ensure_no_nan,
    _ensure_positive_int,
    _ensure_callable,
    _ensure_numeric_scalar,
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
    distance_func : callable, optional
        Distance function. Defaults to EuclideanDistance.

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

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]])
    >>> labels = dbscan(X, eps=1.5, min_samples=2)
    >>> set(labels) <= {0, 1, -1}
    True
    """
    # --- Input validation ---
    X = _ensure_numeric_array(X, name="X", ndim=2)   # Ensure numeric 2D array
    _ensure_no_nan(X, "X")                           # No NaN values allowed

    # Validate eps (radius) and min_samples
    if not isinstance(eps, numbers.Real) or eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    min_samples = _ensure_positive_int(min_samples, "min_samples")

    # Validate distance function
    _ensure_callable(distance_func, "distance_func")

    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=int)       # Initialize all points as noise (-1)
    cluster_id = 0                                   # Cluster counter
    visited = np.zeros(n_samples, dtype=bool)        # Track visited points

    def region_query(point_idx):
        """Find all neighbors within eps of point_idx."""
        neighbors = []
        for j in range(n_samples):
            d = distance_func(X[point_idx], X[j])    # Compute distance
            _ensure_numeric_scalar(d, "distance_func")  # Ensure numeric scalar
            if d <= eps:
                neighbors.append(j)
        return neighbors

    # --- Main loop over all points ---
    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Not enough neighbors → noise
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
                        # Add new neighbors to seeds if not already present
                        seeds.extend([n for n in j_neighbors if n not in seeds])
                # Assign cluster label if point was noise
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels
