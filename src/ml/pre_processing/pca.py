import warnings
import numpy as np

from ml.utils._errors_and_warnings._general_error_handling import _ensure_numeric_array, _ensure_no_nan, _ensure_positive_int
def compute_pca(data: np.ndarray, n_components: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) on a 2D numeric dataset.

    Parameters
    ----------
    data : array_like of shape (n_samples, n_features)
        Input data matrix. Rows are samples, columns are features.
    n_components : int, optional
        Number of principal components to return. If None, all components are returned.

    Returns
    -------
    tuple
        (components, explained_variance, transformed_data)
        - components : np.ndarray of shape (n_components, n_features)
            Principal component vectors (unit length).
        - explained_variance : np.ndarray of shape (n_components,)
            Variance explained by each component.
        - transformed_data : np.ndarray of shape (n_samples, n_components)
            Data projected onto the principal components.

    Raises
    ------
    TypeError
        If input is not numeric.
    ValueError
        If input is empty, not 2D, or n_components is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> comps, var, X_proj = compute_pca(X, n_components=1)
    >>> comps.shape
    (1, 2)
    >>> var
    array([8.])
    >>> X_proj.shape
    (3, 1)
    """
    # Ensure numeric 2D input
    arr = _ensure_numeric_array(data, name="data", ndim=2)
    _ensure_no_nan(arr, "data")

    n_samples, n_features = arr.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("Input data must not be empty.")

    # Validate n_components
    if n_components is None:
        n_components = n_features
    else:
        n_components = _ensure_positive_int(n_components, "n_components")
        if n_components > n_features:
            raise ValueError(f"n_components ({n_components}) cannot exceed number of features ({n_features}).")

    # Center the data (mean = 0 per feature)
    centered = arr - np.mean(arr, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(centered, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues/vectors in descending order
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    # Select top n_components
    components = eigvecs[:, :n_components].T
    explained_variance = eigvals[:n_components]

    # Project data
    transformed_data = centered @ components.T

    return components, explained_variance, transformed_data
