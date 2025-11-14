import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array,
    _ensure_no_nan,
    _ensure_same_shape_1d,
    _ensure_string,
)


def LnDistanceConstructor(p: float):
    """
    Create an L-p (Minkowski) distance function.

    Parameters
    ----------
    p : float
        The order of the norm. Must satisfy ``p >= 1``.

    Returns
    -------
    function
        A function ``f(vec1, vec2)`` that computes the L-p distance.

    Raises
    ------
    ValueError
        If `p` is less than 1.

    Examples
    --------
    >>> L3 = LnDistanceConstructor(3)
    >>> L3([1, 2, 3], [4, 5, 6])
    4.3267487109222245
    >>> EuclideanDistance([0, 0], [3, 4])
    5.0
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1")

    def ln_distance(vec1, vec2):
        # Ensure both inputs are numeric 1-D arrays
        a = _ensure_numeric_array(vec1, name="vec1", ndim=1)
        b = _ensure_numeric_array(vec2, name="vec2", ndim=1)

        # Ensure no NaN values
        _ensure_no_nan(a, name="vec1")
        _ensure_no_nan(b, name="vec2")

        # Ensure same shape
        _ensure_same_shape_1d(a,"vec1", b, "vec2")

        # Compute Minkowski distance
        diff = np.abs(a - b)
        return float(np.linalg.norm(diff, ord=p))

    return ln_distance


# Euclidean distance (p = 2)
EuclideanDistance = LnDistanceConstructor(2)


def LinfinityDistance(vec1, vec2):
    """
    Compute the L-infinity (Chebyshev) distance.

    Parameters
    ----------
    vec1, vec2 : array_like
        One-dimensional numeric vectors of equal length.

    Returns
    -------
    float
        The L-infinity distance.

    Examples
    --------
    >>> LinfinityDistance([1, 2, 3], [2, 4, 0])
    3.0
    >>> LinfinityDistance(np.array([5, -1]), np.array([2, 3]))
    4.0
    """
    a = _ensure_numeric_array(vec1, name="vec1", ndim=1)
    b = _ensure_numeric_array(vec2, name="vec2", ndim=1)

    _ensure_no_nan(a, name="vec1")
    _ensure_no_nan(b, name="vec2")
    _ensure_same_shape_1d(a,"vec1", b, "vec2")

    return float(np.max(np.abs(a - b)))


taxicab_distance = LnDistanceConstructor(1)


def ascii_word_dist(str1: str, str2: str) -> int:
    """
    Compute ASCII-based string distance.

    The distance is defined as the sum of absolute differences in ASCII
    values at each character position. If one string is shorter, missing
    characters are treated as ASCII 0.

    Parameters
    ----------
    str1, str2 : str
        Input strings.

    Returns
    -------
    int
        The ASCII distance.

    Examples
    --------
    >>> ascii_word_dist("abc", "abd")
    1
    >>> ascii_word_dist("cat", "dog")
    28
    >>> ascii_word_dist("hi", "hello")
    331
    """
    # Ensure both inputs are strings
    str1 = _ensure_string(str1, "str1")
    str2 = _ensure_string(str2, "str2")

    # Pad shorter string with ASCII 0
    max_len = max(len(str1), len(str2))
    c1 = np.fromiter((ord(str1[i]) if i < len(str1) else 0 for i in range(max_len)), dtype=int)
    c2 = np.fromiter((ord(str2[i]) if i < len(str2) else 0 for i in range(max_len)), dtype=int)

    return int(np.sum(np.abs(c1 - c2)))


