import numpy as np

def LnDistanceConstructor(p: float):
    """
    Create an L-p distance function.

    Constructs a function that computes the L-p (Minkowski) distance between
    two one-dimensional numeric vectors using NumPy.

    Parameters
    ----------
    p : float
        The order of the norm. Must satisfy ``p >= 1``.

    Returns
    -------
    function
        A function ``f(vec1, vec2)`` that computes the L-p distance between
        two vectors.

    Raises
    ------
    ValueError
        If `p` is less than 1.
    ValueError
        If either input vector is ``None``, empty, not 1-D, or of unequal length.
    TypeError
        If either input contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> L3 = LnDistanceConstructor(3)
    >>> L3([1, 2, 3], [4, 5, 6])
    4.3267487109222245
    >>> EuclideanDistance([0, 0], [3, 4])
    5.0
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1")

    def ln_distance(vec1, vec2):
        if vec1 is None or vec2 is None:
            raise ValueError("Input vectors must not be None")

        a = np.asarray(vec1)
        b = np.asarray(vec2)

        if a.size == 0 or b.size == 0:
            raise ValueError("Input vectors must be non-empty")
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Only 1-D vectors are supported")
        if a.shape != b.shape:
            raise ValueError("Vectors must be of the same length")
        if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
            raise TypeError("Both vectors must contain only numeric values")

        diff = np.abs(a - b)
        return float(np.linalg.norm(diff, ord=p))

    return ln_distance


# Euclidean distance (p = 2)
EuclideanDistance = LnDistanceConstructor(2)


def LinfinityDistance(vec1, vec2):
    """
    Compute the L-infinity (Chebyshev) distance.

    The L-infinity distance is the maximum absolute difference between
    corresponding elements of two vectors.

    Parameters
    ----------
    vec1, vec2 : array_like
        One-dimensional numeric vectors of equal length.

    Returns
    -------
    float
        The L-infinity distance between `vec1` and `vec2`.

    Raises
    ------
    ValueError
        If either vector is ``None``, empty, not 1-D, or of unequal length.
    TypeError
        If either vector contains non-numeric values.

    Examples
    --------
    >>> LinfinityDistance([1, 2, 3], [2, 4, 0])
    3.0
    >>> LinfinityDistance(np.array([5, -1]), np.array([2, 3]))
    4.0
    """
    if vec1 is None or vec2 is None:
        raise ValueError("Input vectors must not be None")

    a = np.asarray(vec1)
    b = np.asarray(vec2)

    if a.size == 0 or b.size == 0:
        raise ValueError("Input vectors must be non-empty")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Only 1-D vectors are supported")
    if a.shape != b.shape:
        raise ValueError("Vectors must be of the same length")
    if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        raise TypeError("Both vectors must contain only numeric values")

    return float(np.max(np.abs(a - b)))


def taxicab_distance(vec1, vec2):
    """
    Compute the Taxicab (Manhattan, L1) distance.

    The Taxicab distance is the sum of absolute differences between
    corresponding elements of two vectors.

    Parameters
    ----------
    vec1, vec2 : array_like
        One-dimensional numeric vectors of equal length.

    Returns
    -------
    float
        The L1 distance between `vec1` and `vec2`.

    Raises
    ------
    ValueError
        If either vector is ``None``, empty, not 1-D, or of unequal length.
    TypeError
        If either vector contains non-numeric values.

    Examples
    --------
    >>> taxicab_distance([1, 2, 3], [4, 0, -1])
    9.0
    >>> taxicab_distance(np.array([0, 0]), np.array([3, 4]))
    7.0
    """
    if vec1 is None or vec2 is None:
        raise ValueError("Input vectors must not be None")

    a = np.asarray(vec1)
    b = np.asarray(vec2)

    if a.size == 0 or b.size == 0:
        raise ValueError("Input vectors must be non-empty")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Only 1-D vectors are supported")
    if a.shape != b.shape:
        raise ValueError("Vectors must be of the same length")
    if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        raise TypeError("Both vectors must contain only numeric values")

    return float(np.sum(np.abs(a - b)))


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
        The ASCII distance between the two strings.

    Raises
    ------
    TypeError
        If either input is not a string.

    Examples
    --------
    >>> ascii_word_dist("abc", "abd")
    1
    >>> ascii_word_dist("cat", "dog")
    28
    >>> ascii_word_dist("hi", "hello")
    331
    """
    if not isinstance(str1, str) or not isinstance(str2, str):
        raise TypeError("Both inputs must be strings.")

    max_len = max(len(str1), len(str2))
    c1 = np.fromiter((ord(str1[i]) if i < len(str1) else 0 for i in range(max_len)), dtype=int)
    c2 = np.fromiter((ord(str2[i]) if i < len(str2) else 0 for i in range(max_len)), dtype=int)
    return int(np.sum(np.abs(c1 - c2)))
