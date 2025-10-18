from itertools import chain

def LnDistanceCostructor(p: float):
    """
    Creates a function to compute the L-n distance between two vectors.

    Parameters:
        p (float): The order of the norm (p >= 1).

    Returns:
        function: A function that takes two vectors and returns their L-n distance.

    Raises:
        ValueError: If p < 1, or if input vectors are empty, None, or of unequal length.
        TypeError: If any vector element is not numeric.
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1")

    def ln_distance(vec1, vec2):
        # Check for None
        if vec1 is None or vec2 is None:
            raise ValueError("Input vectors must not be None")

        # Check for emptiness
        if not vec1 or not vec2:
            raise ValueError("Input vectors must be non-empty")

        # Check for length mismatch
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must be of the same length")

        # Check for numeric types
       

        if not all(isinstance(x, (int, float)) for x in chain(vec1, vec2)):
            raise TypeError("Both vectors must contain only numeric values.")


        # Compute L-n distance
        return sum(abs(a - b) ** p for a, b in zip(vec1, vec2)) ** (1 / p)

    return ln_distance




EuclideanDistance = LnDistanceCostructor(2)
"""
Computes the Euclidean distance between two vectors.

This function is a specialization of the L-n norm with p = 2, corresponding to the standard
Euclidean metric in ℝⁿ. It returns the square root of the sum of squared differences between
corresponding elements of the input vectors.

Returns:
    float: The Euclidean distance between two vectors of equal length.

Raises:
    ValueError: If the input vectors are not of the same length.
"""
def LinfinityDistance(vec1, vec2):
    """
    Computes the L-infinity distance (maximum norm) between two vectors.

    The L-infinity distance is defined as the maximum absolute difference between
    corresponding elements of two input vectors. It is the limiting case of the Lⁿ norm
    as p → ∞, and is commonly used in uniform convergence and sup-norm analysis.

    Parameters:
        vec1 (list or array-like): First input vector.
        vec2 (list or array-like): Second input vector.

    Returns:
        float: The L-infinity distance between vec1 and vec2.

    Raises:
        ValueError: If either vector is None, empty, or if their lengths do not match.
        TypeError: If any element in either vector is not numeric.
    """
    if vec1 is None or vec2 is None:
        raise ValueError("Input vectors must not be None")

    if not vec1 or not vec2:
        raise ValueError("Input vectors must be non-empty")

    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    if not all(isinstance(x, (int, float)) for x in chain(vec1, vec2)):
            raise TypeError("Both vectors must contain only numeric values.")

    return max(abs(a - b) for a, b in zip(vec1, vec2))

#this is for some other test I made
def ascii_word_dist(str1: str, str2: str) -> int:
    """
    Computes the distance between two strings by summing the absolute differences
    in ASCII values at each character position. If one string is shorter, missing
    characters are treated as having ASCII value 0.

    Parameters:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        int: Total ASCII difference across all positions.

    Raises:
        TypeError: If either input is not a string.
    """
    if not isinstance(str1, str) or not isinstance(str2, str):
        raise TypeError("Both inputs must be strings.")

    max_len = max(len(str1), len(str2))
    total = 0
    for i in range(max_len):
        c1 = ord(str1[i]) if i < len(str1) else 0
        c2 = ord(str2[i]) if i < len(str2) else 0
        total += abs(c1 - c2)

    return total

