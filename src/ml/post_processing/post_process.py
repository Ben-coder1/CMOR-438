import numpy as np
from collections import Counter

def majorityLabel(labels):
    """
    Return the most frequent label from a sequence or NumPy array.

    Parameters
    ----------
    labels : array_like
        A sequence or NumPy array of hashable labels (e.g., strings, ints, floats).

    Returns
    -------
    label
        The label that appears most frequently. In case of a tie, the label
        that appears first in the input is returned.

    Raises
    ------
    ValueError
        If the input is None or empty.
    TypeError
        If labels are not hashable.

    Examples
    --------
    >>> majorityLabel([1, 2, 2, 3])
    2
    >>> majorityLabel(np.array(['cat', 'dog', 'cat']))
    'cat'
    >>> majorityLabel(['x', 'y', 'x', 'y'])  # tie -> first occurrence
    'x'
    """
    if labels is None:
        raise ValueError("Labels must not be None.")

    arr = np.asarray(labels)
    if arr.size == 0:
        raise ValueError("Label list/array must not be empty.")

    counts = Counter(arr.tolist())
    max_count = max(counts.values())

    for label in arr:
        if counts[label] == max_count:
            # Convert NumPy scalars to native Python types
            if isinstance(label, np.generic):
                return label.item()
            return label


def averageLabel(labels):
    """
    Compute the arithmetic mean of numeric labels.

    Parameters
    ----------
    labels : array_like
        A sequence or NumPy array of numeric values (int or float).

    Returns
    -------
    float
        The arithmetic mean of the labels.

    Raises
    ------
    ValueError
        If the input is None or empty.
    TypeError
        If any element is not numeric.

    Examples
    --------
    >>> averageLabel([1, 2, 3, 4])
    2.5
    >>> averageLabel(np.array([10.0, 20.0]))
    15.0
    >>> averageLabel([1, 2.5, 3])
    2.1666666666666665
    """
    if labels is None:
        raise ValueError("Labels must not be None.")

    arr = np.asarray(labels)
    if arr.size == 0:
        raise ValueError("Label list/array must not be empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("All labels must be numeric.")

    return float(np.mean(arr))
