def normalize_by_max_abs(data):
    """
    Normalizes a list of numeric values by the absolute value of the maximum.

    Parameters:
        data (list): A non-empty list of numeric values.

    Returns:
        list: Normalized values where each element is divided by abs(max(data)).

    Raises:
        ValueError: If data is empty or max absolute value is zero.
        TypeError: If data is not a list or contains non-numeric elements.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list.")
    if not data:
        raise ValueError("Input list must not be empty.")
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All elements must be numeric.")

    max_val = max(abs(x) for x in data)
    if max_val == 0:
        raise ValueError("Maximum absolute value must not be zero.")

    return [x / max_val for x in data]


def normalize_by_average_abs(data):
    """
    Normalizes a list of numeric values by the average of their absolute values.

    Parameters:
        data (list): A non-empty list of numeric values.

    Returns:
        list: Normalized values where each element is divided by the average of abs(x_i).

    Raises:
        ValueError: If data is empty or average absolute value is zero.
        TypeError: If data is not a list or contains non-numeric elements.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list.")
    if not data:
        raise ValueError("Input list must not be empty.")
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All elements must be numeric.")

    avg_abs = sum(abs(x) for x in data) / len(data)
    if avg_abs == 0:
        raise ValueError("Average absolute value must not be zero.")

    return [x / avg_abs for x in data]


def normalize_vectors_by_max_abs(vectors):
    """
    Applies normalize_by_max_abs componentwise to a list of numeric vectors.

    Parameters:
        vectors (list of list of numbers): A non-empty list of equal-length numeric vectors.

    Returns:
        list of list: Componentwise normalized vectors.

    Raises:
        ValueError: If input is empty or vectors are not uniform in length.
        TypeError: If input is not a list of numeric lists.
    """
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("Input must be a non-empty list of vectors.")
    if not all(isinstance(vec, list) for vec in vectors):
        raise TypeError("Each element must be a list.")
    if not all(len(vec) == len(vectors[0]) for vec in vectors):
        raise ValueError("All vectors must have the same length.")
    if not all(all(isinstance(x, (int, float)) for x in vec) for vec in vectors):
        raise TypeError("All elements must be numeric.")

    transposed = list(zip(*vectors))
    normalized = [normalize_by_max_abs(list(col)) for col in transposed]
    return [list(row) for row in zip(*normalized)]


def normalize_vectors_by_average_abs(vectors):
    """
    Applies normalize_by_average_abs componentwise to a list of numeric vectors.

    Parameters:
        vectors (list of list of numbers): A non-empty list of equal-length numeric vectors.

    Returns:
        list of list: Componentwise normalized vectors.

    Raises:
        ValueError: If input is empty or vectors are not uniform in length.
        TypeError: If input is not a list of numeric lists.
    """
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("Input must be a non-empty list of vectors.")
    if not all(isinstance(vec, list) for vec in vectors):
        raise TypeError("Each element must be a list.")
    if not all(len(vec) == len(vectors[0]) for vec in vectors):
        raise ValueError("All vectors must have the same length.")
    if not all(all(isinstance(x, (int, float)) for x in vec) for vec in vectors):
        raise TypeError("All elements must be numeric.")

    transposed = list(zip(*vectors))
    normalized = [normalize_by_average_abs(list(col)) for col in transposed]
    return [list(row) for row in zip(*normalized)]


import random

def train_test_split(data, train_ratio=0.7, seed=None):
    """
    Splits a dataset into training and testing sets based on a specified ratio.

    Parameters:
        data (list): A non-empty list of items to split.
        train_ratio (float): Proportion of data to assign to training set (default 0.7).
        seed (int, optional): Random seed for reproducible shuffling.

    Returns:
        tuple: (train_data, test_data)

    Raises:
        TypeError: If data is not a list.
        ValueError: If data is empty or train_ratio is not between 0 and 1.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list.")
    if not data:
        raise ValueError("Input list must not be empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    shuffled = data[:]
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)
    return shuffled[:split_index], shuffled[split_index:]
