from collections import Counter

def majorityLabel(labels):
    """
    Returns the most frequent label from a list.

    Parameters:
        labels (list): A list of hashable labels (e.g., strings, integers).

    Returns:
        The label that appears most frequently.

    Raises:
        ValueError: If the input is None or empty.
        TypeError: If labels are not hashable or not in a list.
    """
    if labels is None or not isinstance(labels, list):
        raise TypeError("Input must be a list of labels.")
    if not labels:
        raise ValueError("Label list must not be empty.")

    counts = Counter(labels)
    max_count = max(counts.values())
    # Return the first label with max count (deterministic tie-breaking)
    for label in labels:
        if counts[label] == max_count:
            return label
        

def averageLabel(labels):
    """
    Computes the average (mean) of a list of numeric labels.

    Parameters:
        labels (list): A list of numeric values (int or float).

    Returns:
        float: The arithmetic mean of the labels.

    Raises:
        ValueError: If the input is None or empty.
        TypeError: If any element is not numeric.
    """
    if labels is None or not isinstance(labels, list):
        raise TypeError("Input must be a list of numeric labels.")
    if not labels:
        raise ValueError("Label list must not be empty.")
    if not all(isinstance(x, (int, float)) for x in labels):
        raise TypeError("All labels must be numeric.")

    return sum(labels) / len(labels)

