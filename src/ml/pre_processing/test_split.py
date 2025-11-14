import numpy as np


def train_test_split(data, train_ratio=0.7, seed=None):
    """
    Split a dataset into training and testing sets.

    Shuffles the data and splits it into training and testing subsets
    according to the specified ratio.

    Parameters
    ----------
    data : sequence
        Input dataset (list, tuple, or array).
    train_ratio : float, optional
        Proportion of data to include in the training set (default 0.7).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of lists
        (train, test) subsets.

    Raises
    ------
    ValueError
        If input is empty or train_ratio is not between 0 and 1.

    Examples
    --------
    >>> train, test = train_test_split([1, 2, 3, 4, 5], train_ratio=0.6, seed=42)
    >>> len(train), len(test)
    (3, 2)
    """
    data = list(data)  # Ensure sequence type
    if len(data) == 0:
        raise ValueError("Input must not be empty.")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    # Shuffle indices reproducibly
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)

    # Split into train/test
    split_index = int(len(data) * train_ratio)
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]

    return train, test