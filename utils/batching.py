import numpy as np


def batch_generator(X, batch_size):
    """
    Simple batching

    Args:
        X: numpy array
        batch_size: int

    Yields:
        batches of X
    """
    n = len(X)
    for i in range(0, n, batch_size):
        yield X[i:i + batch_size]


def window_batch_generator(X, window_size, batch_size, stride=1):
    """
    Efficient generator:
    Creates windows AND batches without storing everything

    Useful for large datasets (Intel, SWaT)

    Yields:
        batch of windows (B, W, F)
    """
    batch = []

    for i in range(0, len(X) - window_size + 1, stride):
        window = X[i:i + window_size]
        batch.append(window)

        if len(batch) == batch_size:
            yield np.array(batch)
            batch = []

    if len(batch) > 0:
        yield np.array(batch)