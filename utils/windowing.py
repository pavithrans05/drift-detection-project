import numpy as np


def create_windows(X, window_size, stride=1):
    """
    Convert time-series into overlapping windows

    Args:
        X: numpy array (N, F)
        window_size: int
        stride: int

    Returns:
        windows: (M, window_size, F)
    """
    windows = []

    for i in range(0, len(X) - window_size + 1, stride):
        windows.append(X[i:i + window_size])

    return np.array(windows)


def flatten_windows(windows):
    """
    Flatten windows for model input

    Args:
        windows: (M, W, F)

    Returns:
        flat_windows: (M, W * F)
    """
    M, W, F = windows.shape
    return windows.reshape(M, W * F)


def create_windows_generator(X, window_size, stride=1):
    """
    Generator version (memory efficient for large datasets)

    Yields:
        window (W, F)
    """
    for i in range(0, len(X) - window_size + 1, stride):
        yield X[i:i + window_size]