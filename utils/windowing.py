import numpy as np


def create_sliding_windows(data, window_size=50, step=1):
    """
    Convert time stream into sliding windows

    Args:
        data: (N, d)
        window_size: L
        step: stride

    Returns:
        windows: (num_windows, L, d)
    """
    num_samples = len(data)
    windows = []

    for i in range(0, num_samples - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)

    return np.array(windows)