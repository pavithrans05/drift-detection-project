import numpy as np


def create_data_stream(batches):
    """
    Convert batch list into continuous stream
    """
    data_stream = np.vstack(batches)
    batch_sizes = [len(b) for b in batches]

    return data_stream, batch_sizes


def get_batch_boundaries(batch_sizes):
    """
    Get start and end indices of each batch
    """
    boundaries = []

    start = 0
    for size in batch_sizes:
        end = start + size
        boundaries.append((start, end))
        start = end

    return boundaries