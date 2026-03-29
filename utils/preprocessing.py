import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_batches(batches):
    """
    Remove NaN and infinite values
    """
    cleaned = []

    for b in batches:
        b = b[~np.isnan(b).any(axis=1)]
        b = b[~np.isinf(b).any(axis=1)]
        cleaned.append(b)

    return cleaned


def normalize_batches(batches):
    """
    Normalize across ALL batches
    """
    scaler = StandardScaler()

    combined = np.vstack(batches)
    scaler.fit(combined)

    normalized = [scaler.transform(b) for b in batches]

    return normalized, scaler