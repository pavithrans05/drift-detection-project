import numpy as np


def detect_drift(scores, percentile=95):
    """
    Convert KS scores to binary drift decisions
    """
    threshold = np.percentile(scores, percentile)
    drift_points = np.where(scores > threshold)[0]

    return drift_points, threshold