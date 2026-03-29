import numpy as np
from river.drift import ADWIN


def adwin_drift(X):
    """
    Detect drift using ADWIN (River library)

    Uses L2 norm as signal

    Args:
        X: (N, F)

    Returns:
        drift_points: list
    """
    adwin = ADWIN()
    drift_points = []

    for i in range(len(X)):
        # Better 1D signal
        value = np.linalg.norm(X[i])

        adwin.update(value)

        # ✅ FIXED ATTRIBUTE
        if adwin.drift_detected:
            drift_points.append(i)

    return drift_points