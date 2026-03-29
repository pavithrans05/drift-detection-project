from river.drift import ADWIN
import numpy as np


def compute_adwin_drift(windows):
    """
    ADWIN drift detection on scalar signal
    """
    adwin = ADWIN()

    scores = []
    drift_flags = []

    for w in windows:
        value = np.mean(w)
        scores.append(value)

        adwin.update(value)
        drift_flags.append(adwin.drift_detected)

    return np.array(scores), drift_flags