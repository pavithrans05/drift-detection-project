from scipy.stats import ks_2samp
import numpy as np


def compute_ks_stat(window1, window2):
    """
    Compute KS statistic across features
    """
    stats = []

    for i in range(window1.shape[1]):
        stat, _ = ks_2samp(window1[:, i], window2[:, i])
        stats.append(stat)

    return np.mean(stats)


def compute_ks_scores(windows, ref_size=50):
    """
    Reference-based KS drift detection
    """
    ref_window = np.vstack(windows[:ref_size])

    scores = []

    for i in range(ref_size, len(windows)):
        curr_window = windows[i]
        score = compute_ks_stat(ref_window, curr_window)
        scores.append(score)

    return np.array(scores)