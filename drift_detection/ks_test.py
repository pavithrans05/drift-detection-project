import numpy as np
from scipy.stats import ks_2samp


def ks_drift_series(windows, ref_size=50, sample_size=5000):
    """
    Fast KS drift detection using sampling

    Args:
        windows: (M, W, F)
        ref_size: number of windows
        sample_size: number of points sampled for KS

    Returns:
        scores: np.array
    """
    scores = []
    total = len(windows)

    for t in range(ref_size, total):
        if t % 500 == 0:
            print(f"Processing KS: {t}/{total}")

        ref = windows[t - ref_size:t]
        curr = windows[t:t + ref_size]

        if len(curr) < ref_size:
            break

        # Flatten
        ref_flat = ref.reshape(-1)
        curr_flat = curr.reshape(-1)

        # 🔥 SAMPLING (KEY SPEEDUP)
        if len(ref_flat) > sample_size:
            ref_flat = np.random.choice(ref_flat, sample_size, replace=False)
            curr_flat = np.random.choice(curr_flat, sample_size, replace=False)

        stat, _ = ks_2samp(ref_flat, curr_flat)
        scores.append(stat)

    return np.array(scores)

def compute_ks(X):
    """
    Standard wrapper for KS-based drift detection

    Args:
        X: (M, D) data (raw or latent)

    Returns:
        scores: list of KS statistics over time
    """

    # If you already have a function like compute_ks_scores
    try:
        return compute_ks_scores(X)
    except NameError:
        pass

    # Fallback: simple implementation
    from scipy.stats import ks_2samp
    import numpy as np

    scores = []

    ref_size = 100  # initial reference window
    ref = X[:ref_size]

    for i in range(ref_size, len(X)):
        curr = X[i]

        # Compare distributions feature-wise (simplified)
        stat = np.mean([
            ks_2samp(ref[:, j], [curr[j]]).statistic
            for j in range(X.shape[1])
        ])

        scores.append(stat)

    return np.array(scores)
