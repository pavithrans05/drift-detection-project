from __future__ import annotations

from typing import Optional

import numpy as np


def _resolve_gamma(
    x: np.ndarray,
    y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    if gamma is not None:
        return gamma

    combined = np.vstack([x, y]).astype(np.float32)
    if len(combined) < 2:
        return 1.0

    sample_size = min(len(combined), 256)
    indices = np.random.choice(len(combined), size=sample_size, replace=False)
    sample = combined[indices]

    distances = []
    for i in range(sample_size - 1):
        diff = sample[i + 1 :] - sample[i]
        distances.extend(np.sum(diff * diff, axis=1))

    distances = np.asarray(distances, dtype=np.float32)
    median_distance = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    return 1.0 / (2.0 * median_distance)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    gamma = _resolve_gamma(x, y, gamma)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    distances = x_sq + y_sq - 2.0 * np.dot(x, y.T)
    return np.exp(-gamma * np.clip(distances, a_min=0.0, a_max=None))


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """
    Biased MMD^2 estimate with RBF kernel.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Both x and y must contain at least one sample")

    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)

    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def mmd_drift_series(
    embeddings: np.ndarray,
    ref_size: int = 50,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Compute an MMD score series using adjacent reference/current windows.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (N, latent_dim), got {embeddings.shape}"
        )

    scores = []
    total = len(embeddings)

    for t in range(ref_size, total):
        ref = embeddings[t - ref_size : t]
        curr = embeddings[t : t + ref_size]

        if len(curr) < ref_size:
            break

        scores.append(compute_mmd(ref, curr, gamma=gamma))

    return np.asarray(scores, dtype=np.float32)
