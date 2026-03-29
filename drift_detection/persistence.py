import numpy as np


def apply_persistence(drift_points, min_consecutive=5):
    """
    Keep only persistent drift points

    Args:
        drift_points: array of indices
        min_consecutive: minimum consecutive detections

    Returns:
        filtered_points
    """
    if len(drift_points) == 0:
        return []

    drift_points = np.sort(drift_points)

    groups = []
    current_group = [drift_points[0]]

    for i in range(1, len(drift_points)):
        if drift_points[i] == drift_points[i - 1] + 1:
            current_group.append(drift_points[i])
        else:
            groups.append(current_group)
            current_group = [drift_points[i]]

    groups.append(current_group)

    # Keep only persistent groups
    filtered = [g[0] for g in groups if len(g) >= min_consecutive]

    return np.array(filtered)