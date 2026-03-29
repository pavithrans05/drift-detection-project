import numpy as np
import os


def get_max_feature_index(file_path):
    """
    Scan file to determine maximum feature index
    """
    max_idx = 0

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            for item in parts[1:]:
                idx, _ = item.split(":")
                max_idx = max(max_idx, int(idx))

    return max_idx


def parse_line(line, num_features):
    """
    Parse one row of gas dataset
    Format: label f1:v1 f2:v2 ...
    """
    parts = line.strip().split()

    features = np.zeros(num_features)

    for item in parts[1:]:
        idx, value = item.split(":")
        features[int(idx) - 1] = float(value)

    return features


def load_gas_batches(data_path):
    """
    Load all gas dataset batches
    """
    batches = []

    for i in range(1, 11):
        file_path = os.path.join(data_path, f"batch{i}.dat")

        num_features = get_max_feature_index(file_path)

        batch_data = []

        with open(file_path, "r") as f:
            for line in f:
                parsed = parse_line(line, num_features)
                batch_data.append(parsed)

        batches.append(np.array(batch_data))

    return batches