import os
import numpy as np

from config import DATA_PATHS
from utils.loaders import load_nasa_dataset
from utils.pipeline import prepare_data

from drift_detection.ks_test import ks_drift_series
from drift_detection.threshold import detect_drift
from drift_detection.persistence import apply_persistence

from utils.visualization import plot_ks_drift


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 50
REF_SIZE = 50
PERCENTILE = 95
PERSISTENCE = 5


print("\n=== NASA BASELINE ===")

# Load dataset (FD001 recommended)
train_file = os.path.join(DATA_PATHS["nasa"], "train_FD001.txt")

X, y, meta = load_nasa_dataset(train_file)

print("Data shape:", X.shape)

# Pipeline
windows, _, _ = prepare_data(X, window_size=WINDOW_SIZE)

# KS
ks_scores = ks_drift_series(windows, ref_size=REF_SIZE)
ks_points, _ = detect_drift(ks_scores, percentile=PERCENTILE)
ks_persistent = apply_persistence(ks_points, min_consecutive=PERSISTENCE)

# Results
print("KS persistent detections:", len(ks_persistent))

# Plot
plot_ks_drift(ks_scores, title="NASA KS Drift (Gradual Degradation)")