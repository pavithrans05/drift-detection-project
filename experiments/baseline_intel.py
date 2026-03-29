import os
import numpy as np

from config import DATA_PATHS
from utils.loaders import load_intel_dataset
from utils.pipeline import prepare_data

from drift_detection.ks_test import ks_drift_series
from drift_detection.threshold import detect_drift
from drift_detection.persistence import apply_persistence
from drift_detection.adwin import adwin_drift

from utils.visualization import plot_ks_drift, plot_ks_with_adwin


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 50
REF_SIZE = 50
PERCENTILE = 95
PERSISTENCE = 5


print("\n=== INTEL BASELINE ===")

# Load data
intel_file = os.path.join(DATA_PATHS["intel"], "data.txt")
X, _, _ = load_intel_dataset(intel_file)

# Use subset (IMPORTANT)
X = X[:50000]

print("Data shape:", X.shape)

# Pipeline
windows, _, _ = prepare_data(X, window_size=WINDOW_SIZE)

# KS
ks_scores = ks_drift_series(windows, ref_size=REF_SIZE)
ks_points, _ = detect_drift(ks_scores, percentile=PERCENTILE)
ks_persistent = apply_persistence(ks_points, min_consecutive=PERSISTENCE)

# ADWIN
adwin_points = adwin_drift(X)

# Results
print("KS persistent detections:", len(ks_persistent))
print("ADWIN detections:", len(adwin_points))

# Plot
offset = REF_SIZE

plot_ks_drift(ks_scores, title="Intel KS Drift Score")

plot_ks_with_adwin(
    ks_scores,
    adwin_points,
    offset,
    title="Intel KS + ADWIN"
)