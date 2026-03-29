import numpy as np

from config import DATA_PATHS
from utils.loaders import load_gas_dataset
from utils.pipeline import prepare_data

from drift_detection.ks_test import ks_drift_series
from drift_detection.adwin import adwin_drift
from drift_detection.threshold import detect_drift
from drift_detection.persistence import apply_persistence

from utils.visualization import (
    plot_ks_drift,
    plot_ks_with_true_drift,
    plot_ks_with_adwin,
    plot_all
)


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 50
REF_SIZE = 50
PERCENTILE = 95
PERSISTENCE = 5


# =========================
# LOAD DATA
# =========================
print("\nLoading Gas Dataset...")

X, y, meta = load_gas_dataset(DATA_PATHS["gas"])
true_drift_points = meta["drift_points"]

print("Data shape:", X.shape)
print("True drift points:", true_drift_points)


# =========================
# PIPELINE (Module 2)
# =========================
print("\nPreparing data...")

windows, flat, _ = prepare_data(X, window_size=WINDOW_SIZE)

print("Windows shape:", windows.shape)


# =========================
# KS DRIFT DETECTION
# =========================
print("\nRunning KS Drift Detection...")

ks_scores = ks_drift_series(windows, ref_size=REF_SIZE)

print("KS scores length:", len(ks_scores))


# =========================
# KS THRESHOLDING
# =========================
ks_drift_points, threshold = detect_drift(ks_scores, percentile=PERCENTILE)

print("\nKS Threshold:", threshold)
print("Raw KS detections:", len(ks_drift_points))


# =========================
# KS PERSISTENCE FILTER (KEY STEP)
# =========================
ks_persistent = apply_persistence(ks_drift_points, min_consecutive=PERSISTENCE)

print("\nAfter persistence filtering:")
print("KS persistent detections:", len(ks_persistent))
print("Persistent points (first few):", ks_persistent[:10])


# =========================
# ADWIN DRIFT DETECTION
# =========================
print("\nRunning ADWIN Drift Detection...")

adwin_points = adwin_drift(X)

print("ADWIN detections:", len(adwin_points))
print("First few ADWIN points:", adwin_points[:10])


# =========================
# VISUALIZATION
# =========================
print("\nPlotting results...")

offset = REF_SIZE

# 1. KS curve
plot_ks_drift(ks_scores, title="KS Drift Score (Gas Dataset)")

# 2. KS vs Ground Truth
plot_ks_with_true_drift(
    ks_scores,
    true_drift_points,
    offset,
    title="KS vs Ground Truth"
)

# 3. KS + ADWIN
plot_ks_with_adwin(
    ks_scores,
    adwin_points,
    offset,
    title="KS + ADWIN"
)

# 4. Combined
plot_all(
    ks_scores,
    true_drift_points,
    adwin_points,
    offset
)


# =========================
# FINAL SUMMARY
# =========================
print("\n========== FINAL SUMMARY ==========")
print("True drift points:", len(true_drift_points))
print("Raw KS detections:", len(ks_drift_points))
print("KS persistent detections:", len(ks_persistent))
print("ADWIN detections:", len(adwin_points))
print("===================================")