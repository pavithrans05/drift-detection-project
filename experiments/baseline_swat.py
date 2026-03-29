import os
import numpy as np

from config import DATA_PATHS
from utils.loaders import load_swat_dataset
from utils.pipeline import prepare_data

from drift_detection.ks_test import ks_drift_series
from drift_detection.threshold import detect_drift
from drift_detection.persistence import apply_persistence
from drift_detection.adwin import adwin_drift

from utils.visualization import plot_ks_with_adwin, plot_ks_drift


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 50
REF_SIZE = 50
PERCENTILE = 95
PERSISTENCE = 5
MAX_SAMPLES = 100000   # limit for speed


print("\n=== SWaT BASELINE ===")


# =========================
# LOAD DATA (OPTION 1)
# =========================
normal_path = os.path.join(DATA_PATHS["swat"], "normal.csv")
attack_path = os.path.join(DATA_PATHS["swat"], "attack.csv")

X, y, _ = load_swat_dataset(normal_path, attack_path)

# Reduce size for faster execution
X = X[:MAX_SAMPLES]
y = y[:MAX_SAMPLES]

print("Data shape:", X.shape)
print("Attack labels distribution:", np.unique(y, return_counts=True))


# =========================
# PIPELINE
# =========================
windows, _, _ = prepare_data(X, window_size=WINDOW_SIZE)

print("Windows shape:", windows.shape)


# =========================
# KS DRIFT DETECTION
# =========================
print("\nRunning KS Drift Detection...")

ks_scores = ks_drift_series(windows, ref_size=REF_SIZE)

ks_points, threshold = detect_drift(ks_scores, percentile=PERCENTILE)
ks_persistent = apply_persistence(ks_points, min_consecutive=PERSISTENCE)

print("\nKS Threshold:", threshold)
print("Raw KS detections:", len(ks_points))
print("KS persistent detections:", len(ks_persistent))
print("First few KS persistent points:", ks_persistent[:10])


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

# KS curve
plot_ks_drift(ks_scores, title="SWaT KS Drift Score")

# KS + ADWIN
plot_ks_with_adwin(
    ks_scores,
    adwin_points,
    offset,
    title="SWaT KS + ADWIN (Attack Detection)"
)


# =========================
# OPTIONAL: ATTACK ALIGNMENT CHECK
# =========================
print("\nAnalyzing alignment with attack labels...")

attack_indices = np.where(y == 1)[0]

print("Number of attack points:", len(attack_indices))
print("First few attack indices:", attack_indices[:10])


# =========================
# FINAL SUMMARY
# =========================
print("\n========== SWaT SUMMARY ==========")
print("Total samples:", len(X))
print("KS persistent detections:", len(ks_persistent))
print("ADWIN detections:", len(adwin_points))
print("Attack points:", len(attack_indices))
print("==================================")