import os
import numpy as np

from config import DATA_PATHS
from utils.loaders import (
    load_gas_dataset,
    load_intel_dataset,
    load_nasa_dataset,
    load_swat_dataset
)

def check_data(X, name):
    print(f"\n--- {name} ---")
    print("Shape:", X.shape)
    print("NaNs:", np.isnan(X).sum())
    print("Min:", np.min(X))
    print("Max:", np.max(X))


# =======================
# 1. GAS DATASET
# =======================
X, y, meta = load_gas_dataset(DATA_PATHS["gas"])
check_data(X, "Gas Dataset")
print("Meta:", meta)


# =======================
# 2. INTEL DATASET
# =======================
intel_file = os.path.join(DATA_PATHS["intel"], "data.txt")  # adjust if needed

X, y, meta = load_intel_dataset(intel_file)
check_data(X, "Intel Dataset")


# =======================
# 3. NASA DATASET
# =======================
nasa_file = os.path.join(DATA_PATHS["nasa"], "train_FD001.txt")

X, y, meta = load_nasa_dataset(nasa_file)
check_data(X, "NASA Dataset")

print("RUL shape:", y.shape)


# =======================
# 4. SWAT DATASET
# =======================
normal_path = os.path.join(DATA_PATHS["swat"], "normal.csv")
attack_path = os.path.join(DATA_PATHS["swat"], "attack.csv")

X, y, meta = load_swat_dataset(normal_path, attack_path)
check_data(X, "SWaT Dataset")

print("Labels distribution:", np.unique(y, return_counts=True))