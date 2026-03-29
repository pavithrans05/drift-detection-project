import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.windowing import create_windows, flatten_windows


def normalize_data(X):
    """
    Standardize data (zero mean, unit variance)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce memory usage
    X_scaled = X_scaled.astype(np.float32)

    return X_scaled, scaler


def prepare_data(X, window_size=50, stride=1, flatten=True, normalize=True):
    """
    Full pipeline:
    X → normalize → window → flatten

    Args:
        X: (N, F)
        window_size: int
        stride: int
        flatten: bool
        normalize: bool

    Returns:
        windows: (M, W, F)
        flat_windows: (M, W*F) or None
        scaler: fitted scaler (or None)
    """

    # ✅ Normalize (IMPORTANT FIX)
    if normalize:
        X, scaler = normalize_data(X)
    else:
        scaler = None
        X = X.astype(np.float32)

    # Create windows
    windows = create_windows(X, window_size, stride)

    # Flatten if needed
    if flatten:
        flat = flatten_windows(windows).astype(np.float32)
        return windows, flat, scaler

    return windows, None, scaler


def prepare_data_streaming(X, window_size=50, stride=1, normalize=True):
    """
    Generator-based pipeline for large datasets

    Yields:
        window (W, F)
    """

    if normalize:
        X, _ = normalize_data(X)
    else:
        X = X.astype(np.float32)

    for i in range(0, len(X) - window_size + 1, stride):
        yield X[i:i + window_size]

# =========================================================
# 🚀 HIGH-LEVEL PIPELINE (USED IN EXPERIMENTS)
# =========================================================

from utils.loaders.gas_loader import load_gas_data
from utils.loaders.intel_loader import load_intel_data
from utils.loaders.nasa_loader import load_nasa_data
from utils.loaders.swat_loader import load_swat_data


def run_pipeline(dataset_name, config):
    """
    Complete pipeline:
    Load → Prepare (normalize + window + flatten)

    Returns:
        X_flat: (M, W*F)
        y
        meta
    """

    # =============================
    # 1. LOAD DATA
    # =============================
    if dataset_name == "gas":
        X, y, meta = load_gas_data(config["path"])
    elif dataset_name == "intel":
        X, y, meta = load_intel_data(config["path"])
    elif dataset_name == "nasa":
        X, y, meta = load_nasa_data(config["path"])
    elif dataset_name == "swat":
        X, y, meta = load_swat_data(config["path"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # =============================
    # 2. PREPARE DATA (your existing pipeline)
    # =============================
    windows, flat_windows, scaler = prepare_data(
        X,
        window_size=config["window_size"],
        stride=config["stride"],
        flatten=config.get("flatten", True),
        normalize=config.get("normalize", True)
    )

    return flat_windows, y, meta