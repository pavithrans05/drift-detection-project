"""
Central configuration file for Drift Detection Project
"""

import torch

# =========================================================
# 🧠 GENERAL SETTINGS
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# =========================================================
# 📦 DATASET CONFIGURATION
# =========================================================
DATA_CONFIG = {
    "gas": {
        "path": "data/gas/",
        "window_size": 50,
        "stride": 1,
        "flatten": True
    },
    "intel": {
        "path": "data/intel/",
        "window_size": 50,
        "stride": 5,   # larger stride for big dataset
        "flatten": True
    },
    "nasa": {
        "path": "data/nasa/",
        "window_size": 30,
        "stride": 1,
        "flatten": True
    },
    "swat": {
        "path": "data/swat/",
        "window_size": 30,
        "stride": 1,
        "flatten": True
    }
}


# =========================================================
# 🔄 PIPELINE CONFIG
# =========================================================
PIPELINE_CONFIG = {
    "normalize": True,
    "scaler": "standard",   # options: standard, minmax
    "batch_size": 512
}


# =========================================================
# 🧪 DRIFT DETECTION CONFIG
# =========================================================

# --- KS TEST ---
KS_CONFIG = {
    "sample_size": 5000,     # sampling for speed
    "reference_size": 100,   # initial stable window
}


# --- THRESHOLDING ---
THRESHOLD_CONFIG = {
    "percentile": 95,        # for KS thresholding
}


# --- PERSISTENCE FILTER ---
PERSISTENCE_CONFIG = {
    "min_consecutive": 5,    # must persist for 5 windows
    "merge_distance": 10     # merge nearby detections
}


# --- ADWIN ---
ADWIN_CONFIG = {
    "delta": 0.002           # sensitivity (lower = stricter)
}


# =========================================================
# 🤖 AUTOENCODER CONFIG (MODULE 4)
# =========================================================
AE_CONFIG = {
    "input_dim": None,       # auto-set during runtime
    "latent_dim": 16,
    "hidden_dims": [1024, 256, 64],

    "batch_size": 128,
    "epochs": 20,
    "lr": 1e-3,

    "device": DEVICE,

    # Regularization
    "weight_decay": 1e-5,

    # Optional improvements
    "dropout": 0.0,
    "batch_norm": False
}


# =========================================================
# 📊 EXPERIMENT SETTINGS
# =========================================================
EXPERIMENT_CONFIG = {
    "save_results": True,
    "output_dir": "outputs/",
    "plot": True,
    "verbose": True
}


# =========================================================
# 🧾 LOGGING
# =========================================================
LOG_CONFIG = {
    "log_interval": 1,   # epochs
    "save_model": True,
    "model_dir": "saved_models/"
}