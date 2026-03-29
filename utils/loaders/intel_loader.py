import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_intel_dataset(file_path):
    # Robust reading
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",     # IMPORTANT
        on_bad_lines="skip"  # skip corrupted rows
    )

    # Assign proper column names
    df.columns = [
        "date", "time", "epoch", "moteid",
        "temperature", "humidity", "light", "voltage"
    ]

    # Combine timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")

    # Drop invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Sort
    df = df.sort_values(by="timestamp")

    # Select features
    features = ["temperature", "humidity", "light", "voltage"]

    df = df[features]

    # Handle missing values
    df = df.interpolate().dropna()

    X = df.values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, None, {"type": "stream"}

def load_intel_data(path):
    """
    Wrapper to standardize loader interface
    """
    return load_intel_dataset(path)