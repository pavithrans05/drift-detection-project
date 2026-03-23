import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_swat_dataset(normal_path, attack_path):
    normal_df = pd.read_csv(normal_path)
    attack_df = pd.read_csv(attack_path)

    df = pd.concat([normal_df, attack_df])

    df.columns = df.columns.str.strip()

    df["label"] = df["Normal/Attack"].map({
        "Normal": 0,
        "Attack": 1
    })

    df = df.drop(columns=["Timestamp", "Normal/Attack"])

    df = df.fillna(method="ffill")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, {"type": "attack_detection"}