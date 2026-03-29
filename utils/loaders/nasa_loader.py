import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_nasa_dataset(file_path):
    cols = ["unit", "cycle"] + \
           [f"op{i}" for i in range(1,4)] + \
           [f"s{i}" for i in range(1,22)]

    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.iloc[:, :26]
    df.columns = cols

    df = df.sort_values(by=["unit", "cycle"])

    # RUL
    max_cycle = df.groupby("unit")["cycle"].max()
    df["RUL"] = df.apply(
        lambda row: max_cycle[row["unit"]] - row["cycle"],
        axis=1
    )

    features = [f"s{i}" for i in range(1,22)]
    X = df[features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, df["RUL"].values, {"units": df["unit"].values}

def load_nasa_data(path):
    return load_nasa_dataset(path)