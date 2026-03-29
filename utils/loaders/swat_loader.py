import pandas as pd
import numpy as np


def load_swat_dataset(normal_path, attack_path):
    """
    Robust SWaT loader handling column mismatches
    """

    # =========================
    # LOAD NORMAL
    # =========================
    df_normal = pd.read_csv(normal_path)
    df_normal = df_normal.ffill()

    # Drop non-feature columns
    drop_cols = [col for col in df_normal.columns if "timestamp" in col.lower()]
    df_normal = df_normal.drop(columns=drop_cols, errors="ignore")

    X_normal = df_normal.values
    y_normal = np.zeros(len(X_normal))


    # =========================
    # LOAD ATTACK
    # =========================
    df_attack = pd.read_csv(attack_path)
    df_attack = df_attack.ffill()

    print("\n[DEBUG] Attack columns:", df_attack.columns.tolist())

    # Find label column automatically
    label_col = None
    for col in df_attack.columns:
        if "attack" in col.lower():
            label_col = col
            break

    if label_col is not None:
        print("[DEBUG] Using label column:", label_col)

        y_attack = df_attack[label_col].astype(str).str.lower().str.contains("attack").astype(int).values
        df_attack = df_attack.drop(columns=[label_col], errors="ignore")
    else:
        print("[WARNING] No label column found — assigning all as attack")
        y_attack = np.ones(len(df_attack))

    # Drop timestamp columns
    drop_cols = [col for col in df_attack.columns if "timestamp" in col.lower()]
    df_attack = df_attack.drop(columns=drop_cols, errors="ignore")

    X_attack = df_attack.values


    # =========================
    # ALIGN COLUMNS (CRITICAL FIX)
    # =========================
    min_cols = min(X_normal.shape[1], X_attack.shape[1])

    X_normal = X_normal[:, :min_cols]
    X_attack = X_attack[:, :min_cols]


    # =========================
    # COMBINE
    # =========================
    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    return X, y, {}


def load_swat_data(path):
    return load_swat_dataset(path)